"""ArXiv ingestion, adaptive retrieval, grounded answers, and judge metrics."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import functools
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("RAG_DATABASE_PATH", ROOT / "rag_memory.sqlite3"))
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", ROOT / "chroma_data"))
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 160
FEATURES = ("semantic", "recency", "feedback")
CITATION_RE = re.compile(r"\[(arxiv:\d{4}\.\d{4,5}:\d+)\]")
DEFAULT_WEIGHTS = {"semantic": 0.65, "recency": 0.2, "feedback": 0.15}


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    title: str
    published: str
    text: str


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    features: dict[str, float]


def connect_database(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            published TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS judgments (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevant INTEGER NOT NULL CHECK (relevant IN (0, 1)),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS weights (
            name TEXT PRIMARY KEY,
            value REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT OR IGNORE INTO weights (name, value) VALUES (?, ?)",
        DEFAULT_WEIGHTS.items(),
    )
    connection.commit()
    return connection


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _request(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SelfImprovingRAG/1.0 (research prototype)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstream request failed ({error.code}): {details[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach upstream service: {error.reason}") from error


def search_arxiv(query: str, limit: int = 10) -> list[dict[str, str]]:
    if not query.strip():
        raise ValueError("An arXiv search query is required.")
    if not 1 <= limit <= 50:
        raise ValueError("Paper limit must be between 1 and 50.")
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query.strip()}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    root = ET.fromstring(_request(f"{ARXIV_API}?{params}"))
    papers: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=ARXIV_NS)).rstrip("/")
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
        if not arxiv_id:
            continue
        pdf_url = next(
            (
                link.attrib["href"]
                for link in entry.findall("atom:link", ARXIV_NS)
                if link.attrib.get("title") == "pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": re.sub(r"\s+", " ", entry.findtext("atom:title", "", ARXIV_NS)).strip(),
                "abstract": re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ARXIV_NS)).strip(),
                "published": entry.findtext("atom:published", "", ARXIV_NS),
                "pdf_url": pdf_url,
            }
        )
    return papers


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF extraction requires PyMuPDF. Install requirements.txt.") from error
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text("text") for page in document)
    except Exception as error:
        raise RuntimeError(f"Could not extract text from arXiv PDF: {error}") from error


def _vector_collection() -> Any:
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("Vector search requires ChromaDB. Install requirements.txt.") from error
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH)).get_or_create_collection(
        "arxiv_chunks", metadata={"hnsw:space": "cosine"}
    )


@functools.lru_cache(maxsize=1)
def _embedder() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic search requires sentence-transformers. Install requirements.txt."
        ) from error
    return SentenceTransformer(MODEL_NAME)


def ingest_arxiv(query: str, limit: int = 10, connection: sqlite3.Connection | None = None) -> dict[str, int]:
    papers = search_arxiv(query, limit)
    if not papers:
        raise ValueError("arXiv returned no papers for this query.")
    own_connection = connection is None
    connection = connection or connect_database()
    collection = _vector_collection()
    embedder = _embedder()
    paper_count = chunk_count = 0
    try:
        for paper in papers:
            pdf = _request(paper["pdf_url"])
            text = extract_pdf_text(pdf)
            chunks = split_text(text)
            if not chunks:
                chunks = split_text(paper["abstract"])
            if not chunks:
                continue
            ids = [f"arxiv:{paper['arxiv_id']}:{index}" for index in range(1, len(chunks) + 1)]
            embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=[
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "published": paper["published"],
                    }
                    for _ in chunks
                ],
            )
            with connection:
                connection.execute(
                    """
                    INSERT INTO papers (arxiv_id, title, abstract, published, pdf_url, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title=excluded.title, abstract=excluded.abstract,
                        published=excluded.published, pdf_url=excluded.pdf_url,
                        indexed_at=excluded.indexed_at
                    """,
                    (
                        paper["arxiv_id"],
                        paper["title"],
                        paper["abstract"],
                        paper["published"],
                        paper["pdf_url"],
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper["arxiv_id"],))
                connection.executemany(
                    "INSERT INTO chunks (id, paper_id, chunk_index, content) VALUES (?, ?, ?, ?)",
                    [
                        (chunk_id, paper["arxiv_id"], index, content)
                        for index, (chunk_id, content) in enumerate(zip(ids, chunks), start=1)
                    ],
                )
            paper_count += 1
            chunk_count += len(chunks)
    finally:
        if own_connection:
            connection.close()
    if paper_count == 0:
        raise ValueError("No readable paper text was returned by arXiv.")
    return {"papers": paper_count, "chunks": chunk_count}


def get_weights(connection: sqlite3.Connection) -> dict[str, float]:
    rows = connection.execute("SELECT name, value FROM weights").fetchall()
    values = {row["name"]: float(row["value"]) for row in rows}
    return {name: values.get(name, DEFAULT_WEIGHTS[name]) for name in FEATURES}


def _recency(published: str) -> float:
    try:
        date = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - date).total_seconds() / 86400)
        return math.exp(-age_days / (365.25 * 3))
    except (ValueError, TypeError):
        return 0.0


def retrieve(question: str, connection: sqlite3.Connection, limit: int = 5) -> list[RankedChunk]:
    if not question.strip():
        raise ValueError("A question is required.")
    if not 1 <= limit <= 20:
        raise ValueError("Result limit must be between 1 and 20.")
    collection = _vector_collection()
    if collection.count() == 0:
        return []
    embedding = _embedder().encode([question], normalize_embeddings=True).tolist()[0]
    candidates = collection.query(
        query_embeddings=[embedding],
        n_results=min(max(limit * 5, 20), collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    ids = candidates["ids"][0]
    distances = candidates["distances"][0]
    docs = candidates["documents"][0]
    metadata = candidates["metadatas"][0]
    weights = get_weights(connection)
    ranked: list[RankedChunk] = []
    for chunk_id, distance, content, meta in zip(ids, distances, docs, metadata):
        row = connection.execute(
            "SELECT AVG(relevant) AS signal FROM judgments WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        feedback = float(row["signal"]) if row["signal"] is not None else 0.5
        features = {
            "semantic": max(0.0, min(1.0, 1.0 - float(distance))),
            "recency": _recency(meta.get("published", "")),
            "feedback": feedback,
        }
        score = sum(weights[name] * features[name] for name in FEATURES)
        chunk = Chunk(
            id=chunk_id,
            paper_id=meta["arxiv_id"],
            title=meta.get("title", ""),
            published=meta.get("published", ""),
            text=content,
        )
        ranked.append(RankedChunk(chunk, score, features))
    return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


def update_weights(
    judged_chunks: Sequence[tuple[RankedChunk, bool]],
    connection: sqlite3.Connection,
    learning_rate: float = 0.25,
) -> dict[str, float]:
    positive = [item.features for item, relevant in judged_chunks if relevant]
    negative = [item.features for item, relevant in judged_chunks if not relevant]
    if not positive or not negative:
        return get_weights(connection)
    separation = {
        name: sum(row[name] for row in positive) / len(positive)
        - sum(row[name] for row in negative) / len(negative)
        for name in FEATURES
    }
    weights = get_weights(connection)
    logits = {
        name: math.log(max(weights[name], 1e-8)) + learning_rate * separation[name]
        for name in FEATURES
    }
    maximum = max(logits.values())
    exponentials = {name: math.exp(value - maximum) for name, value in logits.items()}
    total = sum(exponentials.values())
    updated = {name: value / total for name, value in exponentials.items()}
    with connection:
        connection.executemany(
            "INSERT INTO weights (name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
            updated.items(),
        )
    return updated


def _api_completion(messages: list[dict[str, str]], json_mode: bool = False) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY to generate answers and run the LLM judge.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload: dict[str, Any] = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    raw = _request_openai(request)
    try:
        value = json.loads(raw)
        content = value["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Chat API returned an unexpected response shape.") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Chat API returned an empty answer.")
    return content.strip()


def _request_openai(request: urllib.request.Request) -> str:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat API returned HTTP {error.code}: {details[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach chat API: {error.reason}") from error


def generate_answer(question: str, chunks: Sequence[RankedChunk]) -> str:
    if not chunks:
        return "I couldn't find supporting passages in the indexed arXiv papers."
    context = "\n\n".join(
        f"[{item.chunk.id}] ({item.chunk.title})\n{item.chunk.text}"
        for item in chunks
    )
    return _api_completion(
        [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied arXiv passages. "
                    "Cite claims with the exact chunk marker provided, e.g. "
                    "[arxiv:2401.12345:2]. Never invent a citation. If evidence is "
                    "insufficient, say so."
                ),
            },
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
    )


def validate_citations(
    answer: str,
    allowed_chunk_ids: set[str],
) -> tuple[str, list[str], int]:
    found = CITATION_RE.findall(answer)
    invalid = [citation for citation in found if citation not in allowed_chunk_ids]
    cleaned = answer
    for citation in invalid:
        cleaned = re.sub(r"[ \t]*" + re.escape(f"[{citation}]"), "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    valid = list(dict.fromkeys(citation for citation in found if citation in allowed_chunk_ids))
    return cleaned, valid, len(invalid)


def judge_answer(
    question: str,
    answer: str,
    chunks: Sequence[RankedChunk],
    citations: Sequence[str],
) -> dict[str, Any]:
    prompt_data = {
        "question": question,
        "answer": answer,
        "chunks": [
            {"chunk_id": item.chunk.id, "text": item.chunk.text}
            for item in chunks
        ],
        "citations": list(citations),
    }
    raw = _api_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are a binary evidence judge. Return JSON only with exactly "
                    "two arrays: chunks [{chunk_id, relevant}] and citations "
                    "[{marker, supported}]. Use booleans only. Judge chunk relevance "
                    "to the question, and whether each cited chunk supports the "
                    "answer. Do not output scores or metrics."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
        ],
        json_mode=True,
    )
    try:
        verdicts = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("Judge returned invalid JSON.") from error
    expected_ids = {item.chunk.id for item in chunks}
    expected_citations = set(citations)
    chunk_verdicts = verdicts.get("chunks")
    citation_verdicts = verdicts.get("citations")
    if not isinstance(chunk_verdicts, list) or not isinstance(citation_verdicts, list):
        raise RuntimeError("Judge response must contain chunks and citations arrays.")
    if len(chunk_verdicts) != len(expected_ids) or len(citation_verdicts) != len(expected_citations):
        raise RuntimeError("Judge response must contain exactly one verdict per item.")
    normalized_chunks: dict[str, bool] = {}
    for verdict in chunk_verdicts:
        if (
            isinstance(verdict, dict)
            and verdict.get("chunk_id") in expected_ids
            and type(verdict.get("relevant")) is bool
        ):
            normalized_chunks[verdict["chunk_id"]] = verdict["relevant"]
    normalized_citations: dict[str, bool] = {}
    for verdict in citation_verdicts:
        if (
            isinstance(verdict, dict)
            and verdict.get("marker") in expected_citations
            and type(verdict.get("supported")) is bool
        ):
            normalized_citations[verdict["marker"]] = verdict["supported"]
    if set(normalized_chunks) != expected_ids or set(normalized_citations) != expected_citations:
        raise RuntimeError("Judge response omitted or duplicated a required verdict.")
    return {
        "chunks": [{"chunk_id": key, "relevant": value} for key, value in normalized_chunks.items()],
        "citations": [{"marker": key, "supported": value} for key, value in normalized_citations.items()],
    }


def compute_metrics(
    verdicts: dict[str, Any],
    retrieved: Sequence[RankedChunk],
    cited_markers: Sequence[str],
    invalid_citations: int,
) -> dict[str, float | int]:
    relevant_ids = {
        item["chunk_id"] for item in verdicts["chunks"] if item["relevant"]
    }
    retrieved_ids = {item.chunk.id for item in retrieved}
    true_chunks = len(relevant_ids & retrieved_ids)
    chunk_precision = true_chunks / len(retrieved_ids) if retrieved_ids else 0.0
    chunk_recall = true_chunks / len(relevant_ids) if relevant_ids else 0.0
    chunk_f1 = (
        2 * chunk_precision * chunk_recall / (chunk_precision + chunk_recall)
        if chunk_precision + chunk_recall
        else 0.0
    )
    citation_support = {
        item["marker"]: item["supported"] for item in verdicts["citations"]
    }
    valid_count = len(cited_markers)
    supported_count = sum(citation_support.get(marker, False) for marker in cited_markers)
    citation_precision = supported_count / valid_count if valid_count else 0.0
    citation_recall = supported_count / len(relevant_ids) if relevant_ids else 0.0
    citation_recall = min(citation_recall, 1.0)
    citation_f1 = (
        2 * citation_precision * citation_recall / (citation_precision + citation_recall)
        if citation_precision + citation_recall
        else 0.0
    )
    return {
        "chunk_precision": round(chunk_precision, 4),
        "chunk_recall": round(chunk_recall, 4),
        "chunk_f1": round(chunk_f1, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "citation_f1": round(citation_f1, 4),
        "invalid_citations": invalid_citations,
    }


def answer_question(question: str, connection: sqlite3.Connection, limit: int = 5) -> dict[str, Any]:
    ranked_candidates = retrieve(question, connection, limit=min(limit * 4, 20))
    selected = ranked_candidates[:limit]
    if not selected:
        return {
            "answer": "I couldn't find supporting passages in the indexed arXiv papers.",
            "sources": [],
            "citations": [],
            "metrics": {
                "chunk_precision": 0.0,
                "chunk_recall": 0.0,
                "chunk_f1": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "citation_f1": 0.0,
                "invalid_citations": 0,
            },
            "weights": get_weights(connection),
        }
    answer = generate_answer(question, selected)
    allowed_ids = {item.chunk.id for item in selected}
    clean_answer, citations, invalid_count = validate_citations(answer, allowed_ids)
    verdicts = judge_answer(question, clean_answer, ranked_candidates, citations)
    metrics = compute_metrics(verdicts, ranked_candidates, citations, invalid_count)
    with connection:
        connection.executemany(
            "INSERT INTO judgments (question, chunk_id, relevant, created_at) VALUES (?, ?, ?, ?)",
            [
                (question, item["chunk_id"], int(item["relevant"]), datetime.now(timezone.utc).isoformat())
                for item in verdicts["chunks"]
            ],
        )
    weights = update_weights(
        [
            (item, next(v["relevant"] for v in verdicts["chunks"] if v["chunk_id"] == item.chunk.id))
            for item in ranked_candidates
        ],
        connection,
    )
    with connection:
        connection.execute(
            "INSERT INTO evaluations (question, metrics_json, weights_json, created_at) VALUES (?, ?, ?, ?)",
            (question, json.dumps(metrics), json.dumps(weights), datetime.now(timezone.utc).isoformat()),
        )
    citation_verdicts = {
        item["marker"]: item["supported"] for item in verdicts["citations"]
    }
    sources = [
        {
            "citation": item.chunk.id,
            "title": item.chunk.title,
            "arxiv_id": item.chunk.paper_id,
            "published": item.chunk.published,
            "text": item.chunk.text,
            "score": round(item.score, 4),
            "features": item.features,
            "relevant": next(v["relevant"] for v in verdicts["chunks"] if v["chunk_id"] == item.chunk.id),
        }
        for item in selected
    ]
    return {
        "answer": clean_answer,
        "sources": sources,
        "citations": [
            {"marker": marker, "supported": citation_verdicts[marker]}
            for marker in citations
        ],
        "metrics": metrics,
        "weights": weights,
    }


def get_stats(connection: sqlite3.Connection) -> dict[str, Any]:
    papers = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    evaluations = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    return {
        "papers": papers,
        "chunks": chunks,
        "evaluations": evaluations,
        "weights": get_weights(connection),
    }
