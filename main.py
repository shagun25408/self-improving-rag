"""A small, local-first RAG assistant that learns from explicit corrections.

Add .txt or .md files under ``knowledge_base``, then run:

    python main.py ingest
    python main.py ask "What does the handbook say about refunds?"
    python main.py learn "What is the refund window?" "Customers have 30 days."

Set OPENAI_API_KEY to use an OpenAI-compatible chat-completions endpoint.
Without a key, answers are based on retrieved passages and include citations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DATABASE_PATH = Path("rag_memory.sqlite3")
KNOWLEDGE_BASE = Path("knowledge_base")
SUPPORTED_EXTENSIONS = {".md", ".txt"}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    citation: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into whitespace-trimmed, overlapping character chunks."""
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
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def connect_database(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            UNIQUE(source, chunk_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def discover_documents(path: Path) -> tuple[Path, list[Path]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge-base path does not exist: {path}")
    if path.is_file():
        if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Supported files are .md and .txt: {path}")
        return path.parent.resolve(), [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Not a file or directory: {path}")

    files = sorted(
        file.resolve()
        for file in path.rglob("*")
        if file.is_file() and file.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    return path.resolve(), files


def ingest(path: Path, connection: sqlite3.Connection) -> int:
    root, files = discover_documents(path)
    if not files:
        raise ValueError(f"No .md or .txt documents found under {path}")

    added = 0
    with connection:
        for file in files:
            content = file.read_text(encoding="utf-8")
            source = file.relative_to(root).as_posix()
            chunks = split_text(content)
            connection.execute("DELETE FROM documents WHERE source = ?", (source,))
            connection.executemany(
                "INSERT INTO documents (source, chunk_index, content) VALUES (?, ?, ?)",
                [(source, index, chunk) for index, chunk in enumerate(chunks, start=1)],
            )
            added += len(chunks)
    return added


def search(
    question: str, connection: sqlite3.Connection, limit: int = 5
) -> list[SearchResult]:
    query_terms = tokenize(question)
    if not query_terms:
        return []

    rows = connection.execute(
        """
        SELECT source, chunk_index, content FROM documents
        UNION ALL
        SELECT 'learned correction ' || id, 1,
               'Question: ' || question || ' Answer: ' || answer
        FROM feedback
        """
    ).fetchall()
    if not rows:
        return []

    documents = [(dict(row), tokenize(row["content"])) for row in rows]
    document_count = len(documents)
    average_length = sum(len(tokens) for _, tokens in documents) / document_count or 1.0
    document_frequency: dict[str, int] = {}
    for term in set(query_terms):
        document_frequency[term] = sum(
            1 for _, tokens in documents if term in set(tokens)
        )

    results: list[SearchResult] = []
    for row, tokens in documents:
        frequencies: dict[str, int] = {}
        for term in tokens:
            frequencies[term] = frequencies.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_normalizer = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / average_length
            )
            score += inverse_frequency * frequency * 2.5 / length_normalizer

        if score > 0:
            if row["source"].startswith("learned correction "):
                citation = f"learned-correction-{row['source'].rsplit(' ', 1)[-1]}"
            else:
                citation = f"{row['source']}#chunk-{row['chunk_index']}"
            results.append(SearchResult(citation, row["content"], score))

    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def extractive_answer(results: Sequence[SearchResult]) -> str:
    if not results:
        return "I couldn't find relevant information in the indexed documents or saved corrections."
    passages = "\n".join(
        f"- {result.text} [{result.citation}]" for result in results[:3]
    )
    return f"Relevant passages:\n{passages}"


def generate_answer(question: str, results: Sequence[SearchResult]) -> str:
    if not results:
        return extractive_answer(results)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return extractive_answer(results)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[{result.citation}]\n{result.text}" for result in results
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied context. If it does not contain "
                    "the answer, say so. Do not invent citations."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Chat API returned HTTP {error.code}: {details}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach chat API: {error.reason}") from error

    try:
        answer = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError("Chat API returned an unexpected response shape") from error
    if not answer:
        raise RuntimeError("Chat API returned an empty answer")
    citations = ", ".join(f"[{item.citation}]" for item in results)
    return f"{answer}\n\nSources: {citations}"


def save_correction(question: str, answer: str, connection: sqlite3.Connection) -> None:
    if not question.strip() or not answer.strip():
        raise ValueError("Both the question and corrected answer must be non-empty.")
    created_at = datetime.now(timezone.utc).isoformat()
    with connection:
        connection.execute(
            "INSERT INTO feedback (question, answer, created_at) VALUES (?, ?, ?)",
            (question.strip(), answer.strip(), created_at),
        )


def ask(question: str, connection: sqlite3.Connection, top_k: int) -> None:
    results = search(question, connection, limit=top_k)
    print(generate_answer(question, results))


def interactive(connection: sqlite3.Connection) -> None:
    print("Local RAG assistant. Ask a question, or type 'quit' to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            return
        if question.casefold() in {"quit", "exit"}:
            return
        if not question:
            continue

        results = search(question, connection)
        print(f"\nAssistant: {generate_answer(question, results)}")
        feedback = input("Was this useful? [y/n/skip] ").strip().casefold()
        if feedback == "n":
            correction = input("What is the correct answer? ").strip()
            save_correction(question, correction, connection)
            print("Saved. Future searches can retrieve this correction.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A local self-improving RAG assistant")
    commands = parser.add_subparsers(dest="command")

    ingest_parser = commands.add_parser("ingest", help="Index .md and .txt files")
    ingest_parser.add_argument("path", nargs="?", type=Path, default=KNOWLEDGE_BASE)

    ask_parser = commands.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)

    learn_parser = commands.add_parser(
        "learn", help="Save a corrected answer for future retrieval"
    )
    learn_parser.add_argument("question")
    learn_parser.add_argument("answer")

    commands.add_parser("stats", help="Show the number of indexed chunks and corrections")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ask" and args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        with connect_database() as connection:
            if args.command == "ingest":
                count = ingest(args.path, connection)
                print(f"Indexed {count} chunks from {args.path}.")
            elif args.command == "ask":
                ask(args.question, connection, args.top_k)
            elif args.command == "learn":
                save_correction(args.question, args.answer, connection)
                print("Correction saved and available to future searches.")
            elif args.command == "stats":
                chunks = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                corrections = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
                print(f"Indexed chunks: {chunks}\nSaved corrections: {corrections}")
            else:
                interactive(connection)
    except (OSError, sqlite3.Error, UnicodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())