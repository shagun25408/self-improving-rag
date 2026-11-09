# Self-Improving RAG over arXiv

A local research assistant that indexes recent arXiv PDFs, retrieves evidence using semantic similarity, recency, and learned feedback, and checks answer citations against stored chunks.

## Requirements

- Python 3.11–3.13
- An OpenAI-compatible chat-completions API key for answer generation and judge verdicts
- Internet access to search arXiv and download papers

## Run locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key"
python front.py
```

Open `http://127.0.0.1:8000`. The first ingestion downloads the configured sentence-transformer model and the requested paper PDFs. Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to use another OpenAI-compatible service.

You can also use the CLI:

```powershell
python main.py ingest "retrieval augmented generation" --limit 10
python main.py ask "How do recent papers improve retrieval?"
python main.py stats
```

The SQLite database defaults to `rag_memory.sqlite3`; Chroma's persistent vector store defaults to `chroma_data`. Override them with `RAG_DATABASE_PATH` and `CHROMA_PATH`. The model defaults to `sentence-transformers/all-MiniLM-L6-v2` and can be changed with `EMBEDDING_MODEL`.

## Learning and grounding

The retriever combines cosine-based semantic similarity, an exponential three-year recency signal, and observed chunk relevance feedback. After each answer the judge returns only boolean verdicts for retrieved chunks and cited markers. Code validates the verdict schema, computes chunk and citation precision/recall/F1, strips citation markers that do not resolve to a retrieved stored chunk, counts invalid markers, and updates normalized retrieval weights from the difference between relevant and irrelevant chunk features.

The judge is not asked to emit metric values or ranking scores. If answer generation or judging is unavailable, the request reports an error instead of returning an unverified success-shaped answer.

## Evaluation harness

Create a UTF-8 JSON file with exactly 24 training questions and 12 held-out paraphrases:

```json
{
  "training_questions": ["question 1", "question 2"],
  "held_out_paraphrases": ["paraphrase 1"]
}
```

The example is intentionally abbreviated; the harness enforces the full counts. Index the evaluation corpus first, then run:

```powershell
python main.py evaluate evaluation_dataset.json --output report.json
```

The harness runs the 24-question training set through three online learning cycles for the proposed retriever, compares it with a semantic-only baseline, and reports mean code-computed metrics on the 12 held-out paraphrases. It requires the API key and a corpus that can answer the benchmark questions.

## Docker

```powershell
docker build -t self-improving-rag .
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY="your-key" `
  -v rag-data:/app/chroma_data `
  -v rag-state:/app/rag_state `
  self-improving-rag
```

For a persistent database in Docker, also set `RAG_DATABASE_PATH=/app/rag_state/rag_memory.sqlite3`.

## API

- `GET /api/health`
- `GET /api/stats`
- `POST /api/ingest` with `{"query":"...", "limit":10}`
- `POST /api/ask` with `{"question":"...", "limit":5}`
- Interactive OpenAPI docs: `/docs`
