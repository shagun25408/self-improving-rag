"""FastAPI application and local browser interface."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import rag_engine as rag


app = FastAPI(
    title="Self-Improving arXiv RAG",
    description="Evidence-grounded question answering over indexed arXiv papers.",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=300)]
    limit: Annotated[int, Field(ge=1, le=50)] = 10


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    limit: Annotated[int, Field(ge=1, le=10)] = 5


PAGE_PATH = Path(__file__).with_name("static") / "index.html"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@app.get("/api/stats")
def stats() -> dict:
    try:
        with closing(rag.connect_database()) as connection:
            return rag.get_stats(connection)
    except (OSError, sqlite3.Error, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/ingest")
def ingest(body: IngestRequest) -> dict[str, int]:
    try:
        with closing(rag.connect_database()) as connection:
            return rag.ingest_arxiv(body.query, body.limit, connection)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, sqlite3.Error, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/ask")
def ask(body: AskRequest) -> dict:
    try:
        with closing(rag.connect_database()) as connection:
            return rag.answer_question(body.question, connection, body.limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, sqlite3.Error, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
