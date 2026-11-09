"""Baseline-versus-adaptive benchmark harness for held-out question paraphrases."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import rag_engine as rag


TRAINING_QUESTION_COUNT = 24
HELD_OUT_COUNT = 12
LEARNING_CYCLES = 3
METRICS = (
    "chunk_precision",
    "chunk_recall",
    "chunk_f1",
    "citation_precision",
    "citation_recall",
    "citation_f1",
)


def _validate_dataset(dataset: Any) -> tuple[list[str], list[str]]:
    if not isinstance(dataset, dict):
        raise ValueError("Evaluation dataset must be a JSON object.")
    training = dataset.get("training_questions")
    held_out = dataset.get("held_out_paraphrases")
    if not isinstance(training, list) or not all(isinstance(item, str) and item.strip() for item in training):
        raise ValueError("'training_questions' must be an array of non-empty strings.")
    if not isinstance(held_out, list) or not all(isinstance(item, str) and item.strip() for item in held_out):
        raise ValueError("'held_out_paraphrases' must be an array of non-empty strings.")
    if len(training) != TRAINING_QUESTION_COUNT:
        raise ValueError(f"Expected exactly {TRAINING_QUESTION_COUNT} training questions; got {len(training)}.")
    if len(held_out) != HELD_OUT_COUNT:
        raise ValueError(f"Expected exactly {HELD_OUT_COUNT} held-out paraphrases; got {len(held_out)}.")
    return [item.strip() for item in training], [item.strip() for item in held_out]


def _judge_one(question: str, connection: sqlite3.Connection, adapt: bool) -> dict[str, float | int]:
    candidates = rag.retrieve(question, connection, limit=20)
    if not candidates:
        raise ValueError(f"No indexed evidence was retrieved for evaluation question: {question}")
    selected = candidates[:5]
    answer = rag.generate_answer(question, selected)
    clean_answer, citations, invalid = rag.validate_citations(
        answer, {item.chunk.id for item in selected}
    )
    verdicts = rag.judge_answer(question, clean_answer, candidates, citations)
    metrics = rag.compute_metrics(verdicts, candidates, citations, invalid)
    with connection:
        connection.executemany(
            "INSERT INTO judgments (question, chunk_id, relevant, created_at) VALUES (?, ?, ?, ?)",
            [
                (question, verdict["chunk_id"], int(verdict["relevant"]), datetime.now(timezone.utc).isoformat())
                for verdict in verdicts["chunks"]
            ],
        )
    if adapt:
        rag.update_weights(
            [
                (item, next(v["relevant"] for v in verdicts["chunks"] if v["chunk_id"] == item.chunk.id))
                for item in candidates
            ],
            connection,
        )
    return metrics


def _mean_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        metric: round(mean(float(row[metric]) for row in rows), 4) if rows else 0.0
        for metric in METRICS
    }


def run_evaluation(dataset: Any) -> dict[str, Any]:
    training, held_out = _validate_dataset(dataset)
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as directory:
        baseline_path = Path(directory) / "baseline.sqlite3"
        proposed_path = Path(directory) / "proposed.sqlite3"
        with closing(rag.connect_database(baseline_path)) as baseline, closing(
            rag.connect_database(proposed_path)
        ) as proposed:
            with baseline:
                baseline.executemany(
                    "UPDATE weights SET value = ? WHERE name = ?",
                    [(1.0, "semantic"), (0.0, "recency"), (0.0, "feedback")],
                )
            train_runs = 0
            for _ in range(LEARNING_CYCLES):
                for question in training:
                    _judge_one(question, proposed, adapt=True)
                    train_runs += 1
            baseline_scores = [_judge_one(question, baseline, adapt=False) for question in held_out]
            proposed_scores = [_judge_one(question, proposed, adapt=True) for question in held_out]
            return {
                "training_questions": len(training),
                "learning_cycles": LEARNING_CYCLES,
                "adaptive_training_judgments": train_runs,
                "held_out_paraphrases": len(held_out),
                "baseline": _mean_metrics(baseline_scores),
                "proposed": _mean_metrics(proposed_scores),
                "final_weights": rag.get_weights(proposed),
            }
