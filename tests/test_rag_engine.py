from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation
import rag_engine as rag


class RagEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.connection = rag.connect_database(Path(self.temp_dir.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.connection.close()
        self.temp_dir.cleanup()

    def test_split_text_overlaps_and_preserves_content(self) -> None:
        chunks = rag.split_text("alpha beta gamma delta epsilon", size=16, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() == chunk for chunk in chunks))
        self.assertIn("gamma", " ".join(chunks))
        self.assertIn(chunks[0].split()[-1], chunks[1].split())

    def test_invalid_citation_markers_are_removed_and_counted(self) -> None:
        answer = "Claim [arxiv:2401.12345:1] and [arxiv:2401.12345:2]."

        cleaned, citations, invalid = rag.validate_citations(
            answer, {"arxiv:2401.12345:1"}
        )

        self.assertEqual(cleaned, "Claim [arxiv:2401.12345:1] and.")
        self.assertEqual(citations, ["arxiv:2401.12345:1"])
        self.assertEqual(invalid, 1)

    def test_metrics_are_computed_from_verdicts(self) -> None:
        first = rag.RankedChunk(
            rag.Chunk("arxiv:2401.12345:1", "2401.12345", "Paper", "2024-01-01", "text"),
            0.9,
            {"semantic": 0.9, "recency": 0.8, "feedback": 0.5},
        )
        second = rag.RankedChunk(
            rag.Chunk("arxiv:2401.12345:2", "2401.12345", "Paper", "2024-01-01", "text"),
            0.4,
            {"semantic": 0.4, "recency": 0.8, "feedback": 0.5},
        )
        verdicts = {
            "chunks": [
                {"chunk_id": first.chunk.id, "relevant": True},
                {"chunk_id": second.chunk.id, "relevant": False},
            ],
            "citations": [{"marker": first.chunk.id, "supported": True}],
        }

        metrics = rag.compute_metrics(verdicts, [first, second], [first.chunk.id], 1)

        self.assertEqual(metrics["chunk_precision"], 0.5)
        self.assertEqual(metrics["chunk_recall"], 1.0)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertEqual(metrics["invalid_citations"], 1)

    def test_weight_update_favors_better_separating_features(self) -> None:
        positive = rag.RankedChunk(
            rag.Chunk("p", "p", "p", "", ""),
            0.0,
            {"semantic": 0.9, "recency": 0.8, "feedback": 0.7},
        )
        negative = rag.RankedChunk(
            rag.Chunk("n", "n", "n", "", ""),
            0.0,
            {"semantic": 0.1, "recency": 0.2, "feedback": 0.3},
        )
        before = rag.get_weights(self.connection)

        after = rag.update_weights([(positive, True), (negative, False)], self.connection)

        self.assertGreater(after["semantic"], before["semantic"])
        self.assertAlmostEqual(sum(after.values()), 1.0)
        self.assertEqual(after, rag.get_weights(self.connection))

    def test_judge_rejects_model_generated_metrics(self) -> None:
        chunk = rag.RankedChunk(
            rag.Chunk("arxiv:2401.12345:1", "2401.12345", "Paper", "2024-01-01", "text"),
            0.9,
            {"semantic": 0.9, "recency": 0.8, "feedback": 0.5},
        )
        response = (
            '{"chunks":[{"chunk_id":"arxiv:2401.12345:1","relevant":true}],'
            '"citations":[],"metrics":{"precision":1}}'
        )

        with patch.object(rag, "_api_completion", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "only chunks and citations"):
                rag.judge_answer("question", "answer", [chunk], [])

    def test_evaluation_requires_exact_benchmark_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 24 training questions"):
            evaluation._validate_dataset(
                {"training_questions": ["one"], "held_out_paraphrases": ["two"]}
            )


if __name__ == "__main__":
    unittest.main()
