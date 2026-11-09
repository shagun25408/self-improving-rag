"""Command-line tools for indexing arXiv papers and asking grounded questions."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Sequence

import rag_engine as rag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-improving RAG over arXiv papers")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Search arXiv and index paper PDFs")
    ingest.add_argument("query", help="arXiv search phrase")
    ingest.add_argument("--limit", type=int, default=10)
    ask = commands.add_parser("ask", help="Ask a question over indexed papers")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    commands.add_parser("stats", help="Show index size and retrieval weights")
    evaluate = commands.add_parser("evaluate", help="Compare fixed and adaptive retrieval")
    evaluate.add_argument("dataset", type=Path, help="JSON with 24 questions and 12 held-out paraphrases")
    evaluate.add_argument("--output", type=Path, help="Optional path for the JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            with closing(rag.connect_database()) as connection:
                result = rag.ingest_arxiv(args.query, args.limit, connection)
            print(f"Indexed {result['papers']} papers and {result['chunks']} chunks.")
        elif args.command == "ask":
            with closing(rag.connect_database()) as connection:
                result = rag.answer_question(args.question, connection, args.top_k)
            print(result["answer"])
            print("\nMetrics:")
            print(json.dumps(result["metrics"], indent=2))
            print("\nWeights:")
            print(json.dumps(result["weights"], indent=2))
        elif args.command == "stats":
            with closing(rag.connect_database()) as connection:
                print(json.dumps(rag.get_stats(connection), indent=2))
        elif args.command == "evaluate":
            from evaluation import run_evaluation

            dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
            report = run_evaluation(dataset)
            rendered = json.dumps(report, indent=2)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
    except (OSError, sqlite3.Error, UnicodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
