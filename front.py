"""Launch the local browser application with ``python front.py``."""

from __future__ import annotations

import argparse
import sys

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the self-improving arXiv RAG interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    try:
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
    except OSError as error:
        print(f"Error: Could not start the web interface: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
