"""Local browser interface for the self-improving RAG assistant.

Run ``python front.py`` and open the local address printed in the terminal.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlspli

import main as rag


MAX_REQUEST_BYTES = 64 * 1024

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#10151d">
  <title>Local RAG Assistant</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #081018;
      color: #eaf3ff;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
      --bg: #081018;
      --bg-elevated: rgba(16, 24, 34, 0.88);
      --panel: rgba(14, 22, 31, 0.9);
      --panel-strong: rgba(19, 28, 38, 0.96);
      --border: rgba(145, 172, 201, 0.18);
      --muted: #9bb0c8;
      --accent: #7ae7c3;
      --accent-strong: #4fd7ad;
      --accent-soft: rgba(122, 231, 195, 0.14);
      --danger: #ff9f9f;
      --shadow: 0 24px 60px rgba(2, 6, 10, 0.55);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(64, 160, 130, 0.18), transparent 24%),
        radial-gradient(circle at top right, rgba(108, 134, 255, 0.14), transparent 22%),
        var(--bg);
      color: #eaf3ff;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 1rem max(1.25rem, calc((100vw - 1180px) / 2));
      border-bottom: 1px solid var(--border);
      background: rgba(8, 16, 24, 0.82);
      backdrop-filter: blur(14px);
    }
    .brand { display: flex; align-items: center; gap: .8rem; }
    .mark {
      display: grid;
      width: 2.7rem;
      height: 2.7rem;
      place-items: center;
      border-radius: .9rem;
      color: #06251d;
      background: linear-gradient(135deg, var(--accent), #b1ffd8);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 12px 28px rgba(122, 231, 195, 0.35);
      font-size: 1.2rem;
      font-weight: 800;
    }
    h1 { margin: 0; font-size: 1rem; letter-spacing: -.02em; }
    .subhead { margin: .18rem 0 0; color: var(--muted); font-size: .8rem; }
    .local-pill {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      padding: .5rem .8rem;
      border: 1px solid rgba(122, 231, 195, 0.2);
      border-radius: 999px;
      background: rgba(122, 231, 195, 0.06);
      color: var(--accent);
      font-size: .76rem;
      white-space: nowrap;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .local-pill::before {
      content: "";
      display: inline-block;
      width: .52rem;
      height: .52rem;
      border-radius: 50%;
      background: #6ef0bf;
      box-shadow: 0 0 12px rgba(110, 240, 191, 0.9);
    }
    main { width: min(1180px, 100%); margin: 0 auto; padding: 2rem 1.25rem 3rem; }
    .intro { margin-bottom: 1.2rem; }
    .eyebrow {
      display: inline-flex;
      padding: .36rem .7rem;
      border: 1px solid rgba(162, 181, 204, 0.2);
      border-radius: 999px;
      background: rgba(255,255,255,0.02);
      color: #cfe0f2;
      letter-spacing: .12em;
      font-size: .68rem;
      text-transform: uppercase;
    }
    .intro h2 {
      margin: .8rem 0 .55rem;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.02;
      letter-spacing: -.06em;
    }
    .intro p {
      max-width: 740px;
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 1rem;
    }
    .stats { display: flex; flex-wrap: wrap; gap: .75rem; margin-top: 1.2rem; }
    .stat {
      min-width: 160px;
      padding: .8rem 1rem;
      border: 1px solid var(--border);
      border-radius: .9rem;
      background: rgba(11, 17, 24, 0.7);
      color: var(--muted);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .stat strong {
      display: block;
      margin-bottom: .18rem;
      color: #edf4ff;
      font-size: 1.1rem;
      font-variant-numeric: tabular-nums;
    }
    .layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(280px, .8fr); gap: 1rem; align-items: start; }
    .panel {
      border: 1px solid var(--border);
      border-radius: 1.2rem;
      background: linear-gradient(180deg, rgba(18, 28, 36, 0.95), rgba(13, 19, 26, 0.9));
      box-shadow: var(--shadow);
    }
    .panel-head { padding: 1.1rem 1.2rem .8rem; }
    .panel-head h3 { margin: 0; font-size: 1rem; }
    .panel-head p { margin: .35rem 0 0; color: var(--muted); font-size: .82rem; line-height: 1.5; }
    .chat { display: flex; min-height: 560px; flex-direction: column; }
    #conversation { display: flex; flex: 1; flex-direction: column; gap: 1rem; padding: .8rem 1.2rem 1rem; }
    .message {
      max-width: min(88%, 720px);
      padding: .95rem 1rem;
      border: 1px solid transparent;
      border-radius: 1rem;
      line-height: 1.6;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .assistant {
      align-self: flex-start;
      border-color: var(--border);
      background: rgba(13, 20, 27, 0.92);
      color: #edf4ff;
    }
    .user {
      align-self: flex-end;
      color: #07281f;
      background: linear-gradient(135deg, var(--accent), #d1ffe9);
      box-shadow: 0 12px 24px rgba(122, 231, 195, 0.18);
    }
    .sources {
      margin: -.2rem 1.2rem .8rem;
      padding-top: .4rem;
    }
    details {
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: .82rem;
    }
    summary {
      padding: .7rem .1rem;
      cursor: pointer;
      color: #d5e2f1;
      user-select: none;
    }
    .source-text {
      margin: 0 0 .9rem;
      color: #dfeaf7;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    form { display: flex; flex-direction: column; gap: .7rem; }
    .ask-form { padding: 1rem 1.2rem 1.2rem; border-top: 1px solid var(--border); }
    .ask-row { display: flex; gap: .7rem; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: .8rem;
      outline: none;
      padding: .8rem .9rem;
      color: #ebf2ff;
      background: rgba(10, 17, 23, 0.75);
      font: inherit;
      transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
    }
    textarea { min-height: 96px; resize: vertical; }
    input:focus, textarea:focus {
      border-color: rgba(122, 231, 195, 0.8);
      box-shadow: 0 0 0 3px rgba(122, 231, 195, 0.12);
      transform: translateY(-1px);
    }
    button {
      min-height: 2.9rem;
      padding: .7rem 1rem;
      border: 0;
      border-radius: .8rem;
      color: #06261f;
      background: linear-gradient(135deg, var(--accent), #baf6d9);
      font: inherit;
      font-weight: 700;
      letter-spacing: -.01em;
      cursor: pointer;
      transition: transform .15s ease, filter .15s ease, box-shadow .15s ease;
      box-shadow: 0 10px 20px rgba(122, 231, 195, 0.2);
    }
    button:hover { transform: translateY(-1px); filter: brightness(1.04); }
    button:disabled { cursor: wait; opacity: .62; transform: none; }
    .ask-row button { min-width: 5.5rem; }
    .side { display: flex; flex-direction: column; gap: 1rem; }
    .side form { padding: 0 1.2rem 1.2rem; }
    .label {
      display: inline-block;
      margin-bottom: .35rem;
      color: #dfeaf7;
      font-size: .78rem;
      font-weight: 700;
    }
    .helper { margin: -.15rem 0 0; color: var(--muted); font-size: .77rem; line-height: 1.6; }
    .notice {
      min-height: 1.3rem;
      margin: 0 1.2rem .8rem;
      color: var(--muted);
      font-size: .82rem;
    }
    .notice.error { color: var(--danger); }
    footer {
      padding-top: 1.6rem;
      color: #8398ad;
      font-size: .78rem;
      text-align: center;
    }
    @media (max-width: 760px) {
      header { padding: 1rem 1.1rem; }
      main { padding: 1.5rem 1rem 2rem; }
      .layout { grid-template-columns: 1fr; }
      .chat { min-height: 460px; }
      .ask-row { flex-direction: column; }
      .local-pill { font-size: .72rem; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="mark" aria-hidden="true">R</div>
      <div><h1>Self-Improving RAG</h1><p class="subhead">Private knowledge assistant on this machine</p></div>
    </div>
    <span class="local-pill">Local interface</span>
  </header>
  <main>
    <section class="intro">
      <div class="eyebrow">Local AI workspace</div>
      <h2>Ask your docs, then improve them.</h2>
      <p>Index Markdown or text files, get answers grounded in retrieved passages, and save corrections to improve future searches.</p>
      <div class="stats" aria-live="polite">
        <div class="stat"><strong id="chunk-count">—</strong>indexed chunks</div>
        <div class="stat"><strong id="correction-count">—</strong>saved corrections</div>
        <div class="stat"><strong>Ready</strong>knowledge base online</div>
      </div>
    </section>
    <div class="layout">
      <section class="panel chat" aria-label="Ask the assistant">
        <div class="panel-head"><h3>Conversation</h3><p>Answers include source citations when relevant documents are found.</p></div>
        <div id="conversation" aria-live="polite">
          <div class="message assistant">Hi! Ask a question about your indexed documents. Start by indexing a folder if this is your first visit.</div>
        </div>
        <div id="sources" class="sources"></div>
        <form id="ask-form" class="ask-form">
          <label class="label" for="question">Your question</label>
          <div class="ask-row">
            <input id="question" name="question" placeholder="What would you like to know?" autocomplete="off" required>
            <button type="submit">Ask</button>
          </div>
        </form>
        <p id="ask-notice" class="notice" role="status"></p>
      </section>
      <aside class="side" aria-label="Knowledge base tools">
        <section class="panel">
          <div class="panel-head"><h3>Index documents</h3><p>Add or refresh .md and .txt files in your local knowledge base.</p></div>
          <form id="ingest-form">
            <label class="label" for="document-path">File or folder path</label>
            <input id="document-path" name="path" value="knowledge_base" autocomplete="off">
            <p class="helper">Relative paths are resolved from the directory where you started this app. Existing indexed files at the same path are refreshed.</p>
            <button type="submit">Index documents</button>
          </form>
          <p id="ingest-notice" class="notice" role="status"></p>
        </section>
        <section class="panel">
          <div class="panel-head"><h3>Teach a correction</h3><p>Save a trusted answer when the assistant gets something wrong or misses important context.</p></div>
          <form id="learn-form">
            <label class="label" for="learn-question">Question</label>
            <input id="learn-question" name="question" placeholder="What was asked?" required>
            <label class="label" for="learn-answer">Correct answer</label>
            <textarea id="learn-answer" name="answer" placeholder="Enter the answer to remember..." required></textarea>
            <button type="submit">Save correction</button>
          </form>
          <p id="learn-notice" class="notice" role="status"></p>
        </section>
      </aside>
    </div>
    <footer>Runs on this computer. Configure OPENAI_API_KEY to enable generated answers; without it, retrieved passages are shown directly.</footer>
  </main>
  <script>
    const conversation = document.querySelector("#conversation");
    const sources = document.querySelector("#sources");

    async function request(path, data) {
      const options = data === undefined ? {} : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      };
      const response = await fetch(path, options);
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
      return result;
    }

    function showNotice(id, message, isError = false) {
      const element = document.querySelector(`#${id}`);
      element.textContent = message;
      element.classList.toggle("error", isError);
    }

    function addMessage(text, kind) {
      const element = document.createElement("div");
      element.className = `message ${kind}`;
      element.textContent = text;
      conversation.append(element);
      element.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function showSources(items) {
      sources.replaceChildren();
      if (!items.length) return;
      const disclosure = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `Retrieved sources (${items.length})`;
      disclosure.append(summary);
      for (const item of items) {
        const source = document.createElement("details");
        const label = document.createElement("summary");
        label.textContent = item.citation;
        const text = document.createElement("p");
        text.className = "source-text";
        text.textContent = item.text;
        source.append(label, text);
        disclosure.append(source);
      }
      sources.append(disclosure);
    }

    async function refreshStats() {
      const stats = await request("/api/stats");
      document.querySelector("#chunk-count").textContent = stats.chunks;
      document.querySelector("#correction-count").textContent = stats.corrections;
    }

    document.querySelector("#ask-form").addEventListener("submit", async event => {
      event.preventDefault();
      const form = event.currentTarget;
      const question = form.elements.question.value.trim();
      if (!question) return;
      const button = form.querySelector("button");
      button.disabled = true;
      showNotice("ask-notice", "Searching your knowledge base...");
      addMessage(question, "user");
      try {
        const result = await request("/api/ask", { question });
        addMessage(result.answer, "assistant");
        showSources(result.sources);
        showNotice("ask-notice", "");
        form.reset();
        document.querySelector("#question").focus();
      } catch (error) {
        addMessage(`I couldn't complete that request: ${error.message}`, "assistant");
        showNotice("ask-notice", error.message, true);
      } finally {
        button.disabled = false;
      }
    });

    document.querySelector("#ingest-form").addEventListener("submit", async event => {
      event.preventDefault();
      const form = event.currentTarget;
      const button = form.querySelector("button");
      button.disabled = true;
      showNotice("ingest-notice", "Indexing documents...");
      try {
        const result = await request("/api/ingest", { path: form.elements.path.value.trim() });
        showNotice("ingest-notice", `Indexed ${result.chunks} chunks from ${result.files} file(s).`);
        await refreshStats();
      } catch (error) {
        showNotice("ingest-notice", error.message, true);
      } finally {
        button.disabled = false;
      }
    });

    document.querySelector("#learn-form").addEventListener("submit", async event => {
      event.preventDefault();
      const form = event.currentTarget;
      const button = form.querySelector("button");
      button.disabled = true;
      showNotice("learn-notice", "Saving correction...");
      try {
        await request("/api/learn", {
          question: form.elements.question.value,
          answer: form.elements.answer.value
        });
        showNotice("learn-notice", "Correction saved and available to future searches.");
        form.reset();
        await refreshStats();
      } catch (error) {
        showNotice("learn-notice", error.message, true);
      } finally {
        button.disabled = false;
      }
    });

    refreshStats().catch(error => showNotice("ask-notice", error.message, true));
  </script>
</body>
</html>
"""


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if content_type != "application/json":
        raise ValueError("Content-Type must be application/json.")

    try:
        content_length = int(handler.headers.get("Content-Length", ""))
    except ValueError as error:
        raise ValueError("A valid Content-Length header is required.") from error
    if content_length < 0 or content_length > MAX_REQUEST_BYTES:
        raise ValueError(f"Request body must be no larger than {MAX_REQUEST_BYTES} bytes.")

    try:
        payload = json.loads(handler.rfile.read(content_length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Request body must contain valid UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value.strip()


def _stats() -> dict[str, int]:
    with closing(rag.connect_database()) as connection:
        chunks = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        corrections = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return {"chunks": chunks, "corrections": corrections}


class FrontendHandler(BaseHTTPRequestHandler):
    server_version = "LocalRAG"
    sys_version = ""

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/stats":
            try:
                self._send_json(200, _stats())
            except sqlite3.Error as error:
                self._send_json(500, {"error": f"Could not read database statistics: {error}"})
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        route = urlsplit(self.path).path
        try:
            payload = _read_json(self)
            if route == "/api/ask":
                result = self._ask(payload)
            elif route == "/api/ingest":
                result = self._ingest(payload)
            elif route == "/api/learn":
                result = self._learn(payload)
            else:
                self._send_json(404, {"error": "Not found."})
                return
        except (ValueError, FileNotFoundError, UnicodeError) as error:
            self._send_json(400, {"error": str(error)})
        except (OSError, sqlite3.Error, RuntimeError) as error:
            self._send_json(500, {"error": str(error)})
        else:
            self._send_json(200, result)

    def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = _required_text(payload, "question")
        with closing(rag.connect_database()) as connection:
            results = rag.search(question, connection)
            answer = rag.generate_answer(question, results)
        return {
            "answer": answer,
            "sources": [
                {"citation": result.citation, "text": result.text}
                for result in results
            ],
        }

    def _ingest(self, payload: dict[str, Any]) -> dict[str, int]:
        path_text = (
            _required_text(payload, "path")
            if "path" in payload
            else "knowledge_base"
        )
        path = Path(path_text.strip()).expanduser()
        with closing(rag.connect_database()) as connection:
            with connection:
                chunks = rag.ingest(path, connection)
        _, files = rag.discover_documents(path)
        return {"chunks": chunks, "files": len(files)}

    def _learn(self, payload: dict[str, Any]) -> dict[str, str]:
        question = _required_text(payload, "question")
        answer = _required_text(payload, "answer")
        with closing(rag.connect_database()) as connection:
            rag.save_correction(question, answer, connection)
        return {"message": "Correction saved."}

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAG browser interface.")
    parser.add_argument("--port", type=int, default=8000, help="Local port to serve on (default: 8000).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("Error: --port must be between 0 and 65535.", file=sys.stderr)
        return 2

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), FrontendHandler)
    except OSError as error:
        print(f"Error: Could not start local web interface: {error}", file=sys.stderr)
        return 1

    host, port = server.server_address
    print(f"Local RAG interface: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local RAG interface.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())