"""FastAPI application and local browser interface."""

from __future__ import annotations

import sqlite3
from contextlib import closing
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


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0b1018">
  <title>Self-Improving RAG</title>
  <style>
    :root{
      --bg:#070b11;
      --bg-2:#111a22;
      --panel:rgba(17,24,31,0.44);
      --panel-strong:rgba(22,29,37,0.7);
      --card:rgba(255,255,255,0.06);
      --stroke:rgba(255,255,255,0.12);
      --muted:rgba(236,240,245,0.72);
      --text:#edf4ff;
      --subtle:#aebed4;
      --accent:#b7d3ff;
      --accent-2:#7ef0d0;
      --danger:#ff8d8d;
      --shadow:rgba(0,0,0,0.35);
    }

    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0;
      min-height:100vh;
      font-family:Inter,Segoe UI,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(110, 146, 181, 0.26), transparent 26%),
        radial-gradient(circle at bottom right, rgba(127, 228, 194, 0.12), transparent 28%),
        linear-gradient(180deg, rgba(9,12,15,0.84), rgba(7,11,17,0.94)),
        url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat fixed;
      display:flex;
      justify-content:center;
      align-items:flex-start;
      padding:28px 20px 40px;
    }

    body::before{
      content:"";
      position:fixed;
      inset:0;
      background:linear-gradient(180deg, rgba(18,20,26,0.16), rgba(8,10,14,0.4));
      backdrop-filter:blur(2px);
      pointer-events:none;
    }

    .app-shell{
      position:relative;
      z-index:1;
      width:min(1280px, 96vw);
      min-height:700px;
      border:1px solid var(--stroke);
      border-radius:28px;
      background:rgba(15,18,24,0.38);
      box-shadow:0 30px 80px var(--shadow);
      backdrop-filter:blur(18px) saturate(1.15);
      overflow:hidden;
    }

    .topbar{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:18px;
      padding:18px 24px 16px;
      background:rgba(25,30,36,0.28);
      border-bottom:1px solid rgba(255,255,255,0.08);
    }

    .brand-group{
      display:flex;
      align-items:center;
      gap:14px;
      min-width:0;
    }

    .brand-mark{
      width:42px;height:42px;border-radius:12px;
      background:linear-gradient(135deg, rgba(234,240,247,0.9), rgba(168,180,194,0.86));
      box-shadow:inset 0 1px 0 rgba(255,255,255,0.65), 0 8px 18px rgba(0,0,0,0.18);
      position:relative;display:grid;place-items:center;
    }

    .brand-mark::before{
      content:"";
      width:18px;height:18px;border:2px solid rgba(15,18,22,0.9);border-radius:5px;
      position:relative;display:block;transform:translateY(-1px);
    }

    .brand-mark::after{
      content:"";
      position:absolute;
      width:22px;height:12px;border:2px solid rgba(15,18,22,0.9);border-bottom:none;border-radius:12px 12px 0 0;
      transform:translateY(8px);
    }

    .brand-text h1{
      margin:0;
      font-size:clamp(1.2rem,2vw,2rem);
      letter-spacing:-0.05em;
      font-weight:700;
    }

    .brand-text p{
      margin:4px 0 0;
      color:var(--subtle);
      font-size:.78rem;
    }

    .header-actions{
      display:flex;
      align-items:center;
      gap:12px;
    }

    .chip,
    .icon-button{
      border:1px solid rgba(255,255,255,0.08);
      background:rgba(255,255,255,0.04);
      color:var(--text);
      border-radius:999px;
      padding:9px 14px;
      font:inherit;
      cursor:pointer;
    }

    .chip{padding:10px 16px; color:var(--muted)}
    .icon-button{width:40px;height:40px;padding:0;display:grid;place-items:center;font-size:1.1rem}

    .stats-grid{
      display:grid;
      grid-template-columns:repeat(4,minmax(160px,1fr));
      gap:16px;
      padding:20px 24px 0;
    }

    .stat-card{
      background:rgba(255,255,255,0.04);
      border:1px solid rgba(255,255,255,0.08);
      border-radius:18px;
      padding:16px 18px;
      min-height:92px;
      backdrop-filter:blur(12px);
      box-shadow:inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .stat-label{color:var(--subtle);font-size:.82rem;margin:0 0 10px;}
    .stat-value{font-size:clamp(1.5rem,2.4vw,2.4rem);letter-spacing:-0.06em;font-weight:700;line-height:1.1;margin:0;}
    .stat-note{color:var(--muted);font-size:.76rem;opacity:.9;margin-top:8px;}

    .content-grid{
      display:grid;
      grid-template-columns:minmax(0,1.3fr) minmax(300px,.9fr);
      gap:18px;
      padding:20px 24px 24px;
    }

    .panel{
      background:rgba(255,255,255,0.04);
      border:1px solid rgba(255,255,255,0.08);
      border-radius:22px;
      backdrop-filter:blur(14px);
      box-shadow:inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .ask-panel{padding:22px 20px 18px}
    .panel-header{
      display:flex;align-items:center;justify-content:space-between;gap:12px;
      margin-bottom:18px;
    }
    .panel-title{margin:0;font-size:clamp(1.4rem,2vw,2.2rem);font-weight:700;letter-spacing:-0.06em}
    .micro-tag{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--subtle);background:rgba(255,255,255,0.05);padding:6px 8px;border-radius:999px;border:1px solid rgba(255,255,255,0.08)}

    #conversation{display:flex;flex-direction:column;gap:12px;min-height:160px;margin-bottom:12px}
    .bubble{
      max-width:92%;padding:12px 14px;border-radius:16px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;
      border:1px solid rgba(255,255,255,0.08);
    }
    .bubble.user{align-self:flex-end;background:rgba(183,211,255,0.15);color:#fff}
    .bubble.assistant{background:rgba(255,255,255,0.035);color:var(--text)}

    .ask-form{
      display:flex;gap:10px;
      margin-top:10px;
    }
    .ask-form input{
      flex:1;min-width:0;padding:14px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.1);
      background:rgba(8,12,16,0.28);color:var(--text);font:inherit;outline:none;
    }
    .ask-form input::placeholder{color:rgba(255,255,255,0.42)}
    .ask-form input:focus{border-color:rgba(126,240,208,0.7);box-shadow:0 0 0 3px rgba(126,240,208,0.12)}
    .primary-btn{
      border:0;border-radius:14px;padding:0 18px;font:inherit;font-weight:700;
      background:linear-gradient(135deg, rgba(126,240,208,0.9), rgba(183,211,255,0.86));
      color:#081218;cursor:pointer;min-width:118px;
    }
    .primary-btn:disabled{opacity:.58;cursor:wait}

    .meta-wrap{margin-top:14px;display:grid;gap:10px}
    .source{border-top:1px solid rgba(255,255,255,0.08);padding-top:12px;color:var(--muted)}
    .source summary{cursor:pointer;color:var(--text);font-size:.95rem}
    .source p{margin:8px 0 0;color:var(--subtle);font-size:.82rem;line-height:1.6}
    .source a{color:var(--accent);text-decoration:none}
    .tag{display:inline-block;padding:5px 8px;border-radius:999px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);color:var(--muted);font-size:.72rem;margin:4px 6px 0 0}
    .tag.good{color:rgba(126,240,208,0.92)}
    .tag.bad{color:rgba(255,141,141,0.95)}

    .side-stack{display:grid;gap:16px}
    .side-panel{padding:18px 18px 16px}
    .mini-title{margin:0 0 10px;font-size:1.1rem;font-weight:700;letter-spacing:-0.02em}
    .tiny-copy{margin:0 0 14px;color:var(--subtle);font-size:.8rem;line-height:1.5}

    .ingest-form{display:flex;gap:10px;flex-wrap:wrap}
    .ingest-form input{flex:1 1 160px;padding:12px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.1);background:rgba(8,12,16,0.28);color:var(--text);font:inherit;min-width:120px}
    .ingest-form input:last-child{max-width:90px}
    .ingest-form .primary-btn{min-width:96px;padding:0 14px}
    .notice{min-height:1.3rem;color:var(--muted);font-size:.78rem;margin:10px 0 0}
    .notice.error{color:var(--danger)}

    .weights-list{display:grid;gap:10px;margin-top:8px}
    .weight-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;color:var(--muted);font-size:.82rem;border-bottom:1px solid rgba(255,255,255,0.06)}
    .weight-row:last-child{border-bottom:0}
    .weight-row strong{color:var(--text);font-weight:700}

    .metrics-wrap{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}
    .metric-box{padding:10px 8px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);color:var(--subtle);font-size:.72rem;line-height:1.4}
    .metric-box strong{display:block;font-size:1rem;color:var(--text);margin-bottom:4px;letter-spacing:-0.04em}

    @media(max-width:920px){
      .stats-grid{grid-template-columns:repeat(2, minmax(150px,1fr));}
      .content-grid{grid-template-columns:1fr;}
      .header-actions{display:none}
    }

    @media(max-width:560px){
      .stats-grid{grid-template-columns:1fr;}
      .ask-form,.ingest-form{flex-direction:column}
      .primary-btn{width:100%}
      .topbar{padding:16px 18px 12px}
      .stats-grid,.content-grid{padding-left:16px;padding-right:16px}
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-group">
        <div class="brand-mark" aria-hidden="true"></div>
        <div class="brand-text">
          <h1>Self-Improving RAG</h1>
          <p>Grounded answers from live arXiv papers</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="chip" type="button">Adjust glass</button>
        <button class="icon-button" type="button" aria-label="Toggle theme">T</button>
      </div>
    </header>

    <section class="stats-grid" aria-live="polite">
      <div class="stat-card">
        <p class="stat-label">Indexed papers</p>
        <p class="stat-value" id="paper-count">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Evidence chunks</p>
        <p class="stat-value" id="chunk-count">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Judged answers</p>
        <p class="stat-value" id="evaluation-count">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Retrieval weights</p>
        <p class="stat-value" id="weight-status">—</p>
      </div>
    </section>

    <main class="content-grid">
      <section class="panel ask-panel">
        <div class="panel-header">
          <h2 class="panel-title">Ask a research question</h2>
          <span class="micro-tag">Evidence</span>
        </div>

        <div id="conversation" aria-live="polite">
          <div class="bubble assistant">Index a topic to get started, then ask a question across the retrieved papers.</div>
        </div>

        <form id="ask-form" class="ask-form">
          <input name="question" aria-label="Research question" placeholder="What are the recent approaches to retrieval augmentation?" required>
          <button class="primary-btn" type="submit">Ask</button>
        </form>

        <div id="answer-meta" class="meta-wrap"></div>
        <p id="ask-notice" class="notice" role="status"></p>
      </section>

      <aside class="side-stack">
        <section class="panel side-panel">
          <h3 class="mini-title">Index live arXiv papers</h3>
          <p class="tiny-copy">Search arXiv, download PDF evidence, embed passages, and store them for retrieval.</p>
          <form id="ingest-form" class="ingest-form">
            <input name="query" aria-label="arXiv search topic" placeholder="retrieval augmented generation" required>
            <input name="limit" type="number" min="1" max="50" value="10" aria-label="Paper count">
            <button class="primary-btn" type="submit">Index</button>
          </form>
          <p id="ingest-notice" class="notice" role="status"></p>
        </section>

        <section class="panel side-panel">
          <h3 class="mini-title">Signal weights</h3>
          <div id="weights" class="weights-list"></div>
        </section>

        <section class="panel side-panel">
          <h3 class="mini-title">Answer evaluation</h3>
          <div id="metrics" class="metrics-wrap"></div>
        </section>
      </aside>
    </main>
  </div>

<script>
  const $ = selector => document.querySelector(selector);
  async function request(url, data) {
    const response = await fetch(url, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || `Request failed (${response.status})`);
    return result;
  }

  function notice(id, text, error=false) {
    const node = $(id);
    if (!node) return;
    node.textContent = text;
    node.classList.toggle('error', error);
  }

  function showWeights(weights) {
    const weightsNode = $('#weights');
    if (!weightsNode) return;
    const entries = Object.entries(weights || {});
    weightsNode.replaceChildren(...entries.map(([name, value]) => {
      const row = document.createElement('div');
      row.className = 'weight-row';
      const label = document.createElement('span');
      label.textContent = name;
      const number = document.createElement('strong');
      number.textContent = `${(value * 100).toFixed(1)}%`;
      row.append(label, number);
      return row;
    }));
    const statusNode = $('#weight-status');
    if (statusNode) statusNode.textContent = 'Adaptive';
  }

  function showMetrics(metrics) {
    const metricsNode = $('#metrics');
    if (!metricsNode) return;
    const names = ['chunk_precision','chunk_recall','chunk_f1','citation_precision','citation_recall','citation_f1'];
    metricsNode.replaceChildren(...names.map((name) => {
      const card = document.createElement('div');
      card.className = 'metric-box';
      const value = document.createElement('strong');
      value.textContent = `${(metrics[name] * 100).toFixed(1)}%`;
      const label = document.createTextNode(name.replaceAll('_', ' '));
      card.append(value, label);
      return card;
    }));

    const invalid = document.createElement('div');
    invalid.className = 'metric-box';
    const invalidValue = document.createElement('strong');
    invalidValue.textContent = metrics.invalid_citations ?? 0;
    invalid.append(invalidValue, document.createTextNode('invalid citations'));
    metricsNode.append(invalid);
  }

  function showAnswer(result) {
    const answer = document.createElement('div');
    answer.className = 'bubble assistant';
    answer.textContent = result.answer;
    $('#conversation').append(answer);

    const meta = $('#answer-meta');
    meta.replaceChildren();
    for (const source of result.sources || []) {
      const details = document.createElement('details');
      details.className = 'source';
      const summary = document.createElement('summary');
      summary.textContent = `[${source.citation}] ${source.title} · ${source.published.slice(0,10)} · ${source.score.toFixed(3)}`;
      const text = document.createElement('p');
      text.textContent = source.text;
      const link = document.createElement('a');
      link.href = `https://arxiv.org/abs/${encodeURIComponent(source.arxiv_id)}`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `Open arXiv:${source.arxiv_id}`;
      details.append(summary, text, link);
      meta.append(details);
    }

    for (const citation of result.citations || []) {
      const tag = document.createElement('span');
      tag.className = `tag ${citation.supported ? 'good' : 'bad'}`;
      tag.textContent = `${citation.marker}: ${citation.supported ? 'supported' : 'unsupported'}`;
      meta.append(tag);
    }

    showMetrics(result.metrics || {});
    showWeights(result.weights || {semantic:0.65, recency:0.2, feedback:0.15});
  }

  $('#ask-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button');
    const question = form.elements.question.value.trim();
    if (!question) return;

    const user = document.createElement('div');
    user.className = 'bubble user';
    user.textContent = question;
    $('#conversation').append(user);

    button.disabled = true;
    notice('#ask-notice', 'Retrieving evidence and checking citations…');

    try {
      const result = await request('/api/ask', {question});
      showAnswer(result);
      form.reset();
      notice('#ask-notice', 'Answer judged; weights updated from chunk relevance verdicts.');
    } catch (error) {
      notice('#ask-notice', error.message, true);
    } finally {
      button.disabled = false;
    }
  });

  $('#ingest-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button');
    button.disabled = true;
    notice('#ingest-notice', 'Searching arXiv and indexing paper PDFs…');

    try {
      const result = await request('/api/ingest', {query: form.elements.query.value.trim(), limit: Number(form.elements.limit.value)});
      notice('#ingest-notice', `Indexed ${result.papers} papers and ${result.chunks} chunks.`);
      await refreshStats();
    } catch (error) {
      notice('#ingest-notice', error.message, true);
    } finally {
      button.disabled = false;
    }
  });

  async function refreshStats() {
    const data = await request('/api/stats');
    const paperValue = $('#paper-count');
    if (paperValue) paperValue.textContent = data.papers ?? 0;
    const chunkValue = $('#chunk-count');
    if (chunkValue) chunkValue.textContent = data.chunks ?? 0;
    const evalValue = $('#evaluation-count');
    if (evalValue) evalValue.textContent = data.evaluations ?? 0;
    showWeights(data.weights || {semantic:0.65, recency:0.2, feedback:0.15});
  }

  refreshStats().catch((error) => notice('#ask-notice', error.message || 'Stats unavailable', true));
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


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
