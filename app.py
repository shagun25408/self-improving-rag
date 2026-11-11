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
  <meta name="theme-color" content="#0b0d0f">
  <title>Automate Flows</title>
  <style>
    :root {
      --bg: #07090a;
      --panel: rgba(30, 26, 22, 0.72);
      --panel-strong: rgba(28, 24, 20, 0.84);
      --stroke: rgba(255,255,255,0.12);
      --muted: rgba(255,255,255,0.6);
      --text: #f4f1ee;
      --soft: rgba(248,236,215,0.62);
      --line: rgba(255,255,255,0.18);
      --green: #d7f0dd;
      --accent: #efe5d8;
      --shadow: rgba(0,0,0,0.42);
      --chart: rgba(255,255,255,0.85);
      --grid: rgba(255,255,255,0.14);
      --danger: #ff8d8d;
      --pill: rgba(255,255,255,0.08);
      --dark-pill: rgba(5, 5, 5, 0.36);
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "Segoe UI", sans-serif;
      background:
        linear-gradient(0deg, rgba(0,0,0,0.36), rgba(0,0,0,0.36)),
        radial-gradient(circle at 15% 15%, rgba(100,70,38,0.35), transparent 30%),
        linear-gradient(180deg, rgba(15,12,10,0.68), rgba(11,11,11,0.9)),
        repeating-linear-gradient(
          90deg,
          rgba(101,76,50,0.32) 0 2px,
          rgba(83,68,49,0.18) 2px 4px,
          rgba(52,41,30,0.28) 4px 6px,
          rgba(73,55,39,0.26) 6px 8px
        );
      color: var(--text);
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding: 26px 18px 28px;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(0,0,0,0.2), rgba(0,0,0,0.04) 25%, rgba(0,0,0,0.18) 60%, rgba(0,0,0,0.32)),
        repeating-linear-gradient(
          90deg,
          rgba(63,45,31,0.38) 0 10px,
          rgba(96,70,49,0.28) 10px 24px,
          rgba(42,31,24,0.28) 24px 34px,
          rgba(80,59,42,0.18) 34px 42px
        );
      opacity: 0.9;
      pointer-events: none;
    }

    .page-shell {
      position: relative;
      width: min(1280px, 96vw);
      min-height: 560px;
      padding: 0 0 12px;
      z-index: 1;
    }

    .page-title {
      color: rgba(255,255,255,0.9);
      font-size: 15px;
      font-weight: 500;
      letter-spacing: 0.01em;
      margin: 0 0 12px 8px;
      opacity: 0.92;
    }

    .page-title small {
      display: block;
      margin-top: 2px;
      font-size: 12px;
      color: rgba(255,255,255,0.52);
      font-weight: 400;
    }

    .dashboard-shell {
      position: relative;
      background: linear-gradient(180deg, rgba(103,72,49,0.54), rgba(93,68,44,0.44));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 28px;
      padding: 18px 20px 18px;
      box-shadow: 0 40px 90px rgba(0,0,0,0.28);
      overflow: hidden;
      min-height: 500px;
      backdrop-filter: blur(10px);
    }

    .dashboard-shell::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      pointer-events: none;
    }

    .shell-header {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 82px;
      padding: 10px 12px 12px 16px;
      border-radius: 20px 20px 0 0;
      background: rgba(30, 26, 22, 0.18);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      box-shadow: inset 0 -1px rgba(255,255,255,0.04);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(247,242,235,0.96), rgba(194,183,170,0.9));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.65), 0 8px 18px rgba(17,17,17,0.18);
      position: relative;
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }

    .brand-mark::before {
      content: "";
      width: 17px;
      height: 17px;
      border: 2px solid rgba(18,18,22,0.9);
      border-top-width: 2.5px;
      border-radius: 4px;
      transform: translateY(-1px);
      display: block;
      position: relative;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04);
    }

    .brand-mark::after {
      content: "";
      position: absolute;
      width: 22px;
      height: 12px;
      border: 2px solid rgba(18,18,22,0.9);
      border-bottom: none;
      border-radius: 12px 12px 0 0;
      transform: translateY(8px);
    }

    .brand-name {
      font-size: clamp(1.15rem, 2vw, 2rem);
      font-weight: 700;
      letter-spacing: -0.04em;
      color: rgba(255,255,255,0.92);
      white-space: nowrap;
    }

    .shell-actions {
      display: flex;
      align-items: center;
      gap: 14px;
      color: rgba(255,255,255,0.72);
    }

    .glass-button {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.06);
      padding: 8px 12px 8px 10px;
      font-size: 0.98rem;
      font-weight: 500;
    }

    .glass-button .icon {
      font-size: 1.1rem;
      opacity: 0.8;
    }

    .theme-pill {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.06);
      display: grid;
      place-items: center;
      font-size: 1.2rem;
      font-weight: 600;
      color: rgba(255,255,255,0.8);
    }

    .content {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(220px, 0.92fr) minmax(380px, 1.6fr);
      gap: 20px;
      padding: 28px 18px 0;
      min-height: 380px;
    }

    .panel {
      position: relative;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(49, 41, 35, 0.28);
      box-shadow: inset 0 1px rgba(255,255,255,0.03);
      backdrop-filter: blur(14px);
    }

    .balance-panel {
      padding: 24px 20px 22px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 290px;
    }

    .balance-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }

    .balance-top .mini-label {
      color: rgba(255,255,255,0.72);
      font-size: 0.9rem;
      margin: 0;
    }

    .balance-icon {
      width: 32px;
      height: 32px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.18);
      display: grid;
      place-items: center;
      color: rgba(255,255,255,0.76);
      background: rgba(255,255,255,0.02);
    }

    .balance-info {
      margin-top: 32px;
    }

    .balance-label {
      font-size: 1.06rem;
      color: rgba(255,255,255,0.68);
      margin-bottom: 8px;
    }

    .amount {
      font-size: clamp(2.1rem, 4vw, 3.1rem);
      line-height: 1;
      letter-spacing: -0.07em;
      font-weight: 700;
      color: rgba(255,255,255,0.96);
      margin: 0;
      font-variant-numeric: tabular-nums;
    }

    .trend {
      margin-top: 14px;
      color: rgba(255,255,255,0.64);
      font-size: 0.96rem;
      letter-spacing: -0.02em;
    }

    .trend strong { color: #cfead8; }

    .chart-panel {
      padding: 24px 22px 18px;
      background: rgba(52, 44, 38, 0.16);
    }

    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }

    .chart-title {
      font-size: clamp(1.4rem, 2vw, 2.2rem);
      margin: 0;
      letter-spacing: -0.06em;
      color: rgba(255,255,255,0.9);
    }

    .segmented {
      display: inline-flex;
      gap: 6px;
      padding: 4px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
    }

    .segmented button {
      border: 0;
      border-radius: 10px;
      background: transparent;
      color: rgba(255,255,255,0.6);
      padding: 7px 10px;
      font-weight: 600;
      cursor: pointer;
    }

    .segmented .active {
      background: rgba(15,16,17,0.45);
      color: rgba(255,255,255,0.9);
    }

    .totals {
      display: flex;
      gap: 40px;
      align-items: center;
      margin-bottom: 18px;
    }

    .totals-item {
      display: flex;
      align-items: center;
      gap: 10px;
      color: rgba(255,255,255,0.62);
      font-size: 0.98rem;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: rgba(255,255,255,0.82);
      display: inline-block;
    }

    .dot.muted { background: rgba(255,255,255,0.42); }

    .totals-values {
      display: flex;
      gap: 28px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-top: 8px;
      margin-bottom: 10px;
    }

    .value {
      font-size: clamp(1.6rem, 2.8vw, 2.5rem);
      letter-spacing: -0.07em;
      font-weight: 700;
      color: rgba(255,255,255,0.96);
      font-variant-numeric: tabular-nums;
    }

    .chart-wrap {
      position: relative;
      margin-top: 6px;
      height: 206px;
      border-radius: 16px;
      overflow: hidden;
    }

    .chart-wrap::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.00)),
        repeating-linear-gradient(
          180deg,
          transparent 0,
          transparent 29px,
          rgba(255,255,255,0.08) 29px,
          rgba(255,255,255,0.08) 30px
        );
      pointer-events: none;
    }

    .chart-wrap svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      opacity: 0.95;
    }

    .chart-wrap .y-labels {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 24px 0 18px 0;
      color: rgba(255,255,255,0.45);
      font-size: 0.72rem;
      pointer-events: none;
    }

    .chart-wrap .y-labels span {
      display: block;
      margin-left: 10px;
    }

    .hidden {
      display: none !important;
    }

    @media (max-width: 900px) {
      .content {
        grid-template-columns: 1fr;
      }

      .shell-header {
        flex-wrap: wrap;
        gap: 12px;
      }

      .brand-name {
        white-space: normal;
      }
    }
  </style>
</head>
<body>
  <div class="page-shell">
    <h1 class="page-title">Top today<small>The most saved components of the day</small></h1>

    <div class="dashboard-shell">
      <div class="shell-header">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true"></div>
          <div class="brand-name">Automate Flows</div>
        </div>

        <div class="shell-actions">
          <div class="glass-button"><span class="icon">◌</span>Adjust glass</div>
          <div class="theme-pill" aria-label="Theme toggle">T</div>
        </div>
      </div>

      <div class="content">
        <section class="panel balance-panel">
          <div class="balance-top">
            <p class="mini-label">Available balance</p>
            <div class="balance-icon">▣</div>
          </div>

          <div class="balance-info">
            <div class="amount" id="paper-count">$28,430.60</div>
            <div class="trend"><strong>+8.2%</strong> compared with last month</div>
          </div>
        </section>

        <section class="panel chart-panel">
          <div class="chart-header">
            <h2 class="chart-title">Cash flow</h2>
            <div class="segmented" aria-label="Range selector">
              <button type="button">Week</button>
              <button type="button" class="active">Month</button>
              <button type="button">Year</button>
            </div>
          </div>

          <div class="totals">
            <div class="totals-item"><span class="dot"></span>Money in</div>
            <div class="totals-item"><span class="dot muted"></span>Money out</div>
          </div>

          <div class="totals-values">
            <div class="value" id="chunk-count">$11,050.00</div>
            <div class="value" id="weight-status">$2,399.30</div>
          </div>

          <div class="chart-wrap">
            <div class="y-labels" aria-hidden="true">
              <span>$1,225.00</span>
              <span>$612.50</span>
            </div>
            <svg viewBox="0 0 700 210" preserveAspectRatio="none" aria-label="Cash flow chart">
              <path d="M0,180 L76,170 L150,120 L220,140 L292,150 L370,145 L450,126 L520,103 L600,126 L700,80 L700,210 L0,210 Z" fill="rgba(255,255,255,0.05)"/>
              <path d="M0,180 L76,170 L150,120 L220,140 L292,150 L370,145 L450,126 L520,103 L600,126 L700,80" fill="none" stroke="rgba(255,255,255,0.88)" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </section>
      </div>
    </div>
  </div>

  <div class="hidden">
    <div id="conversation" aria-live="polite"></div>
    <form id="ask-form"><input name="question" aria-label="Research question" placeholder="What are the recent approaches to retrieval augmentation?" required></form>
    <div id="answer-meta"></div>
    <p id="ask-notice" class="notice" role="status"></p>
    <form id="ingest-form" class="query-row"><input name="query" aria-label="arXiv search topic" placeholder="retrieval augmented generation" required><input name="limit" type="number" min="1" max="50" value="10" aria-label="Paper count"><button>Index</button></form>
    <p id="ingest-notice" class="notice" role="status"></p>
    <div id="weights" class="weights"></div>
    <div id="metrics" class="metrics"></div>
    <span id="evaluation-count">0</span>
  </div>

<script>
  const $ = selector => document.querySelector(selector);
  async function request(url, data) {
    const response = await fetch(url, data === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || `Request failed (${response.status})`);
    return result;
  }
  function notice(id, text, error=false){const node=$(id);if(!node)return;node.textContent=text;node.classList.toggle("error",error);}
  function showWeights(weights){const weightsNode=$("#weights"); if(!weightsNode) return; weightsNode.replaceChildren(...Object.entries(weights).map(([name,value])=>{const row=document.createElement("div");row.className="weight";const label=document.createElement("span");label.textContent=name;const number=document.createElement("strong");number.textContent=`${(value*100).toFixed(1)}%`;row.append(label,number);return row;}));const statusNode=$("#weight-status");if(statusNode) statusNode.textContent="Adaptive";}
  function showMetrics(metrics){
    const metricsNode=$("#metrics"); if(!metricsNode) return;
    const names=["chunk_precision","chunk_recall","chunk_f1","citation_precision","citation_recall","citation_f1"];
    metricsNode.replaceChildren(...names.map(name=>{const card=document.createElement("div");card.className="metric";const value=document.createElement("strong");value.textContent=`${(metrics[name]*100).toFixed(1)}%`;card.append(value,document.createTextNode(name.replaceAll("_"," ")));return card;}));
    const invalid=document.createElement("div");invalid.className="metric";const n=document.createElement("strong");n.textContent=metrics.invalid_citations;invalid.append(n,document.createTextNode("invalid citations stripped"));metricsNode.append(invalid);
  }
  function showAnswer(result){
    const answer=document.createElement("div");answer.className="bubble assistant";answer.textContent=result.answer;$("#conversation").append(answer);
    const meta=$("#answer-meta");meta.replaceChildren();
    for(const source of result.sources){const details=document.createElement("details");details.className="source";const summary=document.createElement("summary");summary.textContent=`[${source.citation}] ${source.title} · ${source.published.slice(0,10)} · ${source.score.toFixed(3)}`;const text=document.createElement("p");text.textContent=source.text;const link=document.createElement("a");link.href=`https://arxiv.org/abs/${encodeURIComponent(source.arxiv_id)}`;link.target="_blank";link.rel="noopener noreferrer";link.textContent=`Open arXiv:${source.arxiv_id}`;details.append(summary,text,link);meta.append(details);}
    for(const citation of result.citations){const tag=document.createElement("span");tag.className=`tag ${citation.supported?"good":"bad"}`;tag.textContent=`${citation.marker}: ${citation.supported?"supported":"unsupported"}`;meta.append(tag);}
    showMetrics(result.metrics);showWeights(result.weights);
  }
  $("#ask-form") && $("#ask-form").addEventListener("submit",async event=>{event.preventDefault();const form=event.currentTarget;const button=form.querySelector("button");const question=form.elements.question.value.trim();if(!question)return;const user=document.createElement("div");user.className="bubble user";user.textContent=question;$("#conversation").append(user);button.disabled=true;notice("#ask-notice","Retrieving evidence and checking citations…");try{const result=await request("/api/ask",{question});showAnswer(result);form.reset();notice("#ask-notice","Answer judged; weights updated from chunk relevance verdicts.");}catch(error){notice("#ask-notice",error.message,true);}finally{button.disabled=false;}});
  $("#ingest-form") && $("#ingest-form").addEventListener("submit",async event=>{event.preventDefault();const form=event.currentTarget;const button=form.querySelector("button");button.disabled=true;notice("#ingest-notice","Searching arXiv and indexing paper PDFs…");try{const result=await request("/api/ingest",{query:form.elements.query.value.trim(),limit:Number(form.elements.limit.value)});notice("#ingest-notice",`Indexed ${result.papers} papers and ${result.chunks} chunks.`);await refreshStats();}catch(error){notice("#ingest-notice",error.message,true);}finally{button.disabled=false;}});
  async function refreshStats(){const data=await request("/api/stats");const paperCount=$("#paper-count"); if(paperCount) paperCount.textContent=`$${Number(data.papers || 0).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}`;const chunkCount=$("#chunk-count"); if(chunkCount) chunkCount.textContent=`$${Number(data.chunks || 0).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}`;const evaluationCount=$("#evaluation-count"); if(evaluationCount) evaluationCount.textContent=data.evaluations ?? 0;showWeights(data.weights || {semantic:0.65,recency:0.2,feedback:0.15});}
  refreshStats().catch(() => {});
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
