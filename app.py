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
  <title>Self-Improving RAG | arXiv</title>
  <style>
    :root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:#0b1018;color:#edf3fb;font-synthesis:none;--panel:#121a25;--border:#253243;--muted:#9cacc1;--mint:#79e6c1;--blue:#8eb8ff;--red:#ff9797}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 8% 0%,#15372e 0,transparent 36%),radial-gradient(ellipse at 92% 10%,#1d2c49 0,transparent 30%),#0b1018}
    header{position:sticky;top:0;z-index:2;display:flex;justify-content:space-between;align-items:center;padding:1rem max(1.2rem,calc((100vw - 1120px)/2));border-bottom:1px solid #253243;background:#0b1018df;backdrop-filter:blur(16px)}
    .brand{display:flex;gap:.8rem;align-items:center}.mark{display:grid;place-items:center;width:2.6rem;height:2.6rem;border-radius:.9rem;background:linear-gradient(135deg,#79e6c1,#b7f5d8);color:#08251d;font-weight:900}.brand h1{font-size:1rem;margin:0}.brand p{margin:.16rem 0 0;color:var(--muted);font-size:.76rem}
    .pill{border:1px solid #325047;border-radius:999px;padding:.45rem .75rem;color:var(--mint);font-size:.75rem}
    main{max-width:1120px;margin:auto;padding:2.2rem 1.2rem 3rem}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.7rem;color:var(--mint)}
    h2{font-size:clamp(2rem,5vw,3.8rem);line-height:1.04;letter-spacing:-.055em;max-width:800px;margin:.7rem 0}.lead{max-width:700px;color:var(--muted);line-height:1.7}
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:1.5rem 0}.stat,.card{border:1px solid var(--border);border-radius:1rem;background:#121a25e8;box-shadow:0 20px 55px #0003}.stat{padding:.9rem 1rem;color:var(--muted);font-size:.8rem}.stat strong{display:block;color:#f1f5fb;font-size:1.25rem;margin-bottom:.25rem;font-variant-numeric:tabular-nums}
    .grid{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(270px,.8fr);gap:1rem;align-items:start}.card{padding:1.2rem}.card h3{margin:0;font-size:1rem}.hint{color:var(--muted);font-size:.82rem;line-height:1.55;margin:.4rem 0 1rem}
    form{display:flex;gap:.65rem}input{min-width:0;flex:1;padding:.8rem .9rem;border:1px solid var(--border);border-radius:.7rem;color:inherit;background:#0b1119;font:inherit;outline:none}input:focus{border-color:var(--mint);box-shadow:0 0 0 3px #79e6c122}button{border:0;border-radius:.7rem;padding:.75rem 1rem;background:linear-gradient(135deg,#79e6c1,#b9f4d8);color:#09241d;font:inherit;font-weight:750;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.query-row{display:flex;gap:.6rem}.query-row input{width:5rem;flex:0 0 5rem}
    #conversation{display:flex;flex-direction:column;gap:.8rem;margin:1rem 0}.bubble{max-width:92%;padding:.9rem 1rem;border-radius:.9rem;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}.bubble.user{align-self:flex-end;background:#79e6c1;color:#08251d}.bubble.assistant{border:1px solid var(--border);background:#0d141d}
    .source{border-top:1px solid var(--border);padding:.7rem 0;color:#dce7f3}.source summary{cursor:pointer}.source p{font-size:.82rem;color:#c2d0df;line-height:1.6;white-space:pre-wrap}.tag{display:inline-block;margin:.35rem .35rem 0 0;padding:.24rem .48rem;border-radius:999px;background:#1d2a37;color:#bed0e1;font-size:.7rem}.good{color:var(--mint)}.bad{color:var(--red)}.notice{min-height:1.2rem;color:var(--muted);font-size:.8rem;margin:.75rem 0 0}.error{color:var(--red)}.weights{display:grid;gap:.55rem;margin-top:.7rem}.weight{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted)}.weight strong{color:#eaf1f9}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}.metric{padding:.55rem;background:#0b1119;border-radius:.6rem;color:var(--muted);font-size:.72rem}.metric strong{display:block;color:#edf3fb;font-size:.95rem;margin-bottom:.15rem}a{color:var(--blue)}
    @media(max-width:760px){header{padding:1rem}.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.query-row{flex-wrap:wrap}.query-row input:first-child{flex-basis:100%}}
  </style>
</head>
<body>
  <header><div class="brand"><div class="mark">R</div><div><h1>Self-Improving RAG</h1><p>Grounded answers from live arXiv literature</p></div></div><span class="pill">Local workspace</span></header>
  <main>
    <div class="eyebrow">Evidence-first research assistant</div>
    <h2>Explore the literature.<br>Keep the evidence in view.</h2>
    <p class="lead">Search arXiv, index paper passages, and ask questions with chunk-level citations. Retrieval weights adapt from per-chunk judge verdicts; evaluation metrics are computed locally from those verdicts.</p>
    <section class="stats" aria-live="polite">
      <div class="stat"><strong id="paper-count">—</strong>indexed papers</div><div class="stat"><strong id="chunk-count">—</strong>evidence chunks</div>
      <div class="stat"><strong id="evaluation-count">—</strong>judged answers</div><div class="stat"><strong id="weight-status">—</strong>retrieval weights</div>
    </section>
    <div class="grid">
      <section class="card">
        <h3>Ask a research question</h3><p class="hint">Answers use only indexed passages. Each citation points to a stored chunk and is checked before display.</p>
        <div id="conversation" aria-live="polite"><div class="bubble assistant">Index a topic to get started, then ask a question across the retrieved papers.</div></div>
        <form id="ask-form"><input name="question" aria-label="Research question" placeholder="What are the recent approaches to retrieval augmentation?" required><button>Ask</button></form>
        <div id="answer-meta"></div><p id="ask-notice" class="notice" role="status"></p>
      </section>
      <aside style="display:grid;gap:1rem">
        <section class="card"><h3>Index live arXiv papers</h3><p class="hint">Find the newest matching papers, download PDFs, extract text, embed chunks, and store them for retrieval.</p>
          <form id="ingest-form" class="query-row"><input name="query" aria-label="arXiv search topic" placeholder="retrieval augmented generation" required><input name="limit" type="number" min="1" max="50" value="10" aria-label="Paper count"><button>Index</button></form><p id="ingest-notice" class="notice" role="status"></p>
        </section>
        <section class="card"><h3>Signal weights</h3><p class="hint">Updated online from how well each signal separates relevant from irrelevant chunks.</p><div id="weights" class="weights"></div></section>
        <section class="card"><h3>Answer evaluation</h3><p class="hint">Six code-computed measures based on judge verdicts, plus a count of unresolved citation markers.</p><div id="metrics" class="metrics"><div class="metric">Metrics appear after an answer is judged.</div></div></section>
        <section class="card"><h3>Run the benchmark</h3><p class="hint">The CLI harness compares fixed semantic-only retrieval against online adaptation over three learning cycles, then evaluates 12 held-out paraphrases.</p><p class="hint">See <a href="/docs">API documentation</a> and the README for setup.</p></section>
      </aside>
    </div>
  </main>
<script>
  const $ = selector => document.querySelector(selector);
  async function request(url, data) {
    const response = await fetch(url, data === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || `Request failed (${response.status})`);
    return result;
  }
  function notice(id, text, error=false){const node=$(id);node.textContent=text;node.classList.toggle("error",error);}
  function showWeights(weights){$("#weights").replaceChildren(...Object.entries(weights).map(([name,value])=>{const row=document.createElement("div");row.className="weight";const label=document.createElement("span");label.textContent=name;const number=document.createElement("strong");number.textContent=`${(value*100).toFixed(1)}%`;row.append(label,number);return row;}));$("#weight-status").textContent="Adaptive";}
  function showMetrics(metrics){
    const names=["chunk_precision","chunk_recall","chunk_f1","citation_precision","citation_recall","citation_f1"];
    $("#metrics").replaceChildren(...names.map(name=>{const card=document.createElement("div");card.className="metric";const value=document.createElement("strong");value.textContent=`${(metrics[name]*100).toFixed(1)}%`;card.append(value,document.createTextNode(name.replaceAll("_"," ")));return card;}));
    const invalid=document.createElement("div");invalid.className="metric";const n=document.createElement("strong");n.textContent=metrics.invalid_citations;invalid.append(n,document.createTextNode("invalid citations stripped"));$("#metrics").append(invalid);
  }
  function showAnswer(result){
    const answer=document.createElement("div");answer.className="bubble assistant";answer.textContent=result.answer;$("#conversation").append(answer);
    const meta=$("#answer-meta");meta.replaceChildren();
    for(const source of result.sources){const details=document.createElement("details");details.className="source";const summary=document.createElement("summary");summary.textContent=`[${source.citation}] ${source.title} · ${source.published.slice(0,10)} · ${source.score.toFixed(3)}`;const text=document.createElement("p");text.textContent=source.text;const link=document.createElement("a");link.href=`https://arxiv.org/abs/${encodeURIComponent(source.arxiv_id)}`;link.target="_blank";link.rel="noopener noreferrer";link.textContent=`Open arXiv:${source.arxiv_id}`;details.append(summary,text,link);meta.append(details);}
    for(const citation of result.citations){const tag=document.createElement("span");tag.className=`tag ${citation.supported?"good":"bad"}`;tag.textContent=`${citation.marker}: ${citation.supported?"supported":"unsupported"}`;meta.append(tag);}
    showMetrics(result.metrics);showWeights(result.weights);
  }
  $("#ask-form").addEventListener("submit",async event=>{event.preventDefault();const form=event.currentTarget;const button=form.querySelector("button");const question=form.elements.question.value.trim();if(!question)return;const user=document.createElement("div");user.className="bubble user";user.textContent=question;$("#conversation").append(user);button.disabled=true;notice("#ask-notice","Retrieving evidence and checking citations…");try{const result=await request("/api/ask",{question});showAnswer(result);form.reset();notice("#ask-notice","Answer judged; weights updated from chunk relevance verdicts.");}catch(error){notice("#ask-notice",error.message,true);}finally{button.disabled=false;}});
  $("#ingest-form").addEventListener("submit",async event=>{event.preventDefault();const form=event.currentTarget;const button=form.querySelector("button");button.disabled=true;notice("#ingest-notice","Searching arXiv and indexing paper PDFs…");try{const result=await request("/api/ingest",{query:form.elements.query.value.trim(),limit:Number(form.elements.limit.value)});notice("#ingest-notice",`Indexed ${result.papers} papers and ${result.chunks} chunks.`);await refreshStats();}catch(error){notice("#ingest-notice",error.message,true);}finally{button.disabled=false;}});
  async function refreshStats(){const data=await request("/api/stats");$("#paper-count").textContent=data.papers;$("#chunk-count").textContent=data.chunks;$("#evaluation-count").textContent=data.evaluations;showWeights(data.weights);}
  refreshStats().catch(error=>notice("#ask-notice",error.message,true));
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
