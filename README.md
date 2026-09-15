# SentinelGraph

**An AI-powered distributed failure tracking and telemetry observability framework.**

SentinelGraph ingests telemetry from background job pipelines, correlates failures across services in a graph database, detects abnormal failure bursts in real time, clusters related errors using unsupervised ML, and synthesizes structured incident post-mortems using an LLM — all exposed live over a WebSocket stream and available as a batch CLI pipeline.

It's designed as a research-oriented prototype for distributed job monitoring, automated log clustering, and root-cause synthesis — the kind of system a small platform/SRE team would run in front of a microservices architecture to cut down diagnostic time during incidents.

---

## Quick Start

```bash
git clone https://github.com/chopradisha86-source/SentinelGraph.git && cd SentinelGraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY

python main.py --count 3000 --skip-db --skip-ai   # batch pipeline, no external services required
uvicorn app:app --reload                            # live WebSocket server → ws://localhost:8000/ws/logs
```

See [Setup](#setup) for full instructions, including Memgraph and Docker.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Local (virtualenv)](#local-virtualenv)
  - [Docker](#docker)
- [Usage](#usage)
  - [Batch pipeline (`main.py`)](#batch-pipeline-mainpy)
  - [Live server (`app.py`)](#live-server-apppy)
- [API Reference](#api-reference)
- [Sample Output](#sample-output)
- [Evaluation](#evaluation)
- [Research Roadmap](#research-roadmap)
- [License](#license)

---

## Architecture

```
                         ┌────────────────────────────────┐
                         │      Log Stream Source         │
                         │      src/generator.py          │
                         │  (synthetic or real telemetry) │
                         └───────────────┬────────────────┘
                                         │
                                         ▼
                         ┌────────────────────────────┐
                         │    FastAPI / WebSockets    │
                         │    app.py  →  /ws/logs     │
                         └───────────────┬────────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                        ▼
       ┌────────────────────────────┐         ┌────────────────────────────┐
       │        Detector            │         │     Memgraph Graph Store   │
       │      src/detector.py       │         │      src/db_handler.py     │
       │  rolling-window burst      │         │  :Job nodes,               │
       │  detection → alerts.json   │         │  :FAILED_TOGETHER edges    │
       └───────────────┬────────────┘         └────────────────────────────┘
                       ▼
       ┌────────────────────────────┐
       │        Clusterer           │
       │      src/clusterer.py      │
       │  TF-IDF + DBSCAN           │
       │  → clusters.json           │
       └───────────────┬────────────┘
                       ▼
       ┌────────────────────────────┐
       │     Gemini LLM Synthesi    │
       │    src/ai_synthesis.py     │
       └───────────────┬────────────┘
                       ▼
       ┌────────────────────────────┐
       │     Markdown Post-Mortem   │
       │   output/incident_reports  │
       └────────────────────────────┘
```

Two independent consumers sit downstream of ingestion: the **Detector → Clusterer → LLM Synthesis** chain (the analytical pipeline that produces alerts and post-mortems), and the **Memgraph graph store** (which persists jobs as nodes and derives `:FAILED_TOGETHER` relationships between failures that share a `trace_id` within a configurable time window — the basis for any cross-service failure-correlation analysis).

---

## Key Features

- **Real-time telemetry streaming** over WebSockets, with a REST root endpoint and interactive OpenAPI docs at `/docs`.
- **Rolling-window anomaly detection** — flags a service the moment its trailing failure count exceeds a configurable threshold within a configurable time window, with rising-edge alerting to avoid duplicate spam during sustained outages.
- **Unsupervised error clustering** via TF-IDF vectorization + DBSCAN (cosine distance), grouping related failures without predefined error categories, plus an evaluated semantic-embedding variant (see [Evaluation](#evaluation)).
- **Graph-based failure correlation** — Memgraph-backed `:Job` nodes and `:FAILED_TOGETHER` relationships linking failures from the same distributed trace.
- **LLM-synthesized incident post-mortems** — Gemini API integration that turns a cluster of related error logs into a structured, on-call-ready Markdown report (Incident Overview, Timeline of Events, Recommended Remediation).
- **Two execution modes**: a batch CLI (`main.py`) for offline analysis of a log corpus, and a live server (`app.py`) for continuous monitoring, including an on-demand WebSocket command to generate an incident report for any currently-forming cluster.
- **Containerized deployment** via Docker Compose, wired to a `memgraph-platform` container with health-gated startup ordering.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API / streaming | FastAPI, WebSockets, Uvicorn |
| Data processing | pandas |
| Clustering | scikit-learn (TF-IDF, DBSCAN); sentence-transformers (semantic embeddings variant) |
| Graph persistence | Memgraph (via gqlalchemy, Bolt protocol) |
| AI synthesis | Google Gemini API (`google-genai`) — `gemini-3.5-flash-lite` |
| Synthetic data | Faker |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
SentinelGraph/
├── src/
│   ├── __init__.py
│   ├── generator.py               # synthetic telemetry log generation
│   ├── db_handler.py              # Memgraph ingestion + FAILED_TOGETHER correlation
│   ├── detector.py                # rolling-window failure-burst detection
│   ├── clusterer.py                # TF-IDF + DBSCAN error clustering
│   ├── clusterer_embeddings.py    # semantic-embedding clustering variant
│   └── ai_synthesis.py             # Gemini API incident/post-mortem synthesis
├── output/
│   ├── alerts.json
│   ├── clusters.json
│   └── incident_reports/
│       └── cluster_0.md
├── main.py                          # CLI batch pipeline runner
├── app.py                           # FastAPI + WebSocket live server
├── eval_clustering.py                # clustering quality eval (lexically-distinct errors)
├── eval_clustering_hard.py           # clustering eval (paraphrased failure variants)
├── eval_clustering_embeddings.py     # embeddings vs. TF-IDF comparison
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Setup

### Local (virtualenv)

```bash
# 1. Clone and enter the repo
git clone https://github.com/chopradisha86-source/SentinelGraph.git
cd SentinelGraph

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and set:
#   GEMINI_API_KEY=your-key-here
```

A running Memgraph instance is required for the graph-correlation step. The quickest way to get one locally:

```bash
docker run -d -p 7687:7687 -p 7444:7444 -p 3000:3000 --name memgraph memgraph/memgraph-platform
```

### Docker

To run the entire stack (app + Memgraph) together:

```bash
cp .env.example .env   # set GEMINI_API_KEY
docker compose up --build
```

This starts Memgraph (with Memgraph Lab at `http://localhost:3000`) and the FastAPI app (`http://localhost:8000`), with the app container waiting on Memgraph's healthcheck before starting.

---

## Usage

### Batch pipeline (`main.py`)

Runs the full pipeline end-to-end against a batch of generated logs: ingestion → graph correlation → burst detection → clustering → AI synthesis.

```bash
python main.py --count 3000
```

Full flag reference:

```bash
--count 2000              # number of synthetic log events to generate
--seed 42                 # random seed, for reproducible runs
--out-dir output          # where logs.jsonl, alerts.json, clusters.json, and reports are written

--skip-db                 # skip Memgraph ingestion (e.g. no local instance running)
--db-host 127.0.0.1       # Memgraph host (defaults to $MEMGRAPH_HOST if set)
--db-port 7687            # Memgraph Bolt port (defaults to $MEMGRAPH_PORT if set)
--correlation-window 10   # FAILED_TOGETHER correlation window, in seconds

--burst-window 120        # failure-burst rolling window, in seconds
--burst-threshold 5       # alert fires when this count is exceeded

--eps 0.5                 # DBSCAN cosine-distance eps
--min-samples 3           # DBSCAN min_samples

--skip-ai                 # skip Gemini report generation (e.g. no API key configured)
--gemini-model gemini-3.5-flash-lite   # Gemini model name (defaults to gemini-2.5-flash in code;
                                          #   override with this flag or edit DEFAULT_MODEL in
                                          #   src/ai_synthesis.py to change the default)
--max-reports 5            # max clusters to generate AI reports for (largest first)
```

### Live server (`app.py`)

```bash
uvicorn app:app --reload
```

Connect a WebSocket client to `ws://localhost:8000/ws/logs` to receive a live event stream, or open `http://localhost:8000/docs` for interactive API documentation.

---

## API Reference

### `GET /`

Health/status check.

```json
{
  "status": "ok",
  "websocket_endpoint": "/ws/logs",
  "active_connections": 1,
  "buffered_logs": 214,
  "latest_cluster_count": 6
}
```

### `WS /ws/logs`

Streams one message per tick (default: every second).

**Server → client, log event:**

```json
{
  "type": "log_event",
  "log": {
    "job_id": "b6e1...",
    "service": "payment",
    "status": "FAILED",
    "error_msg": "Payment gateway timeout",
    "trace_id": "a1c9...",
    "timestamp": "2026-08-23T14:02:11.482+00:00"
  },
  "alerts": []
}
```

`alerts` is populated only on the tick a new failure-burst alert first fires:

```json
{
  "type": "log_event",
  "log": { "...": "..." },
  "alerts": [
    {
      "service": "payment",
      "triggered_at": "2026-08-23T14:03:04.120+00:00",
      "window_failure_count": 6,
      "window_seconds": 120,
      "threshold": 5,
      "trace_ids": ["a1c9...", "f02b..."]
    }
  ]
}
```

**Server → client, periodic cluster summary:**

```json
{
  "type": "cluster_update",
  "clusters": { "0": 12, "1": 5, "2": 3 }
}
```

**Client → server, on-demand incident report:**

```json
{ "action": "generate_report", "cluster_id": 0 }
```

**Server → client, response:**

```json
{ "type": "incident_report", "cluster_id": 0, "report": "Incident Overview\n\n* Primary Failing Service: payment\n..." }
```

> **Note:** `app.py`'s live `generate_report` command always uses `DEFAULT_MODEL` from `src/ai_synthesis.py` (currently `gemini-2.5-flash`) — it doesn't expose a per-request model override the way `main.py --gemini-model` does. If you want the live server to use `gemini-3.5-flash-lite` (or any other model), pass it explicitly when calling `generate_incident_report()` from your own code, or change `DEFAULT_MODEL` in `src/ai_synthesis.py`.

---

## Sample Output

`clusters.json` (excerpt):

```json
{
  "0": [
    {
      "job_id": "b6e1...",
      "service": "payment",
      "status": "FAILED",
      "error_msg": "Payment gateway timeout",
      "trace_id": "a1c9...",
      "timestamp": "2026-08-23T14:02:19.001+00:00"
    }
  ]
}
```

`output/incident_reports/cluster_0.md` (excerpt — see the full file in the repo for a complete example):

```markdown
Incident Overview

* Primary Failing Service: payment
* Impact Severity: High — payment job failures exceeded the 5-per-2-minute
  burst threshold three separate times within an 8-minute span.
* Root Cause Hypothesis: Redis-backed job queue for payment processing
  became saturated, causing consumers to time out waiting for queue pops.

Timeline of Events

* 14:02:11 UTC — First payment job failure: Redis connection pool exhausted...
* 14:03:04 UTC — Failure-burst alert triggered for payment...
...

Recommended Remediation

1. Immediately check Redis memory usage and maxmemory-policy...
...
```

Every report follows this fixed three-section structure (Incident Overview / Timeline of Events / Recommended Remediation), enforced via a system prompt in `src/ai_synthesis.py` so output stays consistent across clusters.

---

## Evaluation

`src/clusterer.py` was evaluated against two synthetic test sets using the `eval_clustering*.py` scripts included in this repo, to move beyond "it runs" and actually measure clustering quality.

**Easy test** (20 lexically-distinct known error types, `eval_clustering.py`): TF-IDF + DBSCAN achieves 20/20 correct separation with 0 false merges across `eps` 0.3–0.8.

**Hard test** (5 underlying failures, each with 4 paraphrased variants — realistic for logs written by different services/authors describing the same failure differently, `eval_clustering_hard.py`): TF-IDF + DBSCAN cannot unify paraphrases and keep unrelated errors apart at the same time. The only `eps` that unifies 4/5 semantic groups (`eps=0.9`) also produces 14 false merges on the easy test — raising the similarity threshold enough to recognize paraphrases also starts merging genuinely unrelated errors.

Replacing TF-IDF with sentence-transformer embeddings (`src/clusterer_embeddings.py`, `all-MiniLM-L6-v2`) resolves this: at `eps ≥ 0.4`, the same hard test achieves 5/5 correct unification with **zero false merges observed across the entire swept range** (`eps` 0.1–0.5) — no precision/recall cliff, unlike the lexical approach.

| Method | Best config | Semantic groups unified | False merges |
|---|---|---|---|
| TF-IDF + DBSCAN | eps=0.9 | 4/5 | 14 (on easy test) |
| Sentence embeddings + DBSCAN | eps≥0.4 | 5/5 | 0 |

Caveat: false merges were not observed within the tested `eps` range for the embedding variant, but higher values weren't swept — this isn't a claim that embeddings never merge, only what was measured.

Reproduce this yourself:

```bash
python eval_clustering.py --sweep --count 4000 --seed 42
python eval_clustering_hard.py --eps 0.5 --count 2000 --seed 42
python eval_clustering_embeddings.py --sweep --count 2000 --seed 42
```

---

## Research Roadmap

Directions under active exploration, extending the current baselines:

- [x] **Semantic log embeddings** — implemented (`src/clusterer_embeddings.py`) and evaluated against TF-IDF (see [Evaluation](#evaluation)); resolves the precision/recall tradeoff inherent to lexical (word-overlap) similarity, at the cost of requiring a local embedding model instead of pure scikit-learn.
- [ ] **Cross-service failure propagation modeling** — mining the Memgraph `:FAILED_TOGETHER` graph for repeated service-to-service failure sequences, to predict cascading failures rather than only detect bursts after the fact.
- [ ] **Retrieval-augmented incident synthesis** — conditioning the Gemini prompt on the most similar past incident reports, to evaluate whether few-shot retrieval reduces root-cause hallucination versus the current zero-shot approach.

---

## License

MIT