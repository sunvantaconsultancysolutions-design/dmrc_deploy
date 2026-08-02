# DMRC Contract Intelligence — Enterprise RAG Platform

![Python](https://img.shields.io/badge/Python-3.10%2F3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)
![React](https://img.shields.io/badge/React-frontend-61DAFB)
![Docker](https://img.shields.io/badge/Docker-supported-2496ED)
![License](https://img.shields.io/badge/license-Proprietary-lightgrey)

## Project Overview

DMRC Contract Intelligence is an **Enterprise Retrieval-Augmented Generation (RAG) system** for querying Delhi Metro Rail Corporation (DMRC) construction contract documents — including ECS/TVS scope-of-work text and Bill of Quantities (BOQ) items — in natural language.

The system ingests clause-level and BOQ-level contract data, indexes it with hybrid dense + lexical retrieval, reranks candidates with a cross-encoder, gates on retrieval confidence before ever calling the LLM, and generates grounded, citation-aware answers. It is designed to let contract, engineering, and legal teams ask direct questions about scope, specifications, and quantities instead of manually searching PDF documents.

**The vector store, rendered page images, and source PDFs are already built and committed — no ingestion step is required to run the app. Every chunk in the database (375/375) resolves to a real scanned page image.**

## Features

- **Clause-level document chunking** — contract text is split into clause- and item-level chunks with preserved hierarchy
- **Metadata-aware retrieval** — chunk metadata (clause number, section, document type, BOQ item fields) filters and grounds every result
- **Query intent routing** — every question is classified `clause` / `boq` / `general` before retrieval
- **Exact-match fast paths** — naming a clause number or BOQ item number verbatim skips retrieval entirely (`score=1.0`), pulling in the rest of that item's family on request
- **Dense semantic search** — `BAAI/bge-m3` embeddings for meaning-based retrieval
- **BM25 lexical search** — label-aware keyword retrieval; the index is built from the same clause-number/heading/BOQ-item-enriched text used for embedding, so identifiers like "Part-H" or "6.7.2-4" are never invisible to keyword search
- **Scoring-stopword filtering** — generic meta-language ("what", "is", "specification") is excluded from BM25 scoring so it can't drown out a genuinely rare, correctly-matching term
- **Hybrid retrieval** — combines dense and BM25 results for recall neither achieves alone
- **Domain synonym expansion** — plain-English phrasing ("penalty", "who trains X", "ACB rating") is expanded toward the contract's own formal terminology before both BM25 and reranking
- **Cross-encoder reranking** — `BAAI/bge-reranker-v2-m3` reorders candidates by relevance, scoring the same synonym-expanded query BM25 already used
- **Confidence gating with sibling recovery** — an absolute floor plus separation-from-pool check reject weak matches before generation; a borderline result gets one retry pulling in its clause/BOQ-item family first
- **Citation-accurate generation** — `google/gemma-2-9b-it` (bf16) copies each cited identifier verbatim from its source block's own header, whatever shape it takes (`6.8`, `PART-I`, `a)`, `iii)`)
- **Evidence-list navigation** — the frontend viewer steps through retrieved evidence ranked by score, skipping any entry with no renderable image
- **FastAPI backend** — serves retrieval and generation through a REST API
- **ChromaDB vector database** — persistent, embedded, committed pre-built
- **REST API** — `/ask` and `/status` endpoints for querying and health checks
- **Docker, auto-built** — every push to `main` builds and publishes an image to GitHub Container Registry

## System Architecture

```
                +--------------------+
                |   React Frontend    |
                +----------+----------+
                           | HTTPS (CORS)
                           v
                +--------------------+
                |   FastAPI Backend   |
                |   (app.py, /ask)    |
                +----------+----------+
                           v
                +--------------------+
                |   Query Router      |
                | clause / boq /      |
                | general             |
                +----------+----------+
                           v
        +------------------+------------------+
        v                  v                  v
 +-----------+     +---------------+   +--------------+
 |  BGE-M3   |     |     BM25      |   |  ChromaDB     |
 |  Dense    |     |   Lexical     |   |  Vector Store |
 |  Retrieval|     |  (label-aware,|   |               |
 |           |     |   stopword-   |   |               |
 |           |     |   filtered)   |   |               |
 +-----+-----+     +-------+-------+   +------+--------+
       +-----------+---------+                |
                 v                            |
         +---------------+                    |
         | Hybrid Fusion  |<-------------------+
         +-------+--------+
                 v
     +-----------------------+
     |  BGE Cross-Encoder     |
     |  Reranker (v2-m3)      |
     |  same expanded query   |
     +-----------+------------+
                 v
     +-----------------------+
     |  Confidence Gate        |
     |  + sibling/family retry  |
     +-----------+------------+
                 v
     +-----------------------+
     |  Prompt Construction   |
     |  (citation rules)       |
     +-----------+------------+
                 v
     +-----------------------+
     |  Gemma-2-9B-it (bf16)  |
     |  Answer Generation     |
     +-----------+------------+
                 v
     Cited Answer + Evidence Page
```

## Project Structure

```
dmrc_deploy/
|-- README.md
|-- requirements.txt
|-- Dockerfile
|-- main.py                        # Ingestion entrypoint (data/*.json -> chroma_db/)
|-- patch_ch3_pdf_page.py          # One-time metadata patch (already applied)
|-- patch_addendum_images.py       # One-time metadata patch (already applied)
|-- data/                          # Transcribed clause + BOQ source JSON
|-- chroma_db/                     # Pre-built, committed vector store
|-- page_images/                   # Rendered scanned pages, one folder per document_id
|-- source_pdfs/                   # Original contract PDFs (6 files)
|-- scripts/                       # Maintenance/diagnostic utilities
|-- docs/                          # Metadata schema reference
|-- src/
|   |-- app.py                     # FastAPI app: /ask, /status, /pages, /figures
|   |-- query_router.py            # clause / boq / general classification
|   |-- query.py                   # Exact clause/BOQ-item fast paths + family lookups
|   |-- hybrid_retriever.py        # BM25 + dense fusion, synonym query expansion
|   |-- bm25_index.py              # In-memory BM25, label-aware, stopword-filtered
|   |-- reranker.py                # Cross-encoder reranking, confidence gate, sibling expansion
|   |-- prompt_engineering.py      # Prompt assembly, citation rules, token budgeting
|   |-- gemma_inference.py         # Gemma-2-9B-it load + generation
|   |-- storage.py                 # ChromaDB collection accessor
|   |-- metadata_loader.py         # Parses data/*.json into embeddable chunk records
|   |-- text_normalization.py      # Enriched text builder shared by embedding/BM25/reranking
|   |-- text_stem.py               # Lightweight stemmer shared by the router and BM25
|   |-- retrieval_caps.py          # Hard ceilings on candidate/context list sizes
|   `-- validate_db.py             # Standalone ChromaDB consistency checker
`-- frontend/                      # React + Vite chat UI and evidence viewer
    `-- src/
```

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10/3.11 |
| Backend API | FastAPI |
| Vector database | ChromaDB |
| Embedding model | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Generation model | `google/gemma-2-9b-it` (bf16) |
| Frontend | React (Vite) |
| Containerization | Docker, auto-built via GitHub Actions → GHCR |
| Development GPU | NVIDIA A100 (Colab / RunPod) |

## Retrieval Pipeline

```
User Query
        |
        v
Exact-match fast path? (clause/BOQ item number named verbatim)
        | no
        v
Query Router (clause / boq / general)
        |
        v
Hybrid Search: BM25 (synonym-expanded, label-aware, stopword-filtered) + BGE-M3 Dense
        |
        v
Cross-Encoder Reranking (same expanded query as BM25)
        |
        v
Confidence Gate (absolute floor + separation check; sibling/family retry on borderline)
        |
        v
Prompt Construction (context + verbatim citation rules)
        |
        v
Gemma-2-9B-it Inference
        |
        v
Cited Answer + Evidence Page
```

## Installation

```bash
git clone https://github.com/sunvantaconsultancysolutions-design/dmrc_deploy
cd dmrc_deploy

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cd frontend && npm install && cd ..
```

`google/gemma-2-9b-it` is a **gated** model — accept its license at
https://huggingface.co/google/gemma-2-9b-it, then set `HF_TOKEN` before
running the server.

## Running the Project

The vector store is already built, so you can start the server directly:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx   # Windows: set HF_TOKEN=hf_xxxxxxxxxxxx
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

First start downloads ~20 GB of model weights (10-20 minutes). Check readiness:

```bash
curl http://127.0.0.1:8000/status
```

Query it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the penalty for delay?"}'
```

Run the frontend:

```bash
cd frontend
echo "VITE_API_URL=http://127.0.0.1:8000" > .env
npm run dev
```

## Running via Docker

**Option A — pull the image GitHub already built:**

```bash
docker pull ghcr.io/sunvantaconsultancysolutions-design/dmrc_deploy:latest

docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v /path/to/persistent/models:/models \
  ghcr.io/sunvantaconsultancysolutions-design/dmrc_deploy:latest
```

**Option B — build directly from GitHub, no clone needed:**

```bash
docker build -t dmrc-backend https://github.com/sunvantaconsultancysolutions-design/dmrc_deploy.git
docker run --gpus all -p 8000:8000 -e HF_TOKEN=hf_xxxxxxxxxxxx dmrc-backend
```

The container's healthcheck allows up to 25 minutes for cold model loading.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic health check |
| `GET` | `/status` | Model/DB readiness (`dense_model_loaded`, `reranker_model_loaded`, `gemma_model_loaded`, `chromadb_connected`) |
| `POST` | `/ask` | `{"query": "..."}` → cited answer + source list |
| `GET` | `/pages/manifest` | Available scanned page images |
| `GET` | `/figures/manifest` | Available extracted figures |
| `GET` | `/pages/{document_id}/{page}.jpg` | Static scanned page image |

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(required)* | HF token with Gemma license accepted |
| `GEMMA_USE_4BIT` | `0` | Set `1` on lower-VRAM GPUs |
| `RAG_MAX_CANDIDATES` / `RAG_MAX_CONTEXT` | `60` / `20` | Retrieval breadth ceilings |
| `RAG_MIN_CONFIDENCE` / `RAG_MIN_SEPARATION` | `0.10` / `0.08` | Confidence gate thresholds |
| `ALLOWED_ORIGINS` | `*` (dev) | CORS allow-list — set explicitly in production |
| `RAG_DEBUG` | unset | Set `1` for verbose retrieval/confidence logging |

## Re-running Ingestion

Only needed when adding a new source document:

```bash
python main.py --input data/your_new_file.json
python scripts/render_pages.py --pdf-dir source_pdfs
python scripts/migrate_page_image_dirs.py
```

## Diagnostics

`scripts/calibrate_confidence.py` replays queries through the real retrieval + reranking pipeline (requires GPU/model access) and reports the actual confidence-gate decision and score for each — useful for investigating a query that returns "not found" despite the underlying data existing:

```bash
python scripts/calibrate_confidence.py --queries my_queries.jsonl --csv results.csv
```

## Known Data-Coverage Limitations

- Civil works (excavation, earthwork, foundations) exist only as a single financial total (Part-G); no line-item detail was transcribed.
- Transformers, generators, batteries, and several other equipment types are not mentioned anywhere in the indexed corpus.
- A handful of BOQ sub-item "rating" fragments can still weakly rank against a generic disclaimer chunk — not fixed, since doing so the same way as "specification" would have broken the already-working ACB/busbar rating queries.

## Future Improvements

- Transcribe remaining civil works and mechanical equipment line items
- User authentication and role-based access control
- Conversation history / multi-turn context
- Automated regression suite run in CI on every push
