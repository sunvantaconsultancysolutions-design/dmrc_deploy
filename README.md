# DMRC Contract Intelligence — Enterprise RAG Platform

![Python](https://img.shields.io/badge/Python-3.10%2F3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)
![React](https://img.shields.io/badge/React-frontend-61DAFB)
![Docker](https://img.shields.io/badge/Docker-supported-2496ED)
![License](https://img.shields.io/badge/license-Proprietary-lightgrey)

## Project Overview

DMRC Contract Intelligence is an **Enterprise Retrieval-Augmented Generation (RAG) system** for querying Delhi Metro Rail Corporation (DMRC) construction contract documents — including ECS/TVS scope-of-work text and Bill of Quantities (BOQ) items — in natural language.

The system ingests clause-level and BOQ-level contract data, indexes it with hybrid dense + lexical retrieval, reranks candidates with a cross-encoder, and generates grounded, citation-aware answers using an LLM. It is designed to let contract, engineering, and legal teams ask direct questions about scope, specifications, and quantities instead of manually searching PDF documents.

## Features

- **Clause-level document chunking** — contract text is split into clause- and item-level chunks with preserved hierarchy
- **Metadata-aware retrieval** — chunk metadata (clause number, section, document type, BOQ item fields) is used to filter and ground results
- **Dense semantic search** — `BAAI/bge-m3` embeddings for meaning-based retrieval
- **BM25 lexical search** — keyword-based sparse retrieval for exact-term matches
- **Hybrid retrieval** — combines dense and BM25 results for improved recall
- **Cross-encoder reranking** — `BAAI/bge-reranker-v2-m3` reorders candidates by relevance
- **Prompt engineering** — structured prompt construction with retrieved context and citation instructions
- **Answer generation** — `google/gemma-2-9b-it` (bf16) produces grounded answers from reranked context
- **FastAPI backend** — serves retrieval and generation through a REST API
- **ChromaDB vector database** — persistent local/embedded vector store
- **REST API** — `/ask` and `/status` endpoints for querying and health checks
- **React frontend** — chat-style UI for submitting questions and viewing answers
- **Docker support** — containerized backend for GPU deployment

## System Architecture

```
                ┌────────────────────┐
                │   React Frontend    │
                │   (Vercel-hosted)   │
                └──────────┬──────────┘
                           │ HTTPS (CORS)
                           ▼
                ┌────────────────────┐
                │   FastAPI Backend   │
                │   (app.py, /ask)    │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 ┌───────────┐     ┌───────────────┐   ┌──────────────┐
 │  BGE-M3   │     │     BM25      │   │  ChromaDB     │
 │  Dense    │     │   Lexical     │   │  Vector Store │
 │  Retrieval│     │   Retrieval   │   │               │
 └─────┬─────┘     └───────┬───────┘   └──────┬────────┘
       └─────────┬─────────┘                  │
                 ▼                             │
         ┌───────────────┐                     │
         │ Hybrid Fusion  │◄────────────────────┘
         └───────┬────────┘
                 ▼
     ┌───────────────────────┐
     │  BGE Cross-Encoder     │
     │  Reranker (v2-m3)      │
     └───────────┬────────────┘
                 ▼
     ┌───────────────────────┐
     │  Prompt Construction   │
     └───────────┬────────────┘
                 ▼
     ┌───────────────────────┐
     │  Gemma-2-9B-it (bf16)  │
     │  Answer Generation     │
     └───────────┬────────────┘
                 ▼
           Final Answer
```

## Project Structure

```
dmrc-contract-intelligence/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/                        # Parsed contract JSON (clause + BOQ chunks)
├── chroma_db/                   # Persistent vector store
├── src/
│   ├── app.py                   # FastAPI application, CORS, /ask, /status
│   ├── text_normalization.py    # Normalization, embedding-input builder
│   ├── metadata_loader.py       # Metadata schema mapping (clause + BOQ)
│   ├── embed_single.py          # Single-chunk BGE-M3 encoding
│   ├── batch_embed.py           # Batched BGE-M3 encoding
│   ├── storage.py               # ChromaDB persistence
│   ├── retrieval.py             # Dense + BM25 hybrid retrieval
│   ├── retrieval_caps.py        # Retrieval limits / candidate caps
│   ├── reranker.py              # Cross-encoder reranking
│   ├── prompt_builder.py        # Prompt construction
│   └── generation.py            # Gemma-2-9B-it inference
├── frontend/                    # React + Vite chat UI
│   ├── src/
│   └── README.md
└── main.py                      # Embedding pipeline entry point
```

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Backend API | FastAPI |
| Vector database | ChromaDB |
| Embedding model | BAAI/bge-m3 (Hugging Face Transformers) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Generation model | google/gemma-2-9b-it |
| Frontend | React (Vite) |
| Containerization | Docker |
| Development/validation | Google Colab (A100 GPU) |

## Data Pipeline

```
JSON Documents (clause + BOQ chunks)
        │
        ▼
Metadata Extraction (metadata_loader.py)
        │
        ▼
Text Normalization (text_normalization.py)
        │
        ▼
Clause / BOQ Chunking
        │
        ▼
Embedding Generation (BAAI/bge-m3)
        │
        ▼
ChromaDB Storage
```

## Retrieval Pipeline

```
User Query
        │
        ▼
Dense Retrieval (BGE-M3, ChromaDB)
        │
        ▼
BM25 Retrieval (lexical)
        │
        ▼
Hybrid Search (fusion of both result sets)
        │
        ▼
Cross-Encoder Reranking (BGE-reranker-v2-m3)
        │
        ▼
Prompt Engineering (context + citation instructions)
        │
        ▼
Gemma-2-9B-it Inference
        │
        ▼
Final Answer
```

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd dmrc-contract-intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend
npm install
cd ..
```

The first run downloads `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, and `google/gemma-2-9b-it` from Hugging Face. `google/gemma-2-9b-it` is a **gated** model — accept its license at https://huggingface.co/google/gemma-2-9b-it and set `HF_TOKEN` before running.

## Running the Project

**1. Run the embedding pipeline** (populates ChromaDB):

```bash
python main.py --input-dir data/
```

**2. Start the FastAPI server:**

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

**3. Query the system:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the scope of work under Clause 5 of the ECS contract?"}'
```

**4. Run the frontend:**

```bash
cd frontend
npm run dev
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Submit a natural-language query and receive a generated, context-grounded answer |
| `GET` | `/status` | Health check — reports ChromaDB connection and model load status (`dense_model_loaded`, `reranker_model_loaded`, `gemma_model_loaded`) |

## Future Improvements

- Support for additional DMRC contract lots and document types
- User authentication and role-based access control
- Conversation history / multi-turn context
- Answer confidence scoring surfaced in the UI
- Support for alternative vector stores (e.g., Qdrant)
- Automated evaluation suite for retrieval and answer quality

## Author

**Name:** _[Your Name]_
**Organization:** _[Your Organization]_
**Contact:** _[Your Email]_
**GitHub:** _[Your GitHub Profile]_

---

## Summary of Changes from Previous README

- Reframed the document from a single "Chapter 7 — Embedding Generation Module" write-up into a full, standalone Enterprise RAG platform README covering the entire system (retrieval, reranking, generation, API, frontend).
- Removed all references to chapters, notebooks-as-source-of-truth, and development history; the system is now presented as a complete, production-ready application.
- Added dedicated **Features**, **System Architecture**, **Data Pipeline**, **Retrieval Pipeline**, and **API Endpoints** sections that were not previously documented in one place.
- Documented the full retrieval stack (dense + BM25 + hybrid + cross-encoder reranking) and generation stage (Gemma-2-9B-it), which were previously only implied via the Colab notebooks and deployment notes.
- Replaced the single-module folder structure with a project-wide structure reflecting `src/` (retrieval, reranking, prompt building, generation), `frontend/`, and `Dockerfile` at the project root.
- Consolidated installation and running instructions into one flow covering both backend and frontend, and both the embedding pipeline and the API server.
- Added a placeholder **Author** section and GitHub-style badges for a professional presentation.
- Removed Colab/RunPod/Vercel deployment mechanics and internal migration notes (kept only the technologies actually used) to keep the README focused and concise.