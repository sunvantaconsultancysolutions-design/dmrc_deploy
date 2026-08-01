"""
app.py

Chapter 12 -- FastAPI Implementation.

Pure orchestration layer: this module does not retrieve, rerank, or
build prompts itself -- it wires together the three pipeline stages
that already exist, unchanged:

    User Query
        |
        v
    hybrid_retriever.hybrid_search()   (Chapter 9  -- dense + BM25 + merge)
        |
        v
    reranker.rerank()                  (Chapter 10 -- BGE cross-encoder)
        |
        v
    prompt_engineering.build_prompt()  (Chapter 11 -- prompt assembly)
        |
        v
    JSON response

Gemma 3 inference (Chapter 12.12's final "Gemma 3 generates the
answer" step) is intentionally NOT wired in yet -- per this chapter's
scope, the /ask endpoint returns the fully-assembled prompt string as
"answer" so the retrieval -> rerank -> prompt pipeline can be verified
end-to-end over the API before the LLM call exists.

------------------------------------------------------------------------
12.6 API Endpoints implemented in this module
------------------------------------------------------------------------
    GET  /        Health Check
    POST /ask     Question Answering (returns the built prompt, not yet an LLM answer)
    GET  /status  System Status

/upload is intentionally NOT implemented here -- per this task's scope,
document ingestion is handled by separate ingestion scripts, not the
API layer.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import retrieval_caps  # side-effect import, must run before
                                # hybrid_search/rerank/expand_with_siblings/
                                # get_chunks_by_parent_clause are imported
                                # below so RAG_MAX_CANDIDATES/RAG_MAX_CONTEXT
                                # actually cap retrieval breadth (see
                                # retrieval_caps.py's docstring). Also
                                # referenced directly below (retrieval_caps.MAX_CONTEXT)
                                # as a final backstop before prompt construction.
from .bm25_index import rebuild_bm25_index
from .hybrid_retriever import hybrid_search
from .prompt_engineering import (
    NO_CONTEXT_ANSWER,
    build_prompt_with_context,
    get_boq_item_number,
    get_boq_page_number,
    get_document_name,
    get_scanned_page,
    has_usable_context,
    wants_detailed_answer,
)
from .query import get_model as get_dense_model
from .query import extract_clause_no, get_chunk_by_clause_no, get_chunks_by_parent_clause
from .query import extract_boq_item_no, get_chunk_by_boq_item_no
from .reranker import evaluate_confidence, expand_with_siblings, get_reranker_model, rerank
from . import query as query_module
from . import reranker as reranker_module
from .gemma_inference import generate_answer, get_gemma_model
from .gemma_inference import MAX_NEW_TOKENS as DETAILED_MAX_NEW_TOKENS
from . import gemma_inference as gemma_module

# ---------------------------------------------------------------------------
# TASK 4 -- Query-intent router (new module; imported here so it can be
# called in the /ask endpoint before hybrid_search()).  The router is a
# pure classification function: it does not call any retrieval code, does
# not load any model, and does not change any existing retrieval logic.
# Its only effect is to pass a metadata_filter dict to hybrid_search()
# when the query intent is unambiguously CLAUSE or BOQ.
# ---------------------------------------------------------------------------
from .query_router import classify_query

# Demo-day performance change: most questions now get the concise
# (5-8 bullet, ~150-250 word) instructions from prompt_engineering.py,
# which need far fewer generated tokens than the original long-form
# answers -- generation cost scales with tokens actually produced, so
# capping this lower directly cuts response latency for the common
# case. wants_detailed_answer() switches both the instructions AND this
# cap back to the full DETAILED_MAX_NEW_TOKENS whenever the user's own
# wording asks for a fuller answer. Overridable via GEMMA_CONCISE_MAX_NEW_TOKENS
# the same way MAX_NEW_TOKENS itself is overridable, for easy tuning
# without a code change.
CONCISE_MAX_NEW_TOKENS = int(os.environ.get("GEMMA_CONCISE_MAX_NEW_TOKENS", "512"))

logger = logging.getLogger("dmrc_rag.api")

# ---------------------------------------------------------------------------
# TASK 4 -- Debug logging flag.
#
# Off by default (production-safe). Set RAG_DEBUG=1 in the environment
# to print the clause_id list surviving each pipeline stage for every
# /ask request. hybrid_retriever.py and reranker.py read this same env
# var independently (see their own module-level RAG_DEBUG) so each
# stage logs itself right where its output is computed, instead of
# app.py reaching into their internals.
# ---------------------------------------------------------------------------
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


def _debug_clause_block(header: str, candidates: List[Dict[str, Any]], score_key: Optional[str] = None) -> None:
    """Prints a 'clause_id [+ score]' block for one pipeline stage.
    Only ever called when RAG_DEBUG is on -- see call sites below.
    """
    print("=" * 22)
    print(header)
    print("=" * 22)
    if not candidates:
        print("(none)")
        return
    for c in candidates:
        clause_no = (c.get("metadata") or {}).get("clause_no", "N/A")
        chunk_id = c.get("chunk_id", "N/A")
        if score_key and c.get(score_key) is not None:
            print(f"  {chunk_id}  clause={clause_no}  {score_key}={c[score_key]}")
        else:
            print(f"  {chunk_id}  clause={clause_no}")


# ---------------------------------------------------------------------------
# 12.12 / 9.12 / 14.7 Retrieval + Reranking Parameters
#
# Call-site overrides of hybrid_retriever.py's and reranker.py's own
# defaults -- NOT edits to those files. These match Table 9.2 / Table
# 14.1's recommended production values (wider candidate pool feeding a
# smaller reranked set) rather than hybrid_retriever.py's smaller
# CLI-friendly defaults.
# ---------------------------------------------------------------------------

# AUDIT FOLLOW-UP (broad-query recall fix): widened from the original
# 20/20/40/10 values. Forensic audit confirmed via a live BM25 run that
# clause 6.8 (parent) and 6.8.3 ranked 25th and 30th for the query
# "What spare parts, tools, and test equipment must the contractor
# provide?" -- outside the old BM25_TOP_K=20 cutoff, so they never
# reached hybrid merge. Raising BM25_TOP_K/DENSE_TOP_K to 30 pulls both
# back into the candidate pool. MERGED_CANDIDATE_POOL is raised in step
# so the wider dense+BM25 output isn't immediately re-truncated. This
# does NOT fix clauses 6.8.5/6.8.6 (ranked 58th/59th) -- those are
# addressed separately by parent_clause sibling expansion, see
# reranker.py::expand_with_siblings().
DENSE_TOP_K = 30        # was 20 -- Table 9.2: Dense Top-k
BM25_TOP_K = 30         # was 20 -- Table 9.2: BM25 Top-k
MERGED_CANDIDATE_POOL = 60   # was 40 -- Table 9.2 / 14.7: Merged Candidates / Retrieved Documents
RERANK_TOP_N = 12       # was 10 -- Table 9.2 / 14.7: Final Re-ranked / Re-ranked Documents


# ---------------------------------------------------------------------------
# 12.3 / 12.10 Pydantic Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """12.7 Request Model. `query` is the only required field, matching
    the design doc's example exactly: a bare {"query": "..."} request.

    SECURITY/COST FIX (post-audit): previously had no length cap, so a
    pathologically long query could reach the tokenizer/reranker
    uncapped, inflating latency and cost for a single request. 2000
    characters is generous for any real contract question (the longest
    realistic query -- quoting a clause back verbatim -- is nowhere
    close to this) while bounding the worst case. FastAPI/Pydantic
    reject over-length requests with a 422 automatically.
    """

    query: str = Field(..., max_length=2000)


class SourceItem(BaseModel):
    """One entry in the response's "sources" list, populated from a
    reranked candidate's metadata (Chapter 6/9's metadata schema).
    Fields are optional because not every chunk carries every field
    (e.g. a contract clause has no clause-level "own" BOQ item number,
    only an optional cross-reference to one; a BOQ row has no
    clause_no). See _build_sources() for how `item_number` and `page`
    are resolved differently for clause vs. BOQ candidates.
    """

    clause: Optional[str] = None
    page: Optional[str] = None  # CHANGED: stamped scan number (Rule 1)
    pdf_page: Optional[int] = None  # NEW: internal file-lookup key (Rule 2)
    document: Optional[str] = None
    document_id: Optional[str] = None  # NEW
    image_url: Optional[str] = None  # /pages/{doc_id}/pNNNN.jpg
    figure_urls: Optional[list] = None  # figures found on this page
    # TASK 2 -- neighboring page image URLs.
    # prev_image_url and next_image_url carry the /pages/{doc_id}/p{N-1:04d}.jpg
    # and /pages/{doc_id}/p{N+1:04d}.jpg URLs respectively, when those
    # rendered files exist on disk (verified against AVAILABLE_PAGES at
    # request time, same check as image_url). Both are None when the
    # adjacent page has no render (first/last page, or unrendered source).
    # These are NEW optional fields; any existing client that ignores them
    # keeps working unchanged.
    prev_image_url: Optional[str] = None  # TASK 2: /pages/{doc_id}/p{pdf_page-1:04d}.jpg
    next_image_url: Optional[str] = None  # TASK 2: /pages/{doc_id}/p{pdf_page+1:04d}.jpg
    item_number: Optional[str] = None
    retrieval_source: Optional[str] = None
    reranker_score: Optional[float] = None
    chunk_id: Optional[str] = None
    max_pdf_page: Optional[int] = None  # total pages available for this doc_id
    # PHASE 2 (Evidence Viewer) -- richer evidence cards.
    # Both fields are populated from metadata that already exists in
    # ChromaDB (clause "heading", BOQ chunk's own indexed text) -- no
    # ingestion/embedding/retrieval change, purely additive on the
    # response side. Optional/back-compat: None when not applicable
    # (e.g. a clause chunk with no transcribed heading, or when the
    # candidate's own metadata simply doesn't carry one).
    heading: Optional[str] = None       # clause title, e.g. "Penalty Clause"
    description: Optional[str] = None   # short BOQ item description (~100 chars)
    # TASK 3 -- UI retrieval label fix. Exposes the existing
    # metadata["chunk_type"] ("clause" or "boq", set unconditionally by
    # metadata_loader.py for every chunk) so the frontend can render
    # "Grounded in N Retrieved Clauses" / "... BOQ Items" / "...
    # Documents" (mixed) from real retrieval metadata instead of the
    # hardcoded "clauses" wording it used before. Optional/back-compat:
    # any existing client that ignores this new field keeps working
    # unchanged.
    chunk_type: Optional[str] = None
    # TASK 4 -- exposes the routing decision for debug / frontend display.
    # "clause" | "boq" | "general" | "exact_clause" | "exact_boq"
    # Optional/back-compat: None when not applicable (exact-match paths).
    query_intent: Optional[str] = None


class AnswerResponse(BaseModel):
    """12.8 Response Model. `confidence` is left as None for now since
    no LLM inference (and therefore no confidence signal) exists yet
    in this chapter's scope.
    """

    answer: str
    sources: List[SourceItem]
    confidence: Optional[float] = None
    # TASK 4 -- top-level routing metadata for callers that want to know
    # which retrieval pool was searched. Optional so existing clients
    # that ignore it keep working unchanged.
    query_intent: Optional[str] = None
    routing_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    """12.6 GET /status -- System Status. Reports whether the heavy
    models are actually loaded in memory yet, rather than just
    asserting the process is up, since model loading is the slow /
    failure-prone part of startup (Section 13.15's memory/GPU caveat).
    Also reports whether the ChromaDB connection backing retrieval is
    reachable, since a "running" process with a dead vector store
    would otherwise silently fail on the first /ask request.
    """

    status: str
    embedding_model: str
    reranker_model: str
    dense_model_loaded: bool
    reranker_model_loaded: bool
    gemma_model_loaded: bool
    chromadb_connected: bool


# ---------------------------------------------------------------------------
# 14.9 Model warm-up during startup (lifespan) -- load the dense
# embedding model and the reranker model once, here, so the first real
# request isn't the one paying the (multi-second, sometimes
# multi-minute on first download) model-load cost. Failures are
# logged, not raised: a slow/failed warm-up shouldn't prevent the API
# from starting, it should just mean the first /ask request pays the
# load cost lazily instead (both get_model() and get_reranker_model()
# are safe to call again on demand).
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Warming up embedding model and reranker model...")
    try:
        get_dense_model()
    except Exception:
        logger.exception("Dense embedding model warm-up failed; will retry lazily on first request.")
    try:
        get_reranker_model()
    except Exception:
        logger.exception("Reranker model warm-up failed; will retry lazily on first request.")
    try:
        get_gemma_model()
    except Exception:
        logger.exception(
            "Gemma 3 warm-up failed; will retry lazily on first request."
        )
    _scan_page_images()
    _load_figure_manifest()
    _check_stamp_integrity()
    # SECURITY FIX (post-audit): the open-by-default /admin/reload-bm25
    # tradeoff is deliberate for a single private RunPod pod (see that
    # endpoint's own comment), but nothing previously surfaced the fact
    # that it's currently open. A log line at startup is cheap insurance
    # against an operator forgetting to set ADMIN_API_KEY before a
    # deploy that's no longer single-tenant/private.
    if not ADMIN_API_KEY:
        logger.warning(
            "ADMIN_API_KEY is not set -- /admin/reload-bm25 is open with no "
            "authentication. Set ADMIN_API_KEY before deploying anywhere "
            "other than a single, private, trusted-network pod."
        )
    yield


# ---------------------------------------------------------------------------
# 12.9 FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DMRC Contract Intelligence",
    version="1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS -- the React frontend is deployed separately (e.g. Vercel) from this
# API (e.g. a RunPod GPU pod), so browser requests are cross-origin by
# definition. ALLOWED_ORIGINS is a comma-separated env var so the deployed
# frontend URL never has to be hardcoded here; "*" is only a local-dev
# fallback and should be overridden in production.
# ---------------------------------------------------------------------------

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# PDF Evidence Viewer (SVS-DMRC-2026-03) -- scanned page images.
#
# RULE 2: pdf_page remains the file-lookup key. Image files are named
# by (document_id, pdf_page). AVAILABLE_PAGES is populated once at
# startup so a missing render degrades _build_sources() to a chip
# without a viewer link instead of a broken image, rather than being
# checked against the filesystem on every request.
#
# TASK 1 FIX: the BOQ image_url previously returned None for every BOQ
# chunk because the page_images/ directory names (e.g.
# "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-28-37") did not match
# the document_id values in ChromaDB (e.g.
# "BOQ-CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-28-37").  The fix is
# in two parts:
#   1. scripts/render_pages.py now derives BOQ directory names via
#      _slugify(pdf_filename_stem), which is identical to what
#      metadata_loader.py computes for image_document_id.  Running
#      scripts/render_pages.py again produces correctly named directories.
#   2. scripts/migrate_page_image_dirs.py renames the existing misnamed
#      directories on disk so that deployments with pre-rendered images
#      do not need to re-render everything.
# No changes to this function are needed: _scan_page_images() reads
# whatever directories exist on disk, so once the directories are
# correctly named the AVAILABLE_PAGES set automatically contains the
# right (document_id, pdf_page) pairs.
# ---------------------------------------------------------------------------

PAGE_IMAGES_DIR = os.environ.get("PAGE_IMAGES_DIR", "page_images")
AVAILABLE_PAGES: set = set()    # set[tuple[str, int]]
_MAX_PDF_PAGE: dict = {}        # doc_id -> highest page number with a rendered image


def _scan_page_images() -> None:
    """Populate AVAILABLE_PAGES and _MAX_PDF_PAGE from PAGE_IMAGES_DIR at startup."""
    AVAILABLE_PAGES.clear()
    _MAX_PDF_PAGE.clear()
    if not os.path.isdir(PAGE_IMAGES_DIR):
        logger.warning("page images dir %s not found - viewer disabled", PAGE_IMAGES_DIR)
        return
    for doc_id in os.listdir(PAGE_IMAGES_DIR):
        doc_dir = os.path.join(PAGE_IMAGES_DIR, doc_id)
        if not os.path.isdir(doc_dir):
            continue
        for fname in os.listdir(doc_dir):
            if fname.startswith("p") and fname.endswith(".jpg"):
                try:
                    page_num = int(fname[1:5])
                    AVAILABLE_PAGES.add((doc_id, page_num))
                    if page_num > _MAX_PDF_PAGE.get(doc_id, 0):
                        _MAX_PDF_PAGE[doc_id] = page_num
                except ValueError:
                    pass
    logger.info("evidence viewer: %d page images available across %d documents",
                len(AVAILABLE_PAGES), len(_MAX_PDF_PAGE))


def _check_stamp_integrity() -> None:
    """Warn on empty, duplicate, or non-contiguous scan stamps.

    The stamped scan number forms one continuous global sequence
    across a volume's chapter files, so gaps and duplicates are a
    reliable OCR-misread detector. Log warnings only -- never fail
    startup -- and review the log after each ingestion change.
    """
    try:
        collection = query_module.get_collection()
        res = collection.get(include=["metadatas"])
    except Exception as exc:  # pragma: no cover
        logger.warning("stamp check skipped: %s", exc)
        return

    seen: Dict[int, tuple] = {}
    empty = 0
    for md in res["metadatas"]:
        stamp = md.get("printed_page") or md.get("stamp_number")
        loc = (md.get("document_id"), md.get("pdf_page"))
        if stamp in (None, ""):
            empty += 1
            continue
        try:
            n = int(str(stamp))
        except ValueError:
            logger.warning("non-numeric stamp %r at %s", stamp, loc)
            continue
        if n in seen and seen[n] != loc:
            logger.warning("duplicate stamp %06d: %s vs %s", n, seen[n], loc)
        seen[n] = loc

    nums = sorted(seen)
    gaps = [f"{a:06d}->{b:06d}" for a, b in zip(nums, nums[1:]) if b - a > 1]
    logger.info(
        "stamp check: %d stamped, %d unstamped chunks, %d gap(s)%s",
        len(seen), empty, len(gaps), f" {gaps[:5]}" if gaps else "",
    )


if os.path.isdir(PAGE_IMAGES_DIR):
    app.mount("/pages", StaticFiles(directory=PAGE_IMAGES_DIR), name="pages")


@app.get("/pages/manifest")
def pages_manifest() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc_id, _p in AVAILABLE_PAGES:
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Figure/diagram images (scripts/extract_page_figures.py) -- distinct from
# the whole-page renders above. FIGURE_MANIFEST is a dict keyed by
# (document_id, pdf_page) -> list of figure filenames, loaded once at
# startup from figure_images/manifest.json (written by that script).
#
# TASK 3 STATUS -- figure retrieval is fully implemented in this file and
# in scripts/extract_page_figures.py; the API endpoints (/figures,
# /figures/manifest, figure_urls in SourceItem) are all present and
# functional. However, the source PDFs in this corpus (BE-12/BE-14 Vol-2
# and CE-10/CE-11 Vol-3) are fully-scanned photocopies: every page is a
# single large raster image, so PyMuPDF's page.get_images() returns only
# that full-page scan, which extract_page_figures.py correctly rejects via
# AREA_RATIO_THRESHOLD (>60% of page area). Running the script produces 0
# extracted figures.  This is correct behavior: there are no embedded
# sub-figures to extract. figure_images/manifest.json is therefore empty
# for this corpus.
#
# The implementation is complete; it simply has no assets to serve for the
# current source documents. When new documents with embedded diagrams are
# ingested, running scripts/extract_page_figures.py will populate the
# manifest and activate the feature with no code changes required.
# ---------------------------------------------------------------------------

import json  # noqa: E402 -- kept as the only json import in this module

FIGURE_IMAGES_DIR = os.environ.get("FIGURE_IMAGES_DIR", "figure_images")
FIGURE_MANIFEST: Dict[tuple, list] = {}  # {(document_id, pdf_page): [filenames]}


def _load_figure_manifest() -> None:
    """Populate FIGURE_MANIFEST from figure_images/manifest.json at
    startup. Missing file/dir degrades to an empty manifest -- same
    graceful-degradation convention as _scan_page_images() above, since
    a document with no extracted figures is the common case, not an
    error.
    """
    FIGURE_MANIFEST.clear()
    manifest_path = os.path.join(FIGURE_IMAGES_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        logger.warning("figure manifest %s not found - figures disabled", manifest_path)
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    total = 0
    for doc_id, entries in raw.items():
        for entry in entries:
            key = (doc_id, entry["pdf_page"])
            FIGURE_MANIFEST.setdefault(key, []).append(entry["filename"])
            total += 1
    logger.info("evidence viewer: %d figures available across %d documents", total, len(raw))


if os.path.isdir(FIGURE_IMAGES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURE_IMAGES_DIR), name="figures")


@app.get("/figures/manifest")
def figures_manifest() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for (doc_id, _pdf_page), filenames in FIGURE_MANIFEST.items():
        counts[doc_id] = counts.get(doc_id, 0) + len(filenames)
    return counts


# ---------------------------------------------------------------------------
# GET / -- 12.6 Health Check
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


# ---------------------------------------------------------------------------
# Helpers -- ChromaDB connectivity check for GET /status
# ---------------------------------------------------------------------------

def _check_chromadb_connected() -> bool:
    """Best-effort check of whether the ChromaDB connection used by
    query.py is up. Deliberately defensive/read-only: any failure to
    reach or introspect the client is treated as "not connected"
    rather than raised, since /status must never itself 500 just
    because the database happens to be down.
    """
    try:
        get_collection = getattr(query_module, "get_collection", None)
        if callable(get_collection):
            collection = get_collection()
            collection.count()
            return True

        client = getattr(query_module, "_client", None) or getattr(query_module, "client", None)
        if client is not None:
            client.heartbeat()
            return True

        collection = getattr(query_module, "_collection", None) or getattr(query_module, "collection", None)
        if collection is not None:
            collection.count()
            return True

        logger.error("No known ChromaDB client/collection accessor found on query module.")
        return False
    except Exception:
        logger.exception("ChromaDB connectivity check failed.")
        return False


# ---------------------------------------------------------------------------
# GET /status -- 12.6 System Status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
def system_status() -> StatusResponse:
    return StatusResponse(
        status="running",
        embedding_model=query_module.MODEL_NAME,
        reranker_model=reranker_module.MODEL_NAME,
        dense_model_loaded=query_module._model is not None,
        reranker_model_loaded=reranker_module._model is not None,
        gemma_model_loaded=gemma_module._model is not None,
        chromadb_connected=_check_chromadb_connected(),
    )


# ---------------------------------------------------------------------------
# Helpers -- building the "sources" list from reranked metadata (12.8)
# ---------------------------------------------------------------------------

def _resolve_page_image_url(doc_id: Optional[str], pdf_page: Any) -> Optional[str]:
    """Return the /pages/{doc_id}/p{N:04d}.jpg URL if that image exists
    in AVAILABLE_PAGES, else None. Centralised so _build_sources() can
    call it for the current page, the previous page, and the next page
    without repeating the int-cast / AVAILABLE_PAGES check.
    """
    if not doc_id or pdf_page in (None, ""):
        return None
    try:
        p = int(pdf_page)
    except (TypeError, ValueError):
        return None
    if (doc_id, p) in AVAILABLE_PAGES:
        return f"/pages/{doc_id}/p{p:04d}.jpg"
    return None


def _build_sources(
    reranked_candidates: List[Dict[str, Any]],
    query_intent: Optional[str] = None,
) -> List[SourceItem]:
    sources: List[SourceItem] = []
    for candidate in reranked_candidates:
        metadata = candidate.get("metadata") or {}

        # BUGFIX: BOQ metadata field mapping.
        #
        # A BOQ row's own item number lives under the "s_no" metadata
        # field, not "item_number" -- "item_number" only exists on
        # CLAUSE metadata, as an optional cross-reference to a related
        # BOQ item. Previously this function always read
        # metadata.get("item_number"), which is correct for clauses
        # but returns None for every BOQ-sourced answer, so BOQ
        # citations never reached the frontend even though
        # prompt_engineering.format_context() was already rendering
        # them correctly for the LLM prompt.
        #
        # Similarly, a BOQ row's page can be under "page_number" if it
        # doesn't carry "pdf_page" -- get_boq_page_number() applies the
        # same fallback prompt_engineering.py already used when
        # formatting BOQ blocks for the prompt.
        #
        # Both accessors are imported from prompt_engineering.py so
        # this stays in sync with prompt formatting by construction,
        # rather than duplicating (and re-diverging from) the lookup.
        is_boq = metadata.get("chunk_type") == "boq"
        item_number = get_boq_item_number(metadata) if is_boq else metadata.get("item_number")

        # Rule 1: display/citation number = the stamp on the scanned page.
        scanned = get_scanned_page(metadata)

        # Rule 2: file lookup stays on the PDF index, never the stamp --
        # stamps are empty on cover pages, arrive as strings with leading
        # zeros, and can be misread by OCR. A wrong stamp must only ever
        # produce a wrong label, never a wrong page image.
        pdf_page = metadata.get("pdf_page")
        if pdf_page in (None, ""):
            pdf_page = metadata.get("page_number") if is_boq else None

        doc_id = metadata.get("document_id")

        # Current page image (unchanged from original).
        image_url = _resolve_page_image_url(doc_id, pdf_page)

        # TASK 2 -- Neighboring page images.
        # Resolve previous and next pages using the same AVAILABLE_PAGES
        # check so we only return URLs for images that actually exist on
        # disk. Both are None when the neighbor page has no render
        # (first/last page of a document, or an unrendered source).
        prev_image_url: Optional[str] = None
        next_image_url: Optional[str] = None
        if pdf_page not in (None, ""):
            try:
                p = int(pdf_page)
                prev_image_url = _resolve_page_image_url(doc_id, p - 1)
                next_image_url = _resolve_page_image_url(doc_id, p + 1)
            except (TypeError, ValueError):
                pass

        # Parallel to image_url above, but for embedded figures/diagrams
        # (scripts/extract_page_figures.py) rather than the whole-page
        # scan render. Uses the raw (unvalidated-against-AVAILABLE_PAGES)
        # pdf_page directly since FIGURE_MANIFEST.get() on a missing key
        # simply returns [], same graceful-degradation convention as
        # AVAILABLE_PAGES above.
        figure_urls = [
            f"/figures/{doc_id}/{fname}"
            for fname in FIGURE_MANIFEST.get((doc_id, pdf_page), [])
        ] if doc_id and pdf_page not in (None, "") else []

        # Compute max available page for this document so the frontend
        # can disable the next-page button when the user reaches the end.
        max_pdf_page = _MAX_PDF_PAGE.get(doc_id) if doc_id else None

        # PHASE 2 (Evidence Viewer) -- richer evidence cards (Feature 3).
        # Clause title: metadata["heading"] already exists (transcribed
        # per-clause, e.g. "Penalty Clause", "Scope and Purpose") but was
        # never exposed on the wire. Empty string is normalised to None
        # so the frontend's `heading || fallback` logic works the same
        # way it already does for every other optional field here.
        heading = metadata.get("heading") or None if not is_boq else None

        # BOQ short description: no separate "description" metadata field
        # exists, but the chunk's own indexed body text (candidate["document"]
        # -- the same text already shown in the answer/prompt context) is a
        # natural short description. Truncated to keep the evidence card
        # compact; not a retrieval change, purely a display-side excerpt of
        # data already retrieved for this candidate.
        description = None
        if is_boq:
            body_text = (candidate.get("document") or "").strip()
            if body_text:
                description = body_text[:100] + ("…" if len(body_text) > 100 else "")

        sources.append(
            SourceItem(
                clause=metadata.get("clause_no"),
                page=None if scanned in (None, "") else str(scanned),
                pdf_page=int(pdf_page) if pdf_page not in (None, "") else None,
                document=get_document_name(metadata),
                document_id=doc_id,
                image_url=image_url,
                figure_urls=figure_urls,
                prev_image_url=prev_image_url,   # TASK 2
                next_image_url=next_image_url,   # TASK 2
                item_number=item_number,
                retrieval_source=candidate.get("retrieval_source"),
                reranker_score=candidate.get("reranker_score"),
                chunk_id=candidate.get("chunk_id") or metadata.get("chunk_id"),
                chunk_type=metadata.get("chunk_type"),
                query_intent=query_intent,       # TASK 4
                max_pdf_page=max_pdf_page,       # navigation bound for PageViewer
                heading=heading,                 # PHASE 2: clause title
                description=description,         # PHASE 2: short BOQ description
            )
        )
    return sources


# ---------------------------------------------------------------------------
# POST /admin/reload-bm25 -- BUGFIX: manual fix for BM25 staleness.
#
# bm25_index.get_bm25_index() builds once per process and caches the
# result; dense search hits ChromaDB live on every call so it always
# sees new data, but BM25 does not. Call this endpoint once after any
# ingestion run (e.g. once BOQ rows are added) so BM25 catches up
# without needing to restart the whole server.
#
# QA REVIEW (Issue 6) -- deployment note: this endpoint was previously
# unauthenticated by design for this project's current deployment model
# (a single private RunPod pod, called only by the operator and by the
# Vercel frontend via /ask, with ALLOWED_ORIGINS gating browser access).
# It is read-only against ChromaDB (rebuilds an in-memory BM25 index;
# never writes to the database), so its worst-case impact if reached is
# a wasted rebuild, not data loss.
#
# SECURITY FIX (pre-deployment review): added a lightweight, OPT-IN API
# key check rather than a new auth system. If the ADMIN_API_KEY
# environment variable is set, this endpoint requires a matching
# `X-Admin-Key` header (401 otherwise). If ADMIN_API_KEY is unset
# (unset by default -- e.g. local dev, or any deployment that hasn't
# opted in yet), the endpoint remains open exactly as before, so
# existing deployments are not broken by this change. Set ADMIN_API_KEY
# at `docker run` / deploy time to require the header in production.
# ---------------------------------------------------------------------------

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


def _require_admin_key(x_admin_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: no-op (endpoint stays open) when ADMIN_API_KEY
    isn't configured; otherwise requires `X-Admin-Key` to match it.
    """
    if not ADMIN_API_KEY:
        return
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key header.")


class ReloadBM25Response(BaseModel):
    status: str
    chunks_indexed: int


@app.post(
    "/admin/reload-bm25",
    response_model=ReloadBM25Response,
    dependencies=[Depends(_require_admin_key)],
)
def reload_bm25() -> ReloadBM25Response:
    try:
        index = rebuild_bm25_index()
    except Exception as exc:
        logger.error("BM25 index rebuild failed.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"BM25 rebuild failed: {exc}") from exc
    return ReloadBM25Response(status="reloaded", chunks_indexed=len(index.chunk_ids))


# ---------------------------------------------------------------------------
# POST /ask -- 12.6 / 12.11 / 12.12 Question Answering Endpoint
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QueryRequest) -> AnswerResponse:
    """12.12 RAG Pipeline Integration (retrieval + rerank + prompt +
    Gemma 3 generation):

        1. Validate the request.                       (FastAPI + Pydantic)
        2. NEW: exact clause-number fast path.          -> get_chunk_by_clause_no()
           If the query names a clause number found verbatim in
           ChromaDB's metadata, use that match directly and skip
           hybrid retrieval entirely. Otherwise fall through to step 3
           exactly as before.
        3. NEW: exact BOQ-item fast path.               -> get_chunk_by_boq_item_no()
           Only runs if step 2 found nothing. If the query names a BOQ
           item identifier found verbatim in ChromaDB's metadata (`parent`,
           `s_no`, `item_header_no`, or `section_no`), use that match
           directly and skip hybrid retrieval entirely. Otherwise fall
           through to step 4 exactly as before.
        4. TASK 4: Query-intent routing.               -> query_router.classify_query()
           Classify the query as CLAUSE, BOQ, or GENERAL and derive a
           metadata_filter for hybrid_search(). Only runs when neither
           fast path fired (steps 2-3). Existing behaviour is preserved
           for all exact-match paths; routing only changes the
           metadata_filter passed to hybrid_search().
        5. Hybrid retrieval over ChromaDB + BM25.       -> hybrid_search()
        6. Rerank the candidates.                       -> rerank()
        7. Build the final LLM prompt.                  -> build_prompt()
        8. Generate the answer.                         -> generate_answer()
        9. Return the answer + sources as JSON.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="`query` must not be empty.")

    # ----------------------------------------------------------------
    # NEW: Exact clause-number fast path (see query.py's
    # extract_clause_no()/get_chunk_by_clause_no() docstrings for the
    # full rationale). Dense + BM25 rank on chunk TEXT, not on the
    # `clause_no` metadata field, so an exact clause reference like
    # "Explain Clause 1.2.1" can still fail to surface that clause
    # first. Checking metadata directly for an exact match up front
    # fixes that without touching embeddings, BM25, or the merge logic.
    #
    # `candidates` is deliberately left empty (rather than raising) on
    # a clause-lookup failure or "clause named but not found in the
    # corpus" -- either case just means we fall back to the existing
    # hybrid_search() pipeline below, unchanged.
    # ----------------------------------------------------------------
    candidates: List[Dict[str, Any]] = []
    clause_no = extract_clause_no(query_text)
    if clause_no:
        try:
            candidates = get_chunk_by_clause_no(clause_no)
        except Exception:
            logger.exception("Exact clause lookup failed for clause_no: %r", clause_no)
            candidates = []

        # BUGFIX: a query naming a PARENT clause (e.g. "Explain Clause 6.8
        # in detail and summarize all of its sub-clauses") also matches
        # CLAUSE_NO_PATTERN on "6.8" and was taking this same exact-match
        # fast path -- which returns only the single 6.8 chunk and then
        # skips expand_with_siblings() entirely, since that only ever runs
        # on the hybrid_search()/rerank() branch below, never on this one.
        # Detect "clause_no has children" via the same metadata-only
        # accessor sibling expansion already uses (no embedding, no ANN,
        # near-free), and pull them in here. For a true leaf clause (e.g.
        # "1.2.1", nothing declares it as a parent_clause) this returns []
        # and candidates/exact_match behavior is unchanged from before.
        #
        # BUGFIX (pre-deployment review, confirmed Bug #1): this lookup
        # previously only ran when `candidates` was already non-empty
        # (i.e. only when a self-referencing clause_no row also exists,
        # as with "6.8"). That silently skipped clause families that have
        # NO self row at all -- e.g. "6.7.2", which only exists in this
        # corpus as children "6.7.2-1".."6.7.2-4" carrying
        # parent_clause=="6.7.2". A query like "Explain Clause 6.7.2" then
        # fell straight through to hybrid_search() instead of returning
        # its exact-match family. Running this lookup unconditionally
        # (not gated on `if candidates:`) covers both shapes -- families
        # with a self row (candidates starts non-empty, siblings are
        # added on top) and families with only children (candidates
        # starts empty and is populated entirely from this lookup) --
        # while a true leaf clause (e.g. "1.2.1", nothing declares it as
        # a parent_clause) still returns [] and is unaffected either way.
        try:
            children = get_chunks_by_parent_clause(clause_no)
        except Exception:
            logger.exception(
                "Child-clause lookup failed for parent clause_no: %r", clause_no
            )
            children = []
        if children:
            already_ids = {c["chunk_id"] for c in candidates}
            for child in children:
                if child["chunk_id"] in already_ids:
                    continue
                entry = dict(child)
                entry["score"] = 1.0
                entry["retrieval_source"] = "exact_clause_family_match"
                entry["dense_score"] = None
                entry["bm25_score"] = None
                candidates.append(entry)
                already_ids.add(child["chunk_id"])

    # ----------------------------------------------------------------
    # NEW: Exact BOQ-item fast path (see query.py's
    # extract_boq_item_no()/get_chunk_by_boq_item_no() docstrings for
    # the full rationale). Same problem as the clause fast path above,
    # for Bill-of-Quantities rows instead of clauses: dense + BM25 rank
    # on chunk TEXT, not on the `parent`/`s_no`/`item_header_no`/
    # `section_no` metadata fields, so an exact BOQ reference like
    # "Describe BOQ item 1.02.E.2" can still fail to surface the
    # correct item(s) first. Checking metadata directly for an exact
    # match up front fixes that without touching embeddings, BM25, or
    # the merge logic.
    #
    # Only attempted if the clause fast path above found nothing --
    # a query that already resolved via an exact clause match is a
    # clause query, not a BOQ query, and should not also be run through
    # the BOQ lookup. `candidates` is deliberately left empty (rather
    # than raising) on a BOQ-lookup failure or "identifier named but
    # not found in the corpus" -- either case just means we fall
    # through to the existing hybrid_search() pipeline below, unchanged.
    # ----------------------------------------------------------------
    if not candidates:
        boq_item_no = extract_boq_item_no(query_text)
        if boq_item_no:
            try:
                candidates = get_chunk_by_boq_item_no(boq_item_no)
            except Exception:
                logger.exception(
                    "Exact BOQ item lookup failed for item_no: %r", boq_item_no
                )
                candidates = []

    # BUGFIX: track whether `candidates` came from an exact-metadata
    # match (clause OR BOQ), not from hybrid_search()/rerank(). An exact
    # clause_no or BOQ-item hit is already the highest-confidence result
    # possible (score=1.0, set in get_chunk_by_clause_no /
    # get_chunk_by_boq_item_no) -- running it through the cross-encoder
    # anyway wastes a GPU call AND overwrites that confidence with a
    # near-zero raw reranker_score (the model is scoring one full
    # question against one short chunk in isolation, which legitimately
    # produces tiny logits -- observed 0.0011/0.0022 in testing). That
    # score then leaks into the API response's `sources[].reranker_score`
    # with nothing in the payload to explain it, making an exact match
    # look like ~0.1% confidence to any caller. Skip reranking entirely
    # on this path instead.
    exact_match = bool(candidates)

    # TASK 4 -- Query-intent routing for free-text queries.
    #
    # classify_query() is called ONLY when the exact-match fast paths
    # above found nothing -- those paths already know the intent
    # (clause or BOQ) from the explicit identifier, and adding a filter
    # would be redundant and could only narrow the already-correct result.
    #
    # For free-text queries, the router classifies the intent and returns
    # a metadata_filter that restricts hybrid_search() to the relevant
    # chunk_type. This prevents clause and BOQ chunks from competing in
    # one pool for unambiguous queries.  When the intent is GENERAL (no
    # clear signal), metadata_filter is None and hybrid_search() runs
    # unfiltered, exactly as before.
    routing_intent = "exact_clause" if (exact_match and clause_no) else (
        "exact_boq" if exact_match else None
    )
    routing_reason: Optional[str] = None
    retrieval_filter: Optional[dict] = None

    if not exact_match:
        try:
            intent_result = classify_query(query_text)
            routing_intent = intent_result.intent
            routing_reason = intent_result.reason
            retrieval_filter = intent_result.metadata_filter
            if RAG_DEBUG:
                logger.info(
                    "Query router: intent=%r  filter=%r  reason=%r",
                    routing_intent, retrieval_filter, routing_reason,
                )
        except Exception:
            # Routing is a best-effort enhancement; if it fails for any
            # reason, fall through to the unfiltered search that was
            # already working.
            logger.exception("Query router failed for query %r; continuing unfiltered.", query_text)
            retrieval_filter = None

        try:
            candidates = hybrid_search(
                query_text,
                top_k_dense=DENSE_TOP_K,
                top_k_bm25=BM25_TOP_K,
                final_top_k=MERGED_CANDIDATE_POOL,
                metadata_filter=retrieval_filter,   # TASK 4: may be None (unfiltered) or chunk_type filter
            )
        except Exception as exc:
            logger.error("Hybrid retrieval failed for query: %r", query_text, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if exact_match:
        reranked = [dict(c, reranker_score=c.get("score", 1.0)) for c in candidates]
    else:
        try:
            reranked = rerank(query_text, candidates, top_n=RERANK_TOP_N)
        except Exception as exc:
            logger.error("Reranking failed for query: %r", query_text, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Reranking failed: {exc}") from exc

        # ------------------------------------------------------------
        # TASK 6 -- low-confidence / out-of-domain short-circuit.
        #
        # Runs only on this branch -- the exact-clause-match path above
        # is already maximally confident by construction (score=1.0) and
        # must never be gated here. has_usable_context() below only
        # checks non-emptiness, which an off-topic query still passes
        # (rerank() always returns its top_n best-of-a-bad-lot
        # candidates); this checks whether those candidates are actually
        # good, using this query's own score distribution rather than a
        # single hand-picked constant. See reranker.py::evaluate_confidence
        # for the two signals used and the calibration caveat.
        # ------------------------------------------------------------
        confidence_info = evaluate_confidence(reranked)

        # ------------------------------------------------------------
        # PHASE 1 AUDIT FIX -- borderline-confidence recovery.
        #
        # Previously, a low-confidence result short-circuited straight
        # to NO_CONTEXT_ANSWER, and sibling expansion only ever ran
        # AFTER confidence had already passed -- i.e. as a bonus on top
        # of an already-good answer, never as a way to rescue a weak
        # one. But reranker.py::expand_with_siblings()'s own docstring
        # describes exactly the opposite failure mode: a broad question
        # about a whole clause family can score individually-low on
        # every member (siblings sharing almost no vocabulary with the
        # query) even though the family AS A WHOLE is a strong match.
        # That is precisely a "borderline confidence" situation this
        # step is meant to catch.
        #
        # Fix: if the first confidence check fails and there is at
        # least one candidate to group by parent_clause, attempt
        # sibling expansion once, then re-evaluate confidence on the
        # expanded list before giving up. This never fabricates
        # anything -- expand_with_siblings() only pulls in chunks that
        # already exist in the corpus and are cross-encoder-confirmed
        # relevant to the same query (see relevance_margin gating in
        # its own docstring). If the retry still isn't confident, the
        # request still correctly reports "not found".
        # ------------------------------------------------------------
        already_expanded = False
        if not confidence_info["confident"] and reranked:
            try:
                expanded = expand_with_siblings(query_text, reranked)
            except Exception:
                logger.exception(
                    "Sibling expansion (recovery path) failed for query: %r; "
                    "continuing without it.", query_text,
                )
                expanded = reranked

            retry_confidence = evaluate_confidence(expanded)
            if RAG_DEBUG:
                logger.info(
                    "Borderline confidence for query %r: initial=%s, after "
                    "sibling-expansion retry=%s",
                    query_text, confidence_info, retry_confidence,
                )
            if retry_confidence["confident"]:
                reranked = expanded
                confidence_info = retry_confidence
                already_expanded = True

        if not confidence_info["confident"]:
            if RAG_DEBUG:
                logger.info(
                    "Low-confidence retrieval for query %r: %s",
                    query_text, confidence_info,
                )
            return AnswerResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                confidence=confidence_info["top_score"],
                query_intent=routing_intent,
                routing_reason=routing_reason,
            )

        # ------------------------------------------------------------
        # TASK 3 -- parent_clause sibling expansion.
        #
        # Only applied on the hybrid_search()/rerank() path, not the
        # exact-clause-number fast path above: a query that already
        # named one specific clause is asking about that clause, not
        # its whole family, so expanding it would reintroduce the
        # "answers get diluted" failure mode this whole audit started
        # from. See reranker.py::expand_with_siblings() for the
        # relevance-gated expansion logic and its own docstring for why
        # this is safe.
        #
        # Skipped here if the borderline-confidence recovery step above
        # already expanded this exact `reranked` list -- calling it
        # again would be a harmless no-op (expand_with_siblings excludes
        # chunk_ids already present) but is unnecessary work.
        # ------------------------------------------------------------
        if not already_expanded:
            try:
                reranked = expand_with_siblings(query_text, reranked)
            except Exception:
                # Sibling expansion is a recall *enhancement*, not a
                # correctness requirement -- if it fails for any reason
                # (e.g. a ChromaDB hiccup), fall back to the reranker's
                # own output rather than failing the whole request.
                logger.exception("Sibling expansion failed for query: %r; continuing without it.", query_text)

    # QA FIX (Issue 2) -- final safety net. retrieval_caps.py now caps
    # hybrid_search(), rerank(), expand_with_siblings(), and
    # get_chunks_by_parent_clause() individually, but this one extra
    # check guarantees the configured context budget
    # (retrieval_caps.MAX_CONTEXT) is never exceeded by the prompt no
    # matter which combination of paths produced `reranked` -- e.g. the
    # exact-clause-match branch above concatenates get_chunk_by_clause_no()
    # (uncapped, normally 1 chunk) with the now-capped children list, so
    # this backstop is what turns "normally fine" into "guaranteed".
    if len(reranked) > retrieval_caps.MAX_CONTEXT:
        if RAG_DEBUG:
            logger.info(
                "Trimming final candidate list %d -> %d (retrieval_caps.MAX_CONTEXT) "
                "before prompt construction.", len(reranked), retrieval_caps.MAX_CONTEXT,
            )
        reranked = reranked[: retrieval_caps.MAX_CONTEXT]

    if RAG_DEBUG:
        _debug_clause_block("Final Prompt (clauses sent to Gemma)", reranked, score_key="reranker_score")

    # 11.14 Handling Missing Information -- no reason to build a prompt
    # (or call an LLM) if retrieval found nothing at all.
    if not has_usable_context(reranked):
        return AnswerResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            confidence=None,
            query_intent=routing_intent,
            routing_reason=routing_reason,
        )

    # Demo-day change: default every answer to the concise (5-8 bullet,
    # ~150-250 word) instructions and a lower generation cap, unless the
    # user's own wording asks for a fuller answer (see
    # wants_detailed_answer() in prompt_engineering.py). Decided once
    # here so the same flag drives both the INSTRUCTIONS block baked
    # into the prompt and the max_new_tokens cap handed to Gemma below --
    # keeping them in sync is what lets the concise cap actually shorten
    # generation instead of just truncating a prompt still asking for a
    # long answer.
    detailed = wants_detailed_answer(query_text)
    gen_max_new_tokens = DETAILED_MAX_NEW_TOKENS if detailed else CONCISE_MAX_NEW_TOKENS

    try:
        # build_prompt_with_context() applies token budgeting on top of
        # retrieval_caps.py's chunk-count cap (see prompt_engineering.py's
        # fit_context_to_budget()) and returns the candidates it actually
        # kept alongside the prompt, so `reranked` -- used below for both
        # sources and confidence -- always matches what was actually sent
        # to Gemma, even on the rare query where budgeting trims it further.
        # max_new_tokens is passed through so the token-budget check
        # reserves exactly the generation headroom that will actually be
        # used (see fit_context_to_budget()), and detailed keeps the
        # INSTRUCTIONS block in sync with gen_max_new_tokens above.
        prompt, reranked = build_prompt_with_context(
            query_text, reranked, max_new_tokens=gen_max_new_tokens, detailed=detailed,
        )
    except Exception as exc:
        logger.error(
            "Prompt construction failed for query: %r",
            query_text,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prompt construction failed: {exc}",
        ) from exc

    try:
        answer = generate_answer(prompt, max_new_tokens=gen_max_new_tokens)
    except Exception as exc:
        logger.error(
            "Gemma inference failed for query: %r",
            query_text,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Gemma inference failed: {exc}",
        ) from exc

    return AnswerResponse(
        answer=answer,
        sources=_build_sources(reranked, query_intent=routing_intent),
        # BUGFIX: was hardcoded None even though reranker_score is
        # available on every entry -- now reports the strongest score
        # actually backing this answer.
        confidence=(max((c.get("reranker_score") or 0.0) for c in reranked) if reranked else None),
        query_intent=routing_intent,
        routing_reason=routing_reason,
    )