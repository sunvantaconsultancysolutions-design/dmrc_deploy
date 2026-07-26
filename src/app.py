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
from pydantic import BaseModel

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
    has_usable_context,
)
from .query import get_model as get_dense_model
from .query import extract_clause_no, get_chunk_by_clause_no, get_chunks_by_parent_clause
from .query import extract_boq_item_no, get_chunk_by_boq_item_no
from .reranker import evaluate_confidence, expand_with_siblings, get_reranker_model, rerank
from . import query as query_module
from . import reranker as reranker_module
from .gemma_inference import generate_answer, get_gemma_model
from . import gemma_inference as gemma_module

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
    """

    query: str


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
    page: Optional[int] = None
    document: Optional[str] = None
    item_number: Optional[str] = None
    retrieval_source: Optional[str] = None
    reranker_score: Optional[float] = None
    chunk_id: Optional[str] = None
    # TASK 3 -- UI retrieval label fix. Exposes the existing
    # metadata["chunk_type"] ("clause" or "boq", set unconditionally by
    # metadata_loader.py for every chunk) so the frontend can render
    # "Grounded in N Retrieved Clauses" / "... BOQ Items" / "...
    # Documents" (mixed) from real retrieval metadata instead of the
    # hardcoded "clauses" wording it used before. Optional/back-compat:
    # any existing client that ignores this new field keeps working
    # unchanged.
    chunk_type: Optional[str] = None


class AnswerResponse(BaseModel):
    """12.8 Response Model. `confidence` is left as None for now since
    no LLM inference (and therefore no confidence signal) exists yet
    in this chapter's scope.
    """

    answer: str
    sources: List[SourceItem]
    confidence: Optional[float] = None


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

def _build_sources(reranked_candidates: List[Dict[str, Any]]) -> List[SourceItem]:
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
        page = get_boq_page_number(metadata) if is_boq else metadata.get("pdf_page")

        sources.append(
            SourceItem(
                clause=metadata.get("clause_no"),
                page=page,
                document=get_document_name(metadata),
                item_number=item_number,
                retrieval_source=candidate.get("retrieval_source"),
                reranker_score=candidate.get("reranker_score"),
                chunk_id=candidate.get("chunk_id") or metadata.get("chunk_id"),
                chunk_type=metadata.get("chunk_type"),
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
        4. Hybrid retrieval over ChromaDB + BM25.       -> hybrid_search()
        5. Rerank the candidates.                       -> rerank()
        6. Build the final LLM prompt.                  -> build_prompt()
        7. Generate the answer.                         -> generate_answer()
        8. Return the answer + sources as JSON.
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

    if not exact_match:
        try:
            candidates = hybrid_search(
                query_text,
                top_k_dense=DENSE_TOP_K,
                top_k_bm25=BM25_TOP_K,
                final_top_k=MERGED_CANDIDATE_POOL,
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
        # ------------------------------------------------------------
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
        return AnswerResponse(answer=NO_CONTEXT_ANSWER, sources=[], confidence=None)

    try:
        # build_prompt_with_context() applies token budgeting on top of
        # retrieval_caps.py's chunk-count cap (see prompt_engineering.py's
        # fit_context_to_budget()) and returns the candidates it actually
        # kept alongside the prompt, so `reranked` -- used below for both
        # sources and confidence -- always matches what was actually sent
        # to Gemma, even on the rare query where budgeting trims it further.
        prompt, reranked = build_prompt_with_context(query_text, reranked)
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
        answer = generate_answer(prompt)
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
        sources=_build_sources(reranked),
        # BUGFIX: was hardcoded None even though reranker_score is
        # available on every entry -- now reports the strongest score
        # actually backing this answer.
        confidence=(max((c.get("reranker_score") or 0.0) for c in reranked) if reranked else None),
    )