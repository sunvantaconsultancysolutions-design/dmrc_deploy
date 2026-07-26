"""
retrieval_caps.py

Runtime caps on retrieval breadth.

Originally a `%%writefile` cell in 02_Gemma_Inference_and_Serving.ipynb --
promoted here to a permanent, version-controlled module so production
deployments (Docker/RunPod) don't depend on a notebook having been run
first to generate this file.

Imported for its side effects: wraps hybrid_search() and rerank() in place
so every caller -- including app.py's /ask endpoint -- gets bounded output
without any change to app.py itself. app.py imports this module once at
startup (see the import added near the top of app.py).

QA FIX (Issue 2): also wraps reranker.expand_with_siblings() and
query.get_chunks_by_parent_clause(), the same way. Neither was
previously capped here, which meant two paths could put more than
MAX_CONTEXT chunks in front of the LLM despite this module's name:
  - expand_with_siblings() runs AFTER rerank() and can append up to
    4 more chunks on top of an already-capped list.
  - get_chunks_by_parent_clause() is called directly by app.py's
    exact-clause-number fast path (a query naming a parent clause),
    which never goes through hybrid_search()/rerank() at all.
Wrapping query.get_chunks_by_parent_clause() here also transparently
caps it inside expand_with_siblings() (which imports that same name
fresh on every call -- see reranker.py's own lazy `from .query import
get_chunks_by_parent_clause`), so one wrap covers both call sites.
"""
import os

import src.hybrid_retriever as hybrid_retriever
import src.query as query_module
import src.reranker as reranker

# AUDIT FOLLOW-UP:
#
# MAX_CANDIDATES: raised from 20 -> 30 (repository default) to match
# app.py's widened MERGED_CANDIDATE_POOL (Task 2). Cost check: the
# reranker batches in groups of DEFAULT_BATCH_SIZE=16 (reranker.py), so
# 20 candidates already took ceil(20/16)=2 batches and 30 candidates
# still take ceil(30/16)=2 batches -- this change adds essentially zero
# reranker latency.
#
# MAX_CONTEXT: left at the repository default of 15. This was NEVER the
# bug -- the forensic audit found the "grounded in 4 clauses" behavior
# was caused by the Colab deployment notebook setting RAG_MAX_CONTEXT=4
# at launch time (overriding this default), not by anything in this
# file. See the updated dmrc_deploy_colab_runner.ipynb, which now omits
# that override entirely so this repository default is what actually
# governs production. 15 comfortably exceeds RERANK_TOP_N=12 (app.py),
# so this cap no longer fires in normal operation at all -- it remains
# only as a defensive ceiling against a future misconfiguration.
MAX_CANDIDATES = int(os.environ.get("RAG_MAX_CANDIDATES", "30"))  # was 20 -- into reranker
MAX_CONTEXT = int(os.environ.get("RAG_MAX_CONTEXT", "15"))          # unchanged -- into the LLM

_orig_hybrid = hybrid_retriever.hybrid_search
_orig_rerank = reranker.rerank
_orig_expand_with_siblings = reranker.expand_with_siblings
_orig_get_chunks_by_parent_clause = query_module.get_chunks_by_parent_clause


def _capped_hybrid(*args, **kwargs):
    hits = _orig_hybrid(*args, **kwargs)
    if hits and len(hits) > MAX_CANDIDATES:
        print(f"[cap] hybrid_search {len(hits)} -> {MAX_CANDIDATES}", flush=True)
        hits = hits[:MAX_CANDIDATES]
    return hits


def _capped_rerank(*args, **kwargs):
    out = _orig_rerank(*args, **kwargs)
    if out and len(out) > MAX_CONTEXT:
        print(f"[cap] rerank {len(out)} -> {MAX_CONTEXT}", flush=True)
        out = out[:MAX_CONTEXT]
    return out


def _capped_expand_with_siblings(*args, **kwargs):
    # expand_with_siblings() already sorts its output by reranker_score
    # descending before returning, so truncating to MAX_CONTEXT here
    # keeps the highest-scoring chunks (original + siblings combined),
    # not an arbitrary cut.
    out = _orig_expand_with_siblings(*args, **kwargs)
    if out and len(out) > MAX_CONTEXT:
        print(f"[cap] expand_with_siblings {len(out)} -> {MAX_CONTEXT}", flush=True)
        out = out[:MAX_CONTEXT]
    return out


def _capped_get_chunks_by_parent_clause(*args, **kwargs):
    out = _orig_get_chunks_by_parent_clause(*args, **kwargs)
    if out and len(out) > MAX_CONTEXT:
        print(f"[cap] get_chunks_by_parent_clause {len(out)} -> {MAX_CONTEXT}", flush=True)
        out = out[:MAX_CONTEXT]
    return out


hybrid_retriever.hybrid_search = _capped_hybrid
reranker.rerank = _capped_rerank
reranker.expand_with_siblings = _capped_expand_with_siblings
query_module.get_chunks_by_parent_clause = _capped_get_chunks_by_parent_clause