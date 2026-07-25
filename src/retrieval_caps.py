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
"""
import os

import src.hybrid_retriever as hybrid_retriever
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


hybrid_retriever.hybrid_search = _capped_hybrid
reranker.rerank = _capped_rerank