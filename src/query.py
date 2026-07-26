"""
query.py

Semantic similarity search over the existing DMRC ChromaDB collection
using the already-trained BAAI/bge-m3 embedding model. Read-only:
this script only queries the vector store populated by the Chapter 7
embedding pipeline, it never writes to it. Covers Chapter 8.10
(Similarity Search) and 8.11 (Metadata Filtering).
"""

import argparse
import re

from sentence_transformers import SentenceTransformer

from .storage import get_collection

MODEL_NAME = "BAAI/bge-m3"

EXAMPLE_QUERIES = [
    "scope of work",
    "testing",
    "contractor obligations",
    "maintenance",
    "commissioning",
    "payment",
]

_model = None


def get_model():
    """Lazily load and cache the BGE-M3 embedding model (loaded once per run)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(query: str):
    """Convert a user query into a normalized dense embedding vector."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def search(query: str, top_k: int = 5, metadata_filter: dict = None):
    """
    Perform semantic similarity search over the ChromaDB collection.

    Parameters
    ----------
    query : str
        The natural language question to search for.
    top_k : int
        Number of top results to return (default 5).
    metadata_filter : dict, optional
        ChromaDB ``where`` filter, e.g. {"clause_no": "6.8.2"} or
        {"chapter": "Chapter 3"}. If None, search runs over the whole
        collection with no metadata restriction.

    Returns
    -------
    list[dict]
        One entry per result: chunk_id, document text, metadata, and
        a similarity_score plus the raw distance.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    results = collection.query(**query_kwargs)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    formatted = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        # storage.py stores normalized embeddings; ChromaDB's default
        # "l2" space returns squared L2 distance, so for unit vectors
        # cosine_similarity = 1 - (squared_l2_distance / 2).
        similarity_score = 1 - (distance / 2)
        formatted.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": round(distance, 4),
                "similarity_score": round(similarity_score, 4),
            }
        )

    return formatted


# ---------------------------------------------------------------------------
# NEW -- Exact Clause-Number Fast Path
#
# Problem: dense retrieval (semantic similarity on chunk TEXT) and BM25
# (lexical term matching on chunk TEXT) both rank candidates by how well
# the query's *wording* matches the chunk's wording -- neither one does an
# exact lookup against the `clause_no` metadata field. So a query like
# "Explain Clause 1.2.1" can fail to rank the 1.2.1 chunk first, even
# though some chunk's metadata says clause_no == "1.2.1" verbatim.
#
# Fix: before hybrid retrieval runs at all, check whether the query names
# an explicit clause number. If it does, do a metadata-only ChromaDB
# lookup (collection.get(where=...), NOT collection.query()) -- this
# needs no embedding, no BM25, and always finds the clause if it exists.
# This does not touch embeddings, bm25_index.py, hybrid_retriever.py, or
# metadata_loader.py. If no exact match is found, the caller (app.py)
# falls back to the existing hybrid_search() pipeline unchanged.
# ---------------------------------------------------------------------------

# Matches clause numbers like "1.2.1", "6.8.2", "6.7.2-1", "6.7.2.1":
# one or more digits followed by 1-4 more separator+digits groups, where
# the separator can be "." or "-" since users type both forms naturally
# (the metadata loader's normalized form uses "." but the as-authored
# clause label may use "-"). Requires at least one separator so a bare
# number ("Section 5", "page 3") is never mistaken for a clause number.
CLAUSE_NO_PATTERN = re.compile(r"\b(\d+(?:[.-]\d+){1,4})\b")


def extract_clause_no(query_text: str):
    """Return the first clause-number-shaped token found in `query_text`,
    exactly as typed (e.g. "6.7.2-1" out of "Explain Clause 6.7.2-1", or
    "6.7.2.1" out of "Explain Clause 6.7.2.1"), or None if the query
    doesn't contain one. Accepts both "." and "-" separators. Pure string
    matching -- no model call, no DB call, no normalization here (that
    happens in get_chunk_by_clause_no, only if the as-typed form doesn't
    match anything).
    """
    match = CLAUSE_NO_PATTERN.search(query_text)
    return match.group(1) if match else None


def get_chunk_by_clause_no(clause_no: str):
    """Exact metadata lookup for a clause number, bypassing dense/BM25
    retrieval entirely.

    Users may type a clause number with either separator (e.g.
    "6.7.2-1" or "6.7.2.1"), and the metadata loader may store the
    as-authored form under `clause_no` (e.g. "6.7.2-1") while keeping a
    dot-normalized form under `clause_no_normalized` (e.g. "6.7.2.1").
    To support both input styles without touching metadata_loader.py or
    re-ingesting any data, this tries two lookups in order:

        1. where={"clause_no": clause_no}                   (as typed)
        2. where={"clause_no_normalized": normalized_clause} ("-" -> ".")

    Only falls through to the second lookup if the first finds nothing,
    so a corpus that only ever populates `clause_no` (never
    `clause_no_normalized`) still works unchanged.

    Uses collection.get(where=...) rather than collection.query(...):
    .get() is ChromaDB's metadata-only accessor (no query embedding, no
    ANN search), which is exactly what an exact clause match needs.

    Returns a list of candidate dicts shaped to match
    hybrid_retriever.merge_candidates()'s output (chunk_id, document,
    metadata, score, retrieval_source, dense_score, bm25_score) so it can
    be fed straight into the existing rerank() / build_prompt() /
    _build_sources() code with no changes to any of them. Returns []
    if no chunk carries this clause number under either field, so the
    caller falls back to hybrid_search() exactly as before.
    """
    collection = get_collection()

    result = collection.get(
        where={"clause_no": clause_no},
        include=["documents", "metadatas"],
    )

    if not result.get("ids"):
        normalized_clause_no = clause_no.replace("-", ".")
        result = collection.get(
            where={"clause_no_normalized": normalized_clause_no},
            include=["documents", "metadatas"],
        )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    candidates = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "score": 1.0,  # exact metadata match -- highest possible confidence
                "retrieval_source": "exact_clause_match",
                "dense_score": None,
                "bm25_score": None,
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# TASK 4 -- Exact BOQ-Item-Number Fast Path
#
# Problem: exactly the same problem get_chunk_by_clause_no() above solves
# for clauses, but for Bill-of-Quantities rows. Dense retrieval (semantic
# similarity on chunk TEXT) and BM25 (lexical term matching on chunk TEXT)
# both rank candidates by how well the query's *wording* matches the
# chunk's wording -- neither one does an exact lookup against the BOQ
# identifier metadata fields (`parent`, `s_no`, `item_header_no`,
# `section_no`). So a query like "Describe BOQ item 1.02.E.2" can fail to
# rank the correct item first, even though some chunk's metadata carries
# that identifier verbatim.
#
# Fix: before hybrid retrieval runs (and after the clause fast path above
# has already had its turn -- see app.py), check whether the query names
# a BOQ-item-shaped identifier. If it does, do a metadata-only ChromaDB
# lookup (collection.get(where=...), NOT collection.query()) against the
# BOQ identifier fields, in priority order. This needs no embedding, no
# BM25, and always finds the item if it exists. Mirrors
# get_chunk_by_clause_no()'s design exactly, one metadata field at a time.
# Does not touch embeddings, bm25_index.py, hybrid_retriever.py, or
# metadata_loader.py. If no exact match is found, the caller (app.py)
# falls back to the existing hybrid_search() pipeline unchanged.
# ---------------------------------------------------------------------------

# Matches BOQ item identifiers like "1.02.E.2", "1.02.E", "4.2", "1.01":
# one or more digits, followed by 1-4 more separator+segment groups, where
# each segment may be digits (an item/sub-item number) OR a single letter
# (a sub-section marker, e.g. the "E" in "1.02.E.2"). This differs from
# CLAUSE_NO_PATTERN above only in allowing a single-letter segment, since
# BOQ identifiers -- unlike clause numbers -- can carry a lettered
# sub-section reference in the middle of the dotted path.
BOQ_ITEM_PATTERN = re.compile(r"\b(\d+(?:\.[A-Za-z0-9]+){1,4})\b")

# BOQ metadata fields that carry a chunk's own identifier, checked in
# this exact order (per the task's specification). `parent` is checked
# first because a fully-qualified identifier such as "1.02.E.2" is the
# field every sub-item (a), b), c), ...) filed under that reference
# actually carries -- matching on it returns the whole item family, the
# BOQ-pipeline equivalent of get_chunks_by_parent_clause()'s family
# semantics for clauses. `s_no` / `item_header_no` / `section_no` cover
# shallower identifiers (e.g. "4.2", "1.01") that a query may name
# directly instead of naming a specific sub-item's parent.
BOQ_IDENTIFIER_FIELDS = ("parent", "s_no", "item_header_no", "section_no")


def extract_boq_item_no(query_text: str):
    """Return the first BOQ-item-shaped token found in `query_text`,
    exactly as typed (e.g. "1.02.E.2" out of "Describe BOQ item
    1.02.E.2"), or None if the query doesn't contain one. Pure string
    matching -- no model call, no DB call, no normalization here (BOQ
    identifiers are stored and matched verbatim, unlike clause numbers,
    which have a separate normalized form).
    """
    match = BOQ_ITEM_PATTERN.search(query_text)
    return match.group(1) if match else None


def get_chunk_by_boq_item_no(item_no: str):
    """Exact metadata lookup for a BOQ item identifier, bypassing
    dense/BM25 retrieval entirely. This is the BOQ-pipeline equivalent
    of get_chunk_by_clause_no() above, applying the same
    collection.get(where=...) approach to the BOQ identifier fields
    (`parent`, `s_no`, `item_header_no`, `section_no`) instead of
    `clause_no` / `clause_no_normalized`.

    Tries each field in BOQ_IDENTIFIER_FIELDS, in order, stopping at the
    first field that yields at least one match -- exactly the same
    "try, then fall through" pattern get_chunk_by_clause_no() uses for
    its two clause_no lookups, just extended to four candidate fields
    instead of two. A match on `parent` naturally returns every sub-item
    chunk sharing that parent (e.g. all of "1.02.E.2"'s a)-o) sub-items),
    since `where` matches every chunk whose field equals `item_no`.

    Uses collection.get(where=...) rather than collection.query(...):
    .get() is ChromaDB's metadata-only accessor (no query embedding, no
    ANN search), which is exactly what an exact BOQ item match needs.

    Returns a list of candidate dicts shaped to match
    get_chunk_by_clause_no()'s output exactly (chunk_id, document,
    metadata, score, retrieval_source, dense_score, bm25_score) so it
    can be fed straight into the existing rerank() / build_prompt() /
    _build_sources() code with no changes to any of them. Returns []
    if no chunk carries this identifier under any of the candidate
    fields, so the caller falls back to hybrid_search() exactly as
    before.
    """
    collection = get_collection()

    result = {"ids": [], "documents": [], "metadatas": []}
    for field in BOQ_IDENTIFIER_FIELDS:
        result = collection.get(
            where={field: item_no},
            include=["documents", "metadatas"],
        )
        if result.get("ids"):
            break

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    candidates = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "score": 1.0,  # exact metadata match -- highest possible confidence
                "retrieval_source": "exact_boq_item_match",
                "dense_score": None,
                "bm25_score": None,
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# TASK 3 -- Sibling-clause lookup by parent_clause metadata.
#
# Same pattern as get_chunk_by_clause_no() above: a pure metadata
# accessor (collection.get(where=...), no embedding, no ANN search), so
# it costs nothing beyond a ChromaDB metadata filter. Used by
# reranker.py::expand_with_siblings() to fetch the full sibling set of
# a clause family (e.g. every 6.8.x chunk given parent_clause="6.8")
# once the reranker's own output shows the user is asking about that
# family broadly. This was previously stored in every chunk's metadata
# (confirmed: all 6.8.x chunks carry parent_clause="6.8" in this
# collection) but never read by any retrieval code -- this function is
# the first caller.
# ---------------------------------------------------------------------------

def get_chunks_by_parent_clause(parent_clause: str):
    """Metadata-only lookup of every chunk sharing the given
    parent_clause (e.g. all children of "6.8"). Returns a list of
    candidate dicts shaped like get_chunk_by_clause_no()'s output
    (chunk_id, document, metadata) -- callers add their own
    score/retrieval_source fields as appropriate for their use case.

    Returns [] if parent_clause is falsy or nothing matches, so callers
    can use this defensively without a separate existence check.
    """
    if not parent_clause:
        return []

    collection = get_collection()
    result = collection.get(
        where={"parent_clause": parent_clause},
        include=["documents", "metadatas"],
    )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    return [
        {"chunk_id": chunk_id, "document": document, "metadata": metadata}
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    ]


def print_results(query: str, results: list):
    """Pretty-print search results to the console."""
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] similarity={result['similarity_score']} "
            f"(distance={result['distance']})  chunk_id={result['chunk_id']}"
        )
        print(
            f"    clause_no={metadata.get('clause_no', 'N/A')}  "
            f"heading={metadata.get('heading', 'N/A')}  "
            f"pdf_page={metadata.get('pdf_page', 'N/A')}  "
            f"document_name={metadata.get('document_name', 'N/A')}"
        )
        text_preview = result["document"][:300]
        print(f"    text: {text_preview}{'...' if len(result['document']) > 300 else ''}")
    print()


def build_filter(filter_arg: str):
    """Parse a single ``key=value`` CLI filter into a ChromaDB ``where`` dict."""
    if not filter_arg:
        return None
    key, sep, value = filter_arg.partition("=")
    if not sep or not key:
        raise ValueError(
            "Filter must be in the form key=value, e.g. --filter clause_no=6.8.2"
        )
    if value.isdigit():
        value = int(value)
    return {key: value}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic similarity search over the DMRC ChromaDB collection."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural language query. If omitted, runs the built-in example queries.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help=(
            "Optional metadata filter as key=value, e.g. "
            "--filter clause_no=6.8.2, --filter chapter='Chapter 3', "
            "--filter approval_status=approved, --filter pdf_page=5"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_filter = build_filter(args.filter)

    if args.query:
        results = search(args.query, top_k=args.top_k, metadata_filter=metadata_filter)
        print_results(args.query, results)
    else:
        print("No query supplied - running example queries:\n")
        for example_query in EXAMPLE_QUERIES:
            results = search(example_query, top_k=args.top_k, metadata_filter=metadata_filter)
            print_results(example_query, results)


if __name__ == "__main__":
    main()
