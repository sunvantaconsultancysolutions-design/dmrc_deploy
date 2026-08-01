"""
text_stem.py

Phase 1 audit (round 2) -- shared lightweight word-form normalization.

Two independent problems traced back to the same root cause: neither
query_router.py's keyword matcher nor bm25_index.py's tokenizer normalize
word forms, so a keyword registered as "training" never matches a query
that says "trains"/"trained"/"who trains", and a BOQ chunk that says
"rated at 2500 A" never matches a query that says "busbar rating".

This is NOT a full linguistic stemmer (no Porter/Snowball algorithm, no
dependency). It is deliberately conservative:
  - Strips at most one common English suffix (-ing, -ed, -s/-es, -er/-ers).
  - Restores a dropped silent "e" for the -ing/-ed cases (rate -> rating,
    rate -> rated) since that pattern is extremely common in engineering
    vocabulary (rate, operate, terminate, indicate, ...).
  - Refuses to touch anything at or below `min_len` characters, which
    protects short domain acronyms (mccb, acb, dlp, ecs, tvs, bms, boq,
    amc) from ever being altered -- verified empirically against the
    full existing router keyword vocabulary with zero unintended
    collisions (see the Phase 1 round-2 validation notes).

Used by:
  - query_router.py::_word_match()  -- keyword matching fallback.
  - bm25_index.py::BM25Index        -- query-time term expansion (the
    indexed corpus tokens are never rewritten; only the *query* is
    expanded to also look up any corpus term sharing a stem with it,
    so this never changes what a query for the ORIGINAL exact term
    already matched).
"""

from typing import FrozenSet

_MIN_STEM_LEN = 4


def stem_candidates(word: str, min_len: int = _MIN_STEM_LEN) -> FrozenSet[str]:
    """Returns the set of plausible normalized forms of `word`.

    Always includes the original word. Short words (<= min_len chars)
    are returned unchanged -- this is what keeps acronyms like "mccb",
    "acb", "dlp", "ecs", "tvs", "bms", "boq", "amc" untouched.
    """
    word = word.lower()
    candidates = {word}
    if len(word) <= min_len:
        return frozenset(candidates)

    if word.endswith("ing"):
        stem = word[:-3]
        if len(stem) >= min_len:
            candidates.add(stem)
        if len(stem) >= min_len - 1:
            candidates.add(stem + "e")  # rating -> rate
    if word.endswith("ed"):
        stem = word[:-2]
        if len(stem) >= min_len:
            candidates.add(stem)
        if len(stem) >= min_len - 1:
            candidates.add(stem + "e")  # rated -> rate
    if word.endswith("es"):
        stem = word[:-2]
        if len(stem) >= min_len:
            candidates.add(stem)
    elif word.endswith("s") and not word.endswith("ss"):
        stem = word[:-1]
        if len(stem) >= min_len:
            candidates.add(stem)
    if word.endswith("ers"):
        stem = word[:-3]
        if len(stem) >= min_len:
            candidates.add(stem)
    elif word.endswith("er"):
        stem = word[:-2]
        if len(stem) >= min_len:
            candidates.add(stem)

    return frozenset(candidates)


def shares_stem(word_a: str, word_b: str) -> bool:
    """True if `word_a` and `word_b` share at least one normalized form
    (e.g. "training" and "trains" both reduce to "train"; "rating" and
    "rated" both reduce to "rate"). Exact-equal words always match.
    """
    if word_a == word_b:
        return True
    return bool(stem_candidates(word_a) & stem_candidates(word_b))
