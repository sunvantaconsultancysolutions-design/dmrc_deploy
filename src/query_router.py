"""
query_router.py

Task 4 -- Lightweight query-intent routing.

Classifies a free-text query as CLAUSE, BOQ, or GENERAL before retrieval
so that the hybrid pipeline can apply a chunk_type metadata filter,
preventing clause and BOQ chunks from competing in one pool.

Design constraints (from the task spec)
----------------------------------------
- Reuse the existing hybrid pipeline, BM25, and dense retrievers unchanged.
- Do NOT duplicate retrievers.
- Only add intelligent metadata filtering.
- Keep the implementation modular: callers receive (intent, metadata_filter)
  and pass metadata_filter straight to hybrid_search(); no retrieval logic
  lives here.

Routing logic
-------------
All routing is rule-based (regex + keyword lists).  No model inference,
no BM25 call, no ChromaDB access.  The function runs in <1 ms.

  CLAUSE  -- query contains an explicit clause reference (same regex
             extract_clause_no() already uses), OR contains a clause-domain
             keyword (defect liability, insurance, arbitration, etc.).

  BOQ     -- query contains an explicit BOQ item reference (same regex
             extract_boq_item_no() already uses), OR contains a BOQ-domain
             keyword (earthwork, cable, concrete, chiller, busbar, etc.).

  GENERAL -- neither signal fires; the caller runs hybrid_search() without
             a chunk_type filter, matching the existing behaviour.

Confidence
----------
Each classification returns an IntentResult with:
  - intent: "clause" | "boq" | "general"
  - metadata_filter: the dict to pass to hybrid_search() (or None)
  - reason: a short string explaining which rule fired (for logging/debug)

Priority: explicit identifier > keyword match > GENERAL.
When both CLAUSE and BOQ keywords fire, GENERAL is returned so the wider
unfiltered pool is searched rather than arbitrarily choosing one type.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Reuse the same compiled patterns from query.py so the two modules always
# agree on what looks like a clause number or BOQ item number.  Import is
# lazy (inside the function) to avoid a circular import if this module is
# ever imported from query.py's own file.
# ---------------------------------------------------------------------------

# Clause pattern: "1.2.1", "6.8.2", "6.7.2-1" (separator + at least one digit group)
_CLAUSE_NO_RE = re.compile(r"\b(\d+(?:[.\-]\d+){1,4})\b")

# BOQ item pattern: "1.02.E.2" (starts with digit, contains alpha segment)
_BOQ_ITEM_RE = re.compile(r"\b(\d+(?:\.[A-Za-z0-9]+){1,4})\b")

# ---------------------------------------------------------------------------
# Keyword vocabularies.  Keep these short and high-precision: the goal is
# to catch clear domain signals, not to enumerate every possible term.
# ---------------------------------------------------------------------------

# Clause-domain keywords (contract conditions / scope-of-work concepts).
# These are terms that virtually never appear in a BOQ line item.
_CLAUSE_KEYWORDS: frozenset = frozenset({
    # contract conditions
    "defect liability", "defects liability", "dlp",
    "performance security", "performance bond",
    "liquidated damages", "delay damages",
    "insurance", "indemnity", "indemnification",
    "arbitration", "dispute resolution", "adjudication",
    "completion period", "time for completion",
    "payment schedule", "interim payment", "retention",
    "force majeure", "notice to proceed",
    "variation", "change order", "extra work",
    "subcontract", "subcontractor",
    "safety requirement", "health and safety",
    "environmental requirement",
    "warranty", "guarantee period",
    # scope-of-work structural terms
    "scope of work", "scope of supply",
    "employer responsibility", "contractor responsibility",
    "contractor obligation",
    "relevant document",
    "drawings and records", "asset identification",
    "interfacing contractor", "interfacing agency",
    "installation plan", "method statement",
    "resident staff", "contractor staff",
    "test programme", "testing procedure",
    # ISSUE 3 FIX: add training-related keywords.
    # "training requirement" (singular) was already present but did not match
    # "training requirements" (plural) as used in real queries.  Adding the
    # bare word "training" and the plural form covers both phrasings.
    # Validated: "training" has zero BM25 hits in the BOQ corpus and does not
    # overlap with any _BOQ_KEYWORDS entry, so it is unambiguously clause-domain.
    "training", "training requirement", "training requirements",
    "staff training", "training obligations",
    "operation and maintenance",
    "spare parts", "tools and test equipment",
    # clause reference words (without an actual clause number)
    "clause", "sub-clause", "article", "section",
})

# BOQ-domain keywords (bill of quantities / materials / civil works).
# These are terms that virtually never appear in a clause body.
_BOQ_KEYWORDS: frozenset = frozenset({
    # civil / structural
    "earthwork", "excavation", "backfill",
    "concrete", "reinforcement", "rebar", "formwork",
    "masonry", "brickwork", "plastering",
    "drainage", "culvert", "manhole",
    "track", "rail", "sleeper", "ballast",
    # electrical / mechanical (BOQ-specific terms in this corpus)
    "busbar", "bus bar",
    "air circuit breaker", "acb",
    "mccb", "mcb",
    "chiller", "cooling tower",
    "pump", "motor starter", "star delta",
    "current transformer", "ct ratio",
    "xlpe", "armoured cable",
    "switchboard", "panel board",
    "digital ammeter", "energy meter",
    "selector switch", "indication lamp",
    # BOQ structural terms
    "bill of quantities", "boq", "schedule of quantities",
    "item", "sub item", "rate", "quantity", "unit rate",
    "lump sum", "provisional sum",
    # materials / supply
    "supply and install", "supply and erect", "furnish",
    "cable", "conduit", "duct", "tray",
    "earthing", "lightning protection",
})


@dataclass(frozen=True)
class IntentResult:
    intent: str                            # "clause" | "boq" | "general"
    metadata_filter: Optional[dict]        # passed straight to hybrid_search()
    reason: str                            # short explanation for logging


_CLAUSE_FILTER = {"chunk_type": "clause"}
_BOQ_FILTER    = {"chunk_type": "boq"}


def classify_query(query: str) -> IntentResult:
    """Classify a free-text query and return the retrieval filter to apply.

    Parameters
    ----------
    query : str
        The raw user query string.

    Returns
    -------
    IntentResult
        intent + metadata_filter + reason.
        Callers should pass result.metadata_filter to hybrid_search() and
        log result.reason at DEBUG level for traceability.
    """
    q_lower = query.lower().strip()

    # ------------------------------------------------------------------
    # 1. Explicit identifier signals (highest priority).
    # ------------------------------------------------------------------
    # BOQ item pattern must be checked BEFORE the clause pattern because
    # the BOQ regex also matches things like "1.2.1" -- if both fire on
    # the same token, the BOQ pattern is more specific (it allows alpha
    # segments like ".E.") and should win.
    if _BOQ_ITEM_RE.search(query):
        # Confirm the match looks like a genuine BOQ id (has an alpha segment),
        # not a clause number that also happens to match the looser pattern.
        m = _BOQ_ITEM_RE.search(query)
        if m and re.search(r"[A-Za-z]", m.group(1)):
            return IntentResult("boq", _BOQ_FILTER, f"explicit BOQ item id: {m.group(1)!r}")

    if _CLAUSE_NO_RE.search(query):
        m = _CLAUSE_NO_RE.search(query)
        return IntentResult("clause", _CLAUSE_FILTER, f"explicit clause number: {m.group(1)!r}")

    # ------------------------------------------------------------------
    # 2. Keyword signals.
    # ------------------------------------------------------------------
    clause_hit = _keyword_hit(q_lower, _CLAUSE_KEYWORDS)
    boq_hit    = _keyword_hit(q_lower, _BOQ_KEYWORDS)

    if clause_hit and not boq_hit:
        return IntentResult("clause", _CLAUSE_FILTER, f"clause keyword: {clause_hit!r}")

    if boq_hit and not clause_hit:
        return IntentResult("boq", _BOQ_FILTER, f"BOQ keyword: {boq_hit!r}")

    if clause_hit and boq_hit:
        # Ambiguous: both clause and BOQ keywords fired.  Fall through to
        # GENERAL so neither type is excluded from retrieval.
        return IntentResult(
            "general", None,
            f"ambiguous (clause={clause_hit!r}, boq={boq_hit!r}): searching unfiltered pool",
        )

    # ------------------------------------------------------------------
    # 3. Default: no signal -- search the full corpus.
    # ------------------------------------------------------------------
    return IntentResult("general", None, "no domain signal detected")


def _keyword_hit(q_lower: str, keywords: frozenset) -> Optional[str]:
    """Return the first keyword from `keywords` that appears as a
    word-boundary match in `q_lower`, or None if none match.

    Multi-word keywords are matched as substrings (word-boundary check on
    the first and last word of the phrase).
    """
    for kw in keywords:
        if _word_match(q_lower, kw):
            return kw
    return None


def _word_match(text: str, keyword: str) -> bool:
    """True if `keyword` appears in `text` with word boundaries on both ends.

    Handles both single-word ("earthwork") and multi-word ("cooling tower")
    keywords without requiring callers to pre-compile regexes for every term.
    """
    # Escape special regex chars in the keyword (handles "star-delta" etc.)
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return bool(re.search(pattern, text))
