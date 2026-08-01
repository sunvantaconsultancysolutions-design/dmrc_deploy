"""
prompt_engineering.py

Chapter 11 -- Prompt Engineering.

Pure prompt CONSTRUCTION only: no model loading, no inference. This
module takes the merged candidate list already produced by Chapter 9's
hybrid_retriever.hybrid_search() and turns it into the exact prompt
text that gets handed to Gemma 2 9B in gemma_inference.py's
generate_answer(). Kept separate from model loading (same split as
batch_embed.py/embed_single.py) so the prompt logic can be built and
unit-tested without a GPU or the model checkpoint in memory.

DOC FIX: this module originally targeted Gemma 3; gemma_inference.py
documents a deliberate later swap to google/gemma-2-9b-it (a different,
text-only architecture -- see that module's own "MODEL CHANGE" note).
This docstring is updated to match; no prompt-construction logic here
changed as a result of that swap.

------------------------------------------------------------------------
11.4 Prompt Components
------------------------------------------------------------------------
The assembled prompt has five parts, each implemented below:
    1. System Prompt        -> SYSTEM_PROMPT
    2. Retrieved Context     -> format_context()
    3. User Query            -> inserted directly into build_prompt()
    4. Response Instructions -> RESPONSE_INSTRUCTIONS
    5. Output Format         -> ANSWER_SECTION_HEADER (model completes from here)
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dmrc_rag.prompt_engineering")

# Mirrors the RAG_DEBUG pattern already used by app.py, hybrid_retriever.py,
# and reranker.py: off by default (production-safe), each stage reads the
# same env var independently and logs itself right where its own output is
# computed.
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# 11.5 System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Engineering Contract Assistant for the DMRC ECS/TVS Contract (BE-12 LOT3, BE-14 LOT3, CE-10 LOT4, CE-11 LOT4).

SYNONYM AWARENESS — Contract documents use formal legal/engineering language that rarely matches a user's exact wording. This applies to EVERY term, not just the examples below:
• "penalty" = liquidated damages, Rs. per day imposed, Penalty Clause
• "warranty" = defects liability period (DLP), guarantee period
• "testing" = SAT, system acceptance test, integrated testing, functional tests, commissioning
• "compensation" = liquidated damages, delay damages
• "retention" = retention money, payment schedule
• "safety" = health and safety, safety requirements
• "scope" = scope of work, scope of supply, contractor obligations
• "who trains X" = "training shall be provided for X", "the Contractor shall train X"
• "rating" of equipment = the equipment's stated amps/kA/volts/kW specifications, even if the word "rating" itself is absent
These are EXAMPLES of a general pattern, not an exhaustive list. Apply the same reasoning to ANY term: if the context describes the same real-world thing the user is asking about, in different words, that IS the answer.

Answer ONLY using the supplied context. Never invent information not present in the context.
A context block does not need to use the user's exact words to be the answer -- judge by real-world meaning, not literal string overlap.
If none of the retrieved context blocks describe the thing the user is asking about, respond with exactly:
"I could not find the requested information in the provided documents."

Always cite: Clause Number, Page Number, BOQ Item Number (when available)."""


# ---------------------------------------------------------------------------
# 11.8 Response Instructions
# ---------------------------------------------------------------------------

RESPONSE_INSTRUCTIONS = """Use only the supplied context. Do not assume missing information.
Quote exact values whenever possible (quantities, penalties, durations, thresholds).
Cite the Clause Number, Page Number, and BOQ Item Number (if available) for every claim.
Clearly say so if the requested information is unavailable in the context.
Generate a concise, engineering-style response.
Review every retrieved context block below, not just the first one.
If more than one block is relevant to the question, combine all relevant clauses into a single, comprehensive answer -- do not stop after the first relevant clause.
Never use information that is not present in the supplied context, and never hallucinate details not stated there."""


# ---------------------------------------------------------------------------
# Response length modes (demo-day change): most questions should get a
# short, scannable answer instead of a multi-paragraph essay -- both for
# generation latency (fewer tokens to generate) and for readability in the
# chat UI. A caller can still ask for the long-form version explicitly (see
# wants_detailed_answer() below), which switches back to the original
# RESPONSE_INSTRUCTIONS above unchanged.
#
# This only changes the INSTRUCTIONS section of the prompt -- retrieval,
# reranking, confidence gating, and context formatting are untouched, so
# the model still sees exactly the same evidence either way; only how much
# of it to write back out changes.
# ---------------------------------------------------------------------------

CONCISE_RESPONSE_INSTRUCTIONS = """Use only the supplied context. Never assume missing information.

Step 1: Read EVERY ranked context block before deciding if the answer exists.
Step 2: Meaning check, not word-matching — if a context block describes the same real-world thing the user asked about (an equipment spec that answers a "rating" question, a clause that describes who does the training, a BOQ line whose category covers what was asked), that IS the answer, even if the exact question words never appear in the text. Never refuse just because of a wording or terminology difference.
Step 3: If ANY context block answers the question by meaning, you MUST answer from it directly. Do not refuse a question that a retrieved block already answers just because the match feels partial, indirect, or the confidence score is not high — a present, on-topic context block always outranks refusing.
Step 4: Only say "I could not find the requested information in the provided documents." if, after reading every block, none of them describe the thing being asked about at all.

Format:
• 3–7 bullet points, 150–250 words
• Quote exact values: amounts (Rs.), durations (months/days), thresholds, quantities, rates
• Cite Clause Number, Page Number, and/or BOQ Item Number inline for every bullet
• Combine multiple relevant blocks into one comprehensive answer

Never invent data not present in the supplied context."""

DETAIL_REQUEST_KEYWORDS = (
    "detail", "detailed", "elaborate", "elaborated", "in depth", "in-depth",
    "comprehensive", "full explanation", "explain fully", "explain in full",
    "everything about", "complete breakdown", "long answer", "at length",
    "step by step", "step-by-step", "thorough",
)


def wants_detailed_answer(query: str) -> bool:
    """True if the user's own wording is explicitly asking for a longer,
    fuller answer than the default concise mode -- e.g. "explain in
    detail", "give me a thorough breakdown". Used by app.py to pick
    between CONCISE_RESPONSE_INSTRUCTIONS (default) and the original
    RESPONSE_INSTRUCTIONS (opt-in, unabridged), per the "if the user
    explicitly asks for detailed information, provide a detailed answer"
    requirement. Pure keyword check on the query text -- no extra model
    call, so it costs nothing on the latency path this exists to shorten.
    """
    lowered = query.lower()
    return any(keyword in lowered for keyword in DETAIL_REQUEST_KEYWORDS)


ANSWER_SECTION_HEADER = "ANSWER"


# ---------------------------------------------------------------------------
# 11.11 Few-Shot Prompting
# ---------------------------------------------------------------------------
# Optional, off by default in build_prompt() unless use_few_shot=True is
# passed explicitly by the caller.
# Kept to a single example -- per 11.16 "context compression" / "maximum
# token limitation", every extra example is retrieved-context budget the
# actual candidates don't get, so this is intentionally minimal.

FEW_SHOT_EXAMPLE = """Question
What is BOQ Item 4.2?

Answer
BOQ Item 4.2 specifies the supply of Cooling Towers with a quantity of 6 Nos.

Source
BOQ Item 4.2"""


# ---------------------------------------------------------------------------
# 11.6 Retrieved Context / 11.10 step 2-4 (dedupe, merge, insert metadata)
# ---------------------------------------------------------------------------

def get_boq_item_number(metadata: Dict[str, Any]) -> Optional[str]:
    """Returns a BOQ row's own identifying item number.

    IMPORTANT: this is read from the "s_no" (Serial Number) metadata
    field -- the column BOQ tables are keyed by -- NOT from
    "item_number". "item_number" is a different field that only
    appears on CLAUSE metadata, where it is a cross-reference to a
    related BOQ item (see format_context()'s clause branch below); it
    is not present on the BOQ row's own metadata.

    Exposed as a shared accessor (rather than inlined only in
    _format_boq_block()) so app.py's API layer reads the exact same
    field this module's prompt formatting does, instead of maintaining
    a second, independent (and previously incorrect) copy of this
    lookup.
    """
    return metadata.get("s_no")


def get_boq_page_number(metadata: Dict[str, Any]) -> Optional[Any]:
    """Returns a BOQ row's page reference.

    Prefers "pdf_page" (the same field clause metadata uses) and falls
    back to "page_number" for BOQ rows that only carry the latter.
    Shared with app.py for the same reason as get_boq_item_number()
    above -- one accessor, used everywhere BOQ page info is needed.
    """
    page_number = metadata.get("pdf_page")
    if page_number in (None, ""):
        page_number = metadata.get("page_number")
    return page_number


def get_scanned_page(metadata: Dict[str, Any]) -> Optional[Any]:
    """Page number stamped on the scanned image.

    Prefers the document-control stamp captured during transcription:
    "printed_page" on clause chunks, "stamp_number" on BOQ chunks
    (added by scripts/patch_boq_stamps.py). Falls back to the PDF page
    index when the scan carries no stamp (cover pages) or the stamp
    was not captured. Leading zeros are stripped for display
    ('000009' -> '9').

    Shared with app.py for the same reason as get_boq_page_number():
    one accessor, used everywhere the citation page is needed.
    """
    stamp = metadata.get("printed_page") or metadata.get("stamp_number")
    if stamp not in (None, ""):
        s = str(stamp).lstrip("0")
        return s or "0"
    if metadata.get("chunk_type") == "boq":
        return get_boq_page_number(metadata)
    return metadata.get("pdf_page")


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate chunk_ids, keeping the first (highest-ranked)
    occurrence. hybrid_retriever.merge_candidates() already dedupes
    across the dense/sparse merge, but this module is written to also
    work standalone against a plain dense- or sparse-only result list
    (e.g. query.py's search() output), so it re-applies the same rule
    defensively rather than assuming the caller already did it.
    """
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(candidate)
    return deduped


def get_document_name(metadata: Dict[str, Any]) -> Optional[str]:
    """Turns a raw source-file reference into a readable document title.

    If metadata already carries a descriptive title (e.g. from a
    document manifest), that is used as-is. Otherwise the raw filename
    (e.g. "chapter3.json") is cleaned up into something readable
    (e.g. "Chapter 3") purely for display -- the underlying metadata
    and retrieval logic are untouched.

    QA FIX (Issue 3): BOQ chunk metadata never carries "document_title"
    or "document_name" -- metadata_loader.py stores the source PDF
    filename under "source_pdf" instead for BOQ records. Falling back
    to that field here (rather than duplicating it into "document_name"
    at ingestion time) means every reader of this metadata sees a
    document name for BOQ chunks too. Exposed as a public, shared
    accessor (like get_boq_item_number/get_boq_page_number above) so
    app.py's API layer resolves this identically to this module's
    prompt formatting, instead of maintaining a second, independent
    copy of this lookup.
    """
    document_title = metadata.get("document_title")
    if document_title:
        return document_title

    document_name = metadata.get("document_name") or metadata.get("source_pdf")
    if not document_name:
        return None

    stem = document_name.rsplit(".", 1)[0] if "." in document_name else document_name
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", stem)
    stem = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", stem)
    stem = stem.strip()
    return stem.title() if stem else document_name


def _format_boq_block(index: int, candidate: Dict[str, Any]) -> str:
    """Renders a single retrieved BOQ chunk as a "Rank N" block, using
    BOQ-specific metadata (item number, section, parent item, item
    type, schedule, contract, page) instead of the clause fields
    format_context() uses. Kept as a separate function so clause
    formatting in format_context() is untouched.

    Item number and page are read via get_boq_item_number() /
    get_boq_page_number() rather than inline dict lookups, so app.py's
    API layer resolves these BOQ fields identically to this prompt
    formatting (see those helpers' docstrings for the s_no vs.
    item_number distinction).
    """
    metadata = candidate["metadata"]

    lines = [f"Rank {index}"]

    s_no = get_boq_item_number(metadata)
    lines.append(f"BOQ Item {s_no}" if s_no else "BOQ Item (number unavailable)")

    detail_fields = [
        ("Section", metadata.get("section")),
        ("Parent", metadata.get("parent")),
        ("Item Type", metadata.get("item_type")),
        ("Schedule", metadata.get("schedule")),
        ("Contract", metadata.get("contract")),
    ]
    for label, value in detail_fields:
        if value:
            lines.append(f"{label}: {value}")

    chunk_text = candidate["document"].strip()
    chunk_text = re.sub(r"\n\s*\n+", "\n\n", chunk_text)
    lines.append("")
    lines.append(chunk_text)

    footer_parts = []
    # Rule 1: cite the number stamped on the scanned page, not the PDF index.
    page_number = get_scanned_page(metadata)
    if page_number not in (None, ""):
        footer_parts.append(f"Page {page_number}")
    document_name = get_document_name(metadata)
    if document_name:
        footer_parts.append(f"Source Document: {document_name}")
    if footer_parts:
        lines.append("")
        lines.append(" | ".join(footer_parts))

    return "\n".join(lines)


def format_context(candidates: List[Dict[str, Any]]) -> str:
    """Renders the retrieved candidates as numbered "Rank N" blocks
    (11.6) -- these are reranked results (BGE Reranker), so "Rank"
    reflects that ordering more accurately than a generic "Document"
    label. Each block inserts the structural metadata (11.10 step 4:
    clause number, heading, page, BOQ item, document name) alongside
    the chunk's text, in the order: Clause Number, Heading, Page
    Number, BOQ Item Number, Source Document.

    BOQ Item Number is only shown when metadata actually carries an
    item_number field, so this stays conditional rather than printing
    an empty "BOQ Item: N/A" on every single result.

    Candidates whose metadata["chunk_type"] == "boq" are instead routed
    to _format_boq_block(), which formats them using BOQ-specific
    fields (item number, section, parent item, item type, schedule,
    contract, page) rather than the clause fields below. Clause
    formatting itself is unchanged.
    """
    if not candidates:
        return ""

    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]

        if metadata.get("chunk_type") == "boq":
            blocks.append(_format_boq_block(i, candidate))
            continue

        header_parts = []
        clause_no = metadata.get("clause_no")
        if clause_no:
            header_parts.append(f"Clause {clause_no}")
        heading = metadata.get("heading")
        if heading:
            header_parts.append(heading)
        header = " | ".join(header_parts) if header_parts else "Clause (number unavailable)"

        chunk_text = candidate["document"].strip()
        chunk_text = re.sub(r"\n\s*\n+", "\n\n", chunk_text)

        lines = [f"Rank {i}", header, "", chunk_text]

        detail_parts = []
        # Rule 1: cite the number stamped on the scanned page, not the PDF index.
        scanned_page = get_scanned_page(metadata)
        if scanned_page not in (None, ""):
            detail_parts.append(f"Page {scanned_page}")
        item_number = metadata.get("item_number")
        if item_number:
            detail_parts.append(f"BOQ Item {item_number}")
        document_name = get_document_name(metadata)
        if document_name:
            detail_parts.append(f"Source Document: {document_name}")
        if detail_parts:
            lines.append("")
            lines.append(" | ".join(detail_parts))

        blocks.append("\n".join(lines))

    return "\n\n--------------------------------\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 11.9 Prompt Template / 11.10 Prompt Construction Workflow
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 48

NO_CANDIDATES_CONTEXT_PLACEHOLDER = "(No relevant context was retrieved for this query.)"


def _section(header: str, body: str) -> str:
    """Wraps a body of text in a "==== HEADER ====" block, used for
    every section of the assembled prompt so the template reads the
    same way debuggers/log viewers see it in 11.9.
    """
    return f"{_SEPARATOR}\n{header}\n{_SEPARATOR}\n{body}"


def _assemble_prompt(
    query: str,
    candidates: List[Dict[str, Any]],
    use_few_shot: bool = False,
    detailed: bool = False,
) -> str:
    """The actual Chapter 11.9/11.10 prompt-assembly logic: query +
    candidates in, final prompt string out. No budget checking here --
    this always renders every candidate it is given. Kept private; call
    build_prompt() or build_prompt_with_context() instead, both of which
    apply token budgeting (see fit_context_to_budget() below) before this
    ever runs. This stays a pure, unbounded renderer so
    fit_context_to_budget() can call it repeatedly (once per
    candidate-list length it tries) without any budget-vs-render
    recursion.

    detailed selects which INSTRUCTIONS block is used: False (default)
    -> CONCISE_RESPONSE_INSTRUCTIONS (5-8 bullets, ~150-250 words), True
    -> the original, unabridged RESPONSE_INSTRUCTIONS. See
    wants_detailed_answer() for how callers decide which to pass.
    """
    deduped = deduplicate_candidates(candidates)
    context_block = format_context(deduped) or NO_CANDIDATES_CONTEXT_PLACEHOLDER

    sections = [_section("SYSTEM", SYSTEM_PROMPT)]

    if use_few_shot:
        sections.append(_section("EXAMPLE", FEW_SHOT_EXAMPLE))

    sections.append(_section("CONTEXT", context_block))
    sections.append(_section("QUESTION", query.strip()))
    instructions = RESPONSE_INSTRUCTIONS if detailed else CONCISE_RESPONSE_INSTRUCTIONS
    sections.append(_section("INSTRUCTIONS", instructions))
    sections.append(f"{_SEPARATOR}\n{ANSWER_SECTION_HEADER}\n{_SEPARATOR}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Token budgeting
#
# retrieval_caps.py bounds how many CHUNKS reach here; it says nothing
# about how many TOKENS those chunks cost once assembled into a prompt.
# A handful of long BOQ rows or clause blocks can still overflow Gemma 2
# 9B's context window even at a capped chunk count. This bounds the
# assembled prompt (context + system/instructions/question overhead) so
# that, together with the tokens reserved for generation, the total never
# exceeds gemma_inference.MODEL_CONTEXT_WINDOW_TOKENS.
#
# Reuses the exact tokenizer generate_answer() uses for the real call
# (gemma_inference.get_tokenizer()) -- no second tokenizer is loaded, and
# get_tokenizer() only loads the tokenizer itself, not the 9B model, so
# this stays cheap even if the model hasn't been warmed up yet.
# ---------------------------------------------------------------------------

# apply_chat_template() wraps the raw prompt string with a handful of
# control tokens at generation time (<bos>, <start_of_turn>user, the
# trailing <start_of_turn>model, etc.) that a plain tokenizer.encode() of
# the prompt text alone does not include. This fixed headroom absorbs that
# difference so the budget check is a real upper bound, not an estimate
# that generate_answer() could still exceed by a few tokens.
CHAT_TEMPLATE_TOKEN_OVERHEAD = 16


def _count_tokens(text: str) -> int:
    """Tokenizes `text` with the same tokenizer generate_answer() uses, so
    this count matches what actually reaches the model.
    add_special_tokens=False because this measures the raw prompt text's
    token cost, not a chat-template-wrapped sequence -- see
    CHAT_TEMPLATE_TOKEN_OVERHEAD above for that wrapping's cost instead.
    """
    from .gemma_inference import get_tokenizer  # local import: avoids importing
                                                  # transformers/torch at module
                                                  # load for callers that only need
                                                  # pure prompt construction.
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def fit_context_to_budget(
    query: str,
    candidates: List[Dict[str, Any]],
    max_new_tokens: Optional[int] = None,
    use_few_shot: bool = False,
    detailed: bool = False,
) -> List[Dict[str, Any]]:
    """Trims `candidates` -- already reranker-ordered, highest-ranked
    first -- so the prompt _assemble_prompt() would build from them, plus
    the tokens reserved for generation, fits inside
    gemma_inference.MODEL_CONTEXT_WINDOW_TOKENS.

    Chunks are dropped ONE AT A TIME FROM THE END of the list (i.e. the
    lowest-ranked survivor goes first) until the assembled prompt fits the
    budget. This never reorders or drops a higher-ranked chunk ahead of a
    lower-ranked one that survives -- it only ever shortens the list from
    the tail, so reranker order is preserved exactly among whatever
    remains.

    Small prompts (the common case) exit on the first iteration -- one
    token count, no trimming -- so this adds one tokenizer call and no
    behavior change for any prompt that already fit.
    """
    if max_new_tokens is None:
        from .gemma_inference import MAX_NEW_TOKENS as max_new_tokens
    from .gemma_inference import MODEL_CONTEXT_WINDOW_TOKENS

    reserved = max_new_tokens + CHAT_TEMPLATE_TOKEN_OVERHEAD
    budget = MODEL_CONTEXT_WINDOW_TOKENS - reserved

    kept = deduplicate_candidates(candidates)
    original_count = len(kept)

    # Check-then-loop (not `while kept:`) so an already-empty candidate
    # list (e.g. a genuine empty-retrieval query) still gets evaluated
    # once, rather than skipping straight past the fit check to the
    # "still overflows with zero chunks" branch below.
    while True:
        prompt = _assemble_prompt(query, kept, use_few_shot=use_few_shot, detailed=detailed)
        n_tokens = _count_tokens(prompt)

        if n_tokens <= budget:
            if RAG_DEBUG:
                logger.info(
                    "[token-budget] query=%r kept=%d/%d chunks, prompt=%d tok "
                    "(budget=%d, reserved_for_generation=%d, window=%d)",
                    query, len(kept), original_count, n_tokens, budget,
                    reserved, MODEL_CONTEXT_WINDOW_TOKENS,
                )
            return kept

        if not kept:
            # Even the system prompt + instructions + question alone (zero
            # retrieved context) exceeds the budget. Extremely unlikely --
            # that's a few hundred tokens against an 8192-token window --
            # but handled rather than assumed away: _assemble_prompt()
            # already renders this exactly like a genuine empty-retrieval
            # result (via NO_CANDIDATES_CONTEXT_PLACEHOLDER); log it as a
            # real overflow rather than silently returning.
            logger.warning(
                "[token-budget] query=%r: prompt=%d tok exceeds budget=%d "
                "even with zero retrieved chunks; proceeding with no "
                "context anyway (nothing left to trim).",
                query, n_tokens, budget,
            )
            return kept

        if RAG_DEBUG:
            dropped = kept[-1]
            logger.info(
                "[token-budget] prompt=%d tok exceeds budget=%d -- dropping "
                "lowest-ranked surviving chunk chunk_id=%r reranker_score=%s",
                n_tokens, budget, dropped.get("chunk_id"), dropped.get("reranker_score"),
            )
        kept = kept[:-1]


def build_prompt_with_context(
    query: str,
    candidates: List[Dict[str, Any]],
    use_few_shot: bool = False,
    max_new_tokens: Optional[int] = None,
    detailed: Optional[bool] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Same as build_prompt() below, but also returns the (possibly
    token-budget-trimmed) candidate list that was actually rendered into
    the prompt, so a caller building citations/sources from "the
    candidates that grounded this answer" -- see app.py's /ask endpoint --
    can keep those in sync with what the model actually saw, even on the
    rare query where token budgeting trims the list app.py otherwise
    received from reranking/expansion.

    detailed selects the INSTRUCTIONS block (see _assemble_prompt). If
    left as None (the default, e.g. for any caller that hasn't been
    updated), it is derived from the query text itself via
    wants_detailed_answer() so old call sites keep working without
    change -- passing it explicitly (as app.py's /ask endpoint does) just
    skips re-deriving it from a query the caller may have already
    inspected.
    """
    if detailed is None:
        detailed = wants_detailed_answer(query)
    kept = fit_context_to_budget(
        query, candidates, max_new_tokens=max_new_tokens, use_few_shot=use_few_shot,
        detailed=detailed,
    )
    return _assemble_prompt(query, kept, use_few_shot=use_few_shot, detailed=detailed), kept


def build_prompt(query: str, candidates: List[Dict[str, Any]], use_few_shot: bool = False) -> str:
    """Assembles the full Gemma prompt from a user query and the merged
    retrieval candidates, following the Chapter 11.10 workflow:

        1. Receive top-ranked documents.      (candidates, as passed in)
        2. Remove duplicate chunks.            -> deduplicate_candidates()
        3. Fit the token budget.               -> fit_context_to_budget()
        4. Merge document context.             -> format_context()
        5. Insert metadata.                    -> done inside format_context()
        6. Insert user query.                  -> below
        7. Generate final prompt.              -> return value
        8. Send prompt to Gemma.               -> generate_answer.py

    use_few_shot defaults to False (11.11: few-shot prompting is
    optional) so the single example doesn't eat into the retrieved
    context's token budget unless explicitly requested.

    Returns the complete prompt string, ending right after the "ANSWER"
    header so the model's own generation is the completion of the
    template shown in 11.9. The prompt this returns is guaranteed to fit
    gemma_inference.MODEL_CONTEXT_WINDOW_TOKENS together with the tokens
    reserved for generation (see fit_context_to_budget()). Callers that
    also need to know which candidates survived that fit (e.g. to keep
    API source citations in sync) should call build_prompt_with_context()
    instead; this is a thin wrapper around it for callers that only need
    the prompt string, matching this function's original signature.
    """
    prompt, _kept = build_prompt_with_context(query, candidates, use_few_shot=use_few_shot)
    return prompt


# ---------------------------------------------------------------------------
# 11.14 Handling Missing Information
# ---------------------------------------------------------------------------

NO_CONTEXT_ANSWER = "I could not find the requested information in the provided documents."


def has_usable_context(candidates: List[Dict[str, Any]]) -> bool:
    """True if there is at least one retrieved candidate to answer from.
    generate_answer.py checks this BEFORE calling Gemma 3 at all -- if
    retrieval returned nothing, there is no reason to spend a GPU
    inference call producing text that the system prompt would force
    into the same canned refusal anyway.
    """
    return bool(candidates)