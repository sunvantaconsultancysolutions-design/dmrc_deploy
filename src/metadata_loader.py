"""
metadata_loader.py

Loads the finalized Unified Metadata Schema (Chapter 6) for each clause
chunk. This module does NOT redesign the schema or the chunking strategy
-- it implements the schema exactly as specified in
DMRC_Unified_Metadata_Schema.md, mapping the parsed JSON structure
(pages[].clauses[]) onto the Project / Document / Structural / Retrieval
/ Version metadata groups.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


NON_CLAUSE_HEADINGS = {"MARGINALIA", "INDEX", "STAMP"}

# Document-level constants extracted once per source file (cover page).
# In production these are extracted by a preprocessing pass over each
# file's cover page text; they are hard-coded here per the finalized
# schema example values for reproducibility of this chapter's examples.
DOCUMENT_METADATA = {
    "DMRC_Chapter1_transcription.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH1",
        "document_name": "Scope of Work - ECS (Chapter 1)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 1",
    },
    "DMRC_Chapter2_transcription.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH2",
        "document_name": "Scope of Work - ECS (Chapter 2)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 2",
    },
    "chapter3.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH3",
        "document_name": "Scope of Work - ECS (Chapter 3)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 3",
    },
}

PROJECT_METADATA = {
    "project_name": "Phase-II Delhi MRTS Project - CS to Qutab Minar & CS to Badarpur Corridors",
    "contract_number": "BE-12 LOT3 & BE-14 LOT3",
    "employer_name": "Delhi Metro Rail Corporation Ltd. (DMRC)",
    "contractor_name": "M/s Blue Star Limited",
}


def normalize_clause_no(clause_no: str) -> str:
    """'6.7.2-1' -> '6.7.2.1' (delimiter normalization only)."""
    if not clause_no:
        return ""
    return re.sub(r"[-_]", ".", clause_no)


def parent_clause_of(clause_no_normalized: str) -> str:
    if not clause_no_normalized or "." not in clause_no_normalized:
        return ""
    return clause_no_normalized.rsplit(".", 1)[0]


def hierarchy_level_of(clause_no_normalized: str) -> int:
    if not clause_no_normalized:
        return 0
    return len(clause_no_normalized.split("."))


def classify_chunk_type(heading: str, clause_no: str, continues_previous: bool) -> str:
    """Classify a raw clauses[] entry into its semantic chunk_type.

    heading is matched with startswith("COVER PAGE") rather than an exact
    match, because the source JSON uses several cover-page variants
    ("COVER PAGE", "COVER PAGE - SCOPE OF WORK / SPECIFICATIONS",
    "COVER PAGE - VOLUME 3(I)", etc.) -- an exact-match set only ever
    caught the bare "COVER PAGE" heading and let the other four variants
    fall through and be classified (and therefore embedded) as "clause".
    """
    if heading == "STAMP":
        return "stamp"
    if heading == "INDEX":
        return "index"
    if heading == "MARGINALIA":
        return "marginalia"
    if heading.startswith("COVER PAGE"):
        return "cover_page"
    if continues_previous:
        return "continuation"
    return "clause"


@dataclass
class ChunkRecord:
    text: str
    metadata: dict = field(default_factory=dict)


def _recompute_hash(record: ChunkRecord) -> None:
    """Keep chunk_hash in sync with record.text whenever text changes
    (e.g. after a continuation fragment is merged in). A stale hash
    computed only at creation time would silently stop reflecting the
    chunk's actual (post-merge) content, defeating its purpose as a
    change-detection/deduplication key.
    """
    record.metadata["chunk_hash"] = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:12]


def build_chunk_records(parsed_json: dict, source_file: str, carry_over_record: Optional[ChunkRecord] = None):
    """Maps pages[].clauses[] onto ChunkRecord objects carrying the
    finalized metadata schema. Chunking granularity (one clause = one
    chunk) is unchanged from Chapter 6/7 -- what changed is that this
    function now actually implements the merge/exclusion rules the
    Chapter 7 Chunking Strategy document specifies, instead of emitting
    every clauses[] entry as its own independent chunk regardless of type.

    Parameters
    ----------
    parsed_json : dict
        The parsed pages[].clauses[] structure for one source file.
    source_file : str
        Filename, used to look up per-document metadata.
    carry_over_record : ChunkRecord, optional
        The last real-clause chunk built by the PREVIOUS file in
        processing order. If this file's first clause entry is itself a
        continuation (continues_previous=True), it is merged into this
        carried-over record instead of becoming an orphaned chunk --
        this is the cross-file continuation observed at the
        Chapter1 -> Chapter2 boundary in this corpus (Chapter 7 SS7.5).

    Returns
    -------
    (records, open_record) : tuple[list[ChunkRecord], Optional[ChunkRecord]]
        records is the list of NEW chunks built from this file (the
        carry_over_record, if merged into, is not included -- it was
        already appended to the previous file's records list by the
        caller). open_record is the last real-clause chunk left open at
        the end of this file, to be threaded into the next file's call
        as its carry_over_record.
    """
    doc_meta = DOCUMENT_METADATA.get(source_file, {})
    records = []
    document_sequence = 0
    chunk_number = 0
    pending_stamps = []

    # open_record is the "current clause" that a leading continuation
    # fragment can merge into. It starts as whatever real clause was
    # still open at the end of the previous file (may be None for the
    # first file processed, or when running a single file in isolation).
    # Only real chunk_type=="clause" chunks are ever assigned to it --
    # a continuation can only continue an actual clause.
    open_record = carry_over_record

    # last_record is the most recently built chunk of ANY type (clause,
    # cover_page, index, marginalia) and is what a trailing STAMP attaches
    # to. This must be tracked separately from open_record: a stamp on a
    # cover page belongs to that cover page, not to whatever real clause
    # happened to precede it several pages earlier. Using open_record for
    # stamps as well would let a cover page's stamps queue up in
    # pending_stamps and cascade forward onto the next real clause chunk
    # instead of staying scoped to the page they actually appear on.
    last_record = carry_over_record

    for page in parsed_json.get("pages", []):
        pdf_page = page.get("pdf_page")
        printed_page = page.get("printed_page", "")
        section_heading = page.get("section_heading", "")

        for clause in page.get("clauses", []):
            clause_no = clause.get("clause_no", "")
            heading = clause.get("heading", "")
            text = clause.get("text", "")
            continues_previous = bool(clause.get("continues_previous", False))
            low_confidence = bool(clause.get("low_confidence", False))

            chunk_type = classify_chunk_type(heading, clause_no, continues_previous)

            # --- STAMP: fold into the nearest preceding real clause -------
            # Stamps are annotations ABOUT the page they sit on, never
            # independent content, so they are never their own chunk. They
            # attach to `open_record` (the clause chunk already built,
            # possibly carried over from the previous file) rather than to
            # whatever chunk gets built next -- in the source JSON a STAMP
            # entry always trails the clause it was scanned on.
            if chunk_type == "stamp":
                # --- Change 2: detect organization from stamp text ---------
                # Simple substring match against the raw stamp text. Order
                # doesn't matter here since "DELHI" and "BLUE STAR" are
                # mutually exclusive in practice; falls back to None
                # (unchanged prior behavior) if neither is present.
                organization = None
                if "DELHI" in text:
                    organization = "DMRC"
                elif "BLUE STAR" in text:
                    organization = "Blue Star Ltd."

                # --- Change 1: preserve complete stamp information ---------
                # Added "text" field carrying the actual stamp text (was
                # previously discarded); existing fields are untouched.
                stamp_entry = {
                    "organization": organization,
                    "type": heading,
                    "text": text,
                    "page_control": printed_page,
                }
                if last_record is not None:
                    last_record.metadata["stamps"].append(stamp_entry)
                else:
                    pending_stamps.append(stamp_entry)
                continue

            # --- CONTINUATION: merge into the clause it continues ---------
            # A continues_previous=True fragment is the tail of a clause
            # that was split across a page (or file) boundary. Per Chapter
            # 7 SS7.4/SS7.5 it must be reunited with that clause's chunk,
            # not stored as its own decontextualized chunk with no
            # clause_no/heading/parent_clause of its own.
            if chunk_type == "continuation":
                if open_record is not None:
                    open_record.text = (open_record.text.rstrip() + "\n\n" + text.strip()).strip()
                    if low_confidence:
                        open_record.metadata["low_confidence"] = True
                    _recompute_hash(open_record)
                else:
                    # No open clause exists at all (continuation with
                    # nothing to attach to, e.g. a single file run in
                    # isolation with no carry_over). Fall back to storing
                    # it as its own clause chunk rather than dropping data.
                    document_sequence += 1
                    chunk_number += 1
                    chunk_id = "-".join(filter(None, [
                        doc_meta.get("document_id", "DOC"),
                        f"p{pdf_page}",
                        f"seq{document_sequence}",
                        f"{document_sequence:03d}",
                    ]))
                    metadata = {
                        **PROJECT_METADATA,
                        **doc_meta,
                        "source_file": source_file,
                        "section_heading": section_heading,
                        "clause_no": "",
                        "clause_no_normalized": "",
                        "parent_clause": "",
                        "hierarchy_level": 0,
                        "heading": "",
                        "chunk_type": "clause",
                        "pdf_page": pdf_page,
                        "printed_page": printed_page,
                        "continues_previous": True,
                        "low_confidence": low_confidence,
                        "chunk_id": chunk_id,
                        "chunk_number": chunk_number,
                        "document_sequence": document_sequence,
                        "chunk_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                        "language": "en",
                        "approval_status": "approved",
                        "stamps": pending_stamps,
                    }
                    pending_stamps = []
                    open_record = ChunkRecord(text=text, metadata=metadata)
                    records.append(open_record)
                    last_record = open_record
                continue

            # --- Empty-text header stubs: not a standalone chunk -----------
            # e.g. clause_no="6.9", heading="TRAINING", text="" -- a section
            # header with no body of its own, immediately followed by its
            # numbered sub-clauses. Per Chapter 7 SS7.4 these are excluded
            # from embedding. Their clause_no/heading don't need to be
            # separately propagated: parent_clause/hierarchy_level for the
            # sub-clauses that follow are already derived purely from each
            # sub-clause's OWN clause_no_normalized, independent of whether
            # this stub has a chunk. `open_record` is deliberately left
            # unchanged so a trailing STAMP/continuation still attaches to
            # the last chunk with real content, not to this empty stub.
            if chunk_type == "clause" and not text.strip():
                continue

            # --- Real chunk: cover_page / index / marginalia / clause -----
            document_sequence += 1
            if chunk_type == "clause":
                chunk_number += 1

            clause_no_normalized = normalize_clause_no(clause_no)

            chunk_id = "-".join(filter(None, [
                doc_meta.get("document_id", "DOC"),
                f"p{pdf_page}",
                f"c{clause_no_normalized}" if clause_no_normalized else f"seq{document_sequence}",
                f"{document_sequence:03d}",
            ]))

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

            metadata = {
                **PROJECT_METADATA,
                **doc_meta,
                "source_file": source_file,
                "section_heading": section_heading,
                "clause_no": clause_no,
                "clause_no_normalized": clause_no_normalized,
                "parent_clause": parent_clause_of(clause_no_normalized),
                "hierarchy_level": hierarchy_level_of(clause_no_normalized),
                "heading": heading,
                "chunk_type": chunk_type,
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "continues_previous": continues_previous,
                "low_confidence": low_confidence,
                "chunk_id": chunk_id,
                "chunk_number": chunk_number,
                "document_sequence": document_sequence,
                "chunk_hash": content_hash,
                "language": "en",
                "approval_status": "approved",
                "stamps": pending_stamps,
            }
            pending_stamps = []

            record = ChunkRecord(text=text, metadata=metadata)
            records.append(record)

            # Any chunk just built is a valid target for a STAMP that
            # trails it (a stamp on a cover page belongs to that cover
            # page, not to some earlier real clause).
            last_record = record

            # Only real, substantive clause chunks stay "open" as a merge
            # target for a later CONTINUATION -- cover_page/index/
            # marginalia chunks are self-contained page furniture, not a
            # clause that a later fragment could be continuing.
            if chunk_type == "clause":
                open_record = record

    return records, open_record


# ============================================================================
# BOQ ingestion (independent pipeline)
# ============================================================================
#
# BOQ transcription JSON has a different shape from the clause corpus:
#   { "document_metadata": {...dynamic...}, "pages": [ { "items": [...] } ] }
# instead of pages[].clauses[]. It also carries its own per-file metadata
# (contract/schedule/discipline/station_columns/...) rather than looking it
# up from a hard-coded DOCUMENT_METADATA table keyed by filename, because a
# BOQ file already states that metadata inline (see document_metadata in
# boq_part1/2/3.json). Nothing above this point is modified.

# item_type values that are pure section/panel/subsection LABELS -- they
# have no body text, unit, or quantity of their own (mirrors the empty-text
# clause header stubs excluded in build_chunk_records). Their description
# is folded into the metadata of every item beneath them instead of
# becoming its own chunk, exactly like section_heading is threaded through
# the clause pipeline.
BOQ_HEADER_TYPES = {"section_header", "boq_item_header", "sub_section_header"}

# item_type values that carry real, independently-retrievable content and
# become their own "boq" chunk.
#   - boq_item / sub_item: the actual scope-of-supply line items (the
#     reason this pipeline exists).
#   - general_note: attaches specification prose to a section (e.g. panel
#     construction requirements) -- semantically equivalent to clause body
#     text, so it is embedded too even though the task brief named only
#     boq_item/sub_item explicitly.
BOQ_EMBEDDABLE_TYPES = {"boq_item", "sub_item", "general_note"}

# Everything else (table, total_row, words_total, letterhead,
# signature_block, letter_*, title, header, section_title, narrative,
# footer_stamp_text, date) is tender-total / discount-letter / part-summary
# boilerplate -- financial roll-up content with no independent technical
# meaning, excluded for the same reason cover_page/index/marginalia/stamp
# are excluded from the clause pipeline.


def is_boq_json(parsed_json: dict) -> bool:
    """True if parsed_json is a BOQ transcription rather than a clause
    file. BOQ pages carry an "items" array (each tagged item_type); clause
    pages carry "clauses" instead. Falls back to checking for a top-level
    document_metadata block if pages is empty/missing, since that key is
    only ever present on BOQ files.
    """
    pages = parsed_json.get("pages", [])
    if pages:
        return "items" in pages[0]
    return "document_metadata" in parsed_json


def _slugify(value: str) -> str:
    """Uppercase, alphanumeric-only slug for building chunk_ids."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").upper()


def _boq_document_id(doc_meta: dict) -> str:
    """BOQ files have no document_id field (unlike the clause corpus's
    filename -> document_id lookup table) -- contract + schedule together
    are what uniquely identify a BOQ schedule across its part files
    (boq_part1/2/3.json all share the same contract+schedule), so they are
    combined here instead. This is the one deliberate chunk_id-generation
    difference from the clause pipeline; noted per the task's "explain if
    you change chunk_id generation" instruction.
    """
    contract = _slugify(doc_meta.get("contract", ""))
    schedule = _slugify(doc_meta.get("schedule", ""))
    return "-".join(filter(None, ["BOQ", contract, schedule])) or "BOQ-UNKNOWN"


def build_boq_chunk_records(parsed_json: dict, source_file: str):
    """Maps pages[].items[] from a BOQ transcription onto ChunkRecord
    objects. Independent of build_chunk_records() -- does not call it,
    does not share its per-clause hierarchy fields, and does not modify
    it. Emits the same ChunkRecord(text, metadata) contract (chunk_id,
    chunk_number, document_sequence, chunk_hash, language,
    approval_status, stamps all present) so nothing downstream
    (batch_embed / storage / query / reranker / hybrid_retriever) needs to
    change to accept "boq" chunks alongside "clause" chunks.

    Sub-items are emitted as their own chunk each (never rolled into their
    parent boq_item) per the task's explicit instruction -- a sub_item's
    description is a distinct line of scope of supply and collapsing it
    into the parent would bury it from retrieval.

    Parameters
    ----------
    parsed_json : dict
        The parsed {document_metadata, pages[].items[]} structure for one
        BOQ source file.
    source_file : str
        Filename, carried into each chunk's metadata (same field name used
        by the clause pipeline).

    Returns
    -------
    records : list[ChunkRecord]
    """
    doc_meta = parsed_json.get("document_metadata", {})
    # Flatten out the one nested dict (transcription_convention is
    # transcriber-facing documentation, not content) so every remaining
    # doc_meta field is a metadata-safe scalar/list, then spread it onto
    # every chunk -- this is what makes document_metadata "read
    # dynamically": whatever fields a given BOQ file happens to carry
    # (contract, schedule, discipline, station_columns, project_title,
    # lot, ...) flow through untouched, instead of being named one by one.
    flat_doc_meta = {k: v for k, v in doc_meta.items() if not isinstance(v, dict)}
    document_id = _boq_document_id(doc_meta)

    records = []
    document_sequence = 0
    chunk_number = 0

    for page in parsed_json.get("pages", []):
        page_label = page.get("page_label", "")
        # source_file here is the per-page scan filename (e.g.
        # "page_007.png"), unique across the whole document even though
        # boq_part1/2/3.json share one contract+schedule and each file's
        # own document_sequence restarts at 1 -- using it (instead of a
        # page index that would collide across the three part files) is
        # what keeps chunk_id globally unique.
        page_ref = (page.get("source_file") or "").replace(".png", "").replace(".jpg", "")
        _pdf_page_match = re.search(r"page_(\d+)", page.get("source_file") or "")
        pdf_page = int(_pdf_page_match.group(1)) if _pdf_page_match else None
        page_stamps = page.get("stamps_seals", []) or []

        # Running context set by the most recent header row(s) seen on
        # this page, folded into every item chunk beneath them -- an item
        # description like "Panel complete as per specifications..." is
        # meaningless in isolation without knowing which panel/section it
        # belongs to, exactly as section_heading gives a clause context.
        section_no, section_desc = "", ""
        item_header_no, item_header_desc, panel_reference = "", "", ""
        subsection_no, subsection_desc = "", ""

        for item in page.get("items", []):
            item_type = item.get("item_type", "")
            s_no = item.get("s_no") or ""
            parent = item.get("parent", "") or ""
            description = (item.get("description") or "").strip()

            if item_type == "section_header":
                section_no, section_desc = s_no, description
                item_header_no, item_header_desc, panel_reference = "", "", ""
                subsection_no, subsection_desc = "", ""
                continue

            if item_type == "boq_item_header":
                item_header_no, item_header_desc = s_no, description
                panel_reference = item.get("panel_reference", "") or ""
                subsection_no, subsection_desc = "", ""
                continue

            if item_type == "sub_section_header":
                subsection_no, subsection_desc = s_no, description
                continue

            if item_type not in BOQ_EMBEDDABLE_TYPES or not description:
                # table / total_row / letter_* / etc boilerplate, or a
                # header-typed row with no description -- never its own
                # chunk, same treatment as clause header stubs.
                continue

            document_sequence += 1
            chunk_number += 1

            ref = s_no or parent or f"seq{document_sequence}"
            chunk_id = "-".join(filter(None, [
                document_id,
                page_ref or f"pg{document_sequence}",
                f"i{_slugify(str(ref))}",
                f"{document_sequence:03d}",
            ]))

            metadata = {
                **flat_doc_meta,
                "source_file": source_file,
                "page_label": page_label,
                "pdf_page": pdf_page,
                "chunk_type": "boq",
                "item_type": item_type,
                "s_no": s_no,
                "parent": parent,
                "section_no": section_no,
                "section_description": section_desc,
                "item_header_no": item_header_no,
                "item_header_description": item_header_desc,
                "panel_reference": panel_reference,
                "subsection_no": subsection_no,
                "subsection_description": subsection_desc,
                "unit": item.get("unit") or "",
                # Numeric fields stored as metadata (not embedded into
                # text -- see build_boq_embedding_input) per the task's
                # NUMERIC FIELDS instruction. quantities is kept as the
                # nested per-station dict rather than flattened into N
                # separate metadata keys, matching the existing
                # precedent of storing structured data (stamps) directly
                # in clause metadata -- storage.py already needs to
                # handle non-scalar metadata values for that field.
                "quantities": item.get("quantities") or {},
                "rate_in_inr": item.get("rate_in_inr"),
                "rate_in_foreign_currency": item.get("rate_in_foreign_currency"),
                "amount_in_inr": item.get("amount_in_inr"),
                "amount_in_foreign_currency": item.get("amount_in_foreign_currency"),
                "has_amendment": bool(item.get("has_amendment", False)),
                "amendments": item.get("amendments") or [],
                "chunk_id": chunk_id,
                "chunk_number": chunk_number,
                "document_sequence": document_sequence,
                "chunk_hash": hashlib.sha256(description.encode("utf-8")).hexdigest()[:12],
                "language": "en",
                "approval_status": "approved",
                "stamps": page_stamps,
            }

            records.append(ChunkRecord(text=description, metadata=metadata))

    return records