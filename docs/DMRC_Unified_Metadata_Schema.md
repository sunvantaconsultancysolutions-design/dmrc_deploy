# Unified Metadata Schema — DMRC BE-12 LOT3 & BE-14 LOT3 Contract
### For BGE-M3 + ChromaDB RAG Pipeline

---

## 1. Source Structure Analysis

All three clause files share one JSON shape:

```
{
  "pages": [
    {
      "pdf_page": int,
      "printed_page": str,        // often "" in Chapter 1, populated "0000NN" elsewhere
      "section_heading": str,     // often "" on cover/index pages
      "clauses": [
        {
          "clause_no": str,       // "", "6.7.1", "6.7.2-1", "6.10.2" etc. — hierarchical, inconsistent delimiters
          "heading": str,         // "", "General", "STAMP", "COVER PAGE", "MARGINALIA", "INDEX"
          "text": str,
          "continues_previous": bool,
          "low_confidence": bool  // OPTIONAL — only present on some clauses
        }
      ]
    }
  ]
}
```

Key observations that drive the schema design:

- **No explicit project/document/volume fields exist in the JSON.** Project name, contract number, employer, contractor, and volume number only appear as free text inside `clauses[].text` on cover pages (e.g. "VOLUME 2", "VOLUME 3(I)"). These must be extracted once per document during preprocessing and then **propagated as constants** to every chunk — they are not per-page/per-clause fields.
- **`heading` is overloaded.** It's used both for real clause titles ("General", "Penalty Clause") and for non-substantive content markers ("STAMP", "COVER PAGE", "MARGINALIA", "INDEX"). This needs to be split into a semantic `chunk_type` field so retrieval can filter out stamps/marginalia noise.
- **`clause_no` is inconsistent** ("6.7.2-1" vs "6.7.2.1" vs "6.10.2") and frequently empty (continuation text, stamps, unheaded paragraphs).
- **`continues_previous` is a genuine cross-chunk linking signal** — critical for retrieval so a chunk's context isn't orphaned from its preceding clause.
- **`low_confidence` is sparse/optional** — an OCR/transcription confidence flag, useful to down-rank or flag content for human review, but must default to `false` when absent.
- **`printed_page` is unreliable** — blank in Chapter 1 cover/front matter, populated as a zero-padded control number elsewhere. Treat as optional/nullable, not a primary key.

### 1B. BOQ Source Structure (introduced during implementation)

A second, independent ingestion pipeline was added for Bill of Quantities transcriptions (`boq_part1/2/3.json`), which use a different JSON shape entirely — a top-level `document_metadata` block plus `pages[].items[]`, instead of `pages[].clauses[]`:

```
{
  "document_metadata": {
    "contract": str, "schedule": str, "discipline": str,
    "station_columns": [...], "project_title": str, "lot": str, ...
    // dynamic — whatever fields a given BOQ file's document_metadata carries
  },
  "pages": [
    {
      "page_label": str,
      "source_file": str,          // per-page scan filename, e.g. "page_007.png"
      "stamps_seals": [ {...} ],   // page-level stamps, distinct from clause STAMP entries
      "items": [
        {
          "item_type": str,        // "section_header" | "boq_item_header" | "sub_section_header" |
                                    // "boq_item" | "sub_item" | "general_note" | boilerplate types
          "s_no": str,
          "parent": str,
          "description": str,
          "unit": str,
          "quantities": {...},     // per-station quantity breakdown
          "rate_in_inr": number,
          "rate_in_foreign_currency": number,
          "amount_in_inr": number,
          "amount_in_foreign_currency": number,
          "has_amendment": bool,
          "amendments": [...]
        }
      ]
    }
  ]
}
```

`document_metadata` is read **dynamically** rather than through a fixed field list — every scalar key it contains (contract, schedule, discipline, project_title, lot, etc.) is propagated verbatim onto every chunk from that file. This is why BOQ chunks do **not** carry the clause pipeline's Document Metadata group (`document_name`, `document_type`, `volume`, `chapter` — see §2.B) — those come from a hard-coded per-filename lookup table that only exists for the three clause files. See §2.F for the full BOQ metadata mapping.

---

## 2. Unified Metadata Schema

### A. Project Metadata *(constant across every clause chunk — same value on every chunk from the 3 clause files)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| project_name | string | Extracted (from cover-page text, once) | `clauses[].text` (cover page) | Full MRTS project name | "Phase-II Delhi MRTS Project – CS to Qutab Minar & CS to Badarpur Corridors" | Mandatory |
| contract_number | string | Extracted | `clauses[].text` (cover page) | Contract package identifier | "BE-12 LOT3 & BE-14 LOT3" | Mandatory |
| employer_name | string | Extracted | `clauses[].text` (cover page) | Contracting authority | "Delhi Metro Rail Corporation Ltd. (DMRC)" | Mandatory |
| contractor_name | string | Extracted | `clauses[].text` (cover page) | Executing contractor | "M/s Blue Star Limited" | Mandatory |

> **Not implemented:** `system_scope` and `stations_covered` were specified at design time but are not produced by the final pipeline — `metadata_loader.py`'s `PROJECT_METADATA` constant carries only the four fields above; no chunk in the collection has these two keys. Removed from the schema rather than left as unpopulated "Optional" fields. Reinstate them (and the extraction logic to populate them) if this becomes a requirement later.

### B. Document Metadata *(constant per source file, differs across the 3 clause files — see §2.F for the BOQ equivalent)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| document_id | string | Generated | — | Stable unique ID for the source document, used as a Foreign key | "DMRC-BE12BE14-VOL2-CH1" | Mandatory |
| document_name | string | Extracted/Default | filename / cover text | Human-readable document name | "Scope of Work – ECS (Chapter 1)" | Mandatory |
| document_type | string (enum) | Extracted | `section_heading` / heading text | Category of content | "Scope of Work" \| "Specification" \| "Data Sheet" | Mandatory |
| volume | string | Extracted | `clauses[].text` (cover page, e.g. "VOLUME 2") | Contract volume number | "Volume 2" | Optional |
| chapter | string | Generated (from filename) | filename | Chapter label mapped from filename | "Chapter 1" | Mandatory |
| source_file | string | Generated | filename | Original filename ingested | "DMRC_Chapter1_transcription.json" | Mandatory |

> **Not implemented:** `total_pages_in_document` was specified at design time but is not computed anywhere in `metadata_loader.py` (no `len(pages)` value is ever added to a chunk's metadata). Removed from the schema for the same reason as `system_scope`/`stations_covered` above.

### C. Structural Metadata *(varies per page / clause — the positional fingerprint of a chunk)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| section_heading | string | Extracted | `pages[].section_heading` | Running header/section title of the page | "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS" | Optional (often "") |
| clause_no | string | Extracted | `clauses[].clause_no` | Raw clause/sub-clause number as transcribed | "6.7.2-1" | Optional (often "") |
| clause_no_normalized | string | Generated | derived from `clause_no` | Delimiter-normalized clause number for consistent filtering | "6.7.2.1" | Optional |
| parent_clause | string | Generated | derived from `clause_no_normalized` | The immediate parent clause one level up the hierarchy — enables "show me the whole 6.7.2 family" style navigation | "6.7.2" | Optional |
| hierarchy_level | integer | Generated | derived from `clause_no_normalized` (count of segments) | Depth of the clause in the numbering hierarchy: `6`→1, `6.7`→2, `6.7.2`→3, `6.7.2.1`→4; `0` if `clause_no` is empty | 4 | Optional |
| heading | string | Extracted | `clauses[].heading` | Clause title (only populated for real clauses — see `chunk_type`) | "Competency of Personnel" | Optional |
| chunk_type | string (enum) | Generated | derived from `heading`/`clause_no`/`continues_previous` | Semantic classification of the chunk's *content role*. `"stamp"` and `"continuation"` are transient classifications used only to route stamp-folding/continuation-merging during ingestion — a stamp entry is folded into its parent's `stamps` list (never its own chunk), and a continuation fragment is either merged into the clause it continues or, if nothing is open to merge into, stored with `chunk_type="clause"`. Neither value is ever persisted as a chunk's own `chunk_type`. | "clause" \| "marginalia" \| "cover_page" \| "index" \| "boq" | Mandatory |
| stamps | list[object] | Generated | extracted out of `clauses[]` where `heading == "STAMP"` | Structured stamp/seal metadata pulled off the page and attached to the nearest preceding real clause, instead of appearing as its own noisy chunk. `organization` is detected via a simple substring match on the stamp text ("DMRC" for a match on "DELHI", "Blue Star Ltd." for a match on "BLUE STAR"; `null` otherwise). Each object: `{organization, type, text, page_control}` — `text` carries the full raw stamp text (a "date" field is not separately parsed out). | `[{"organization":"DMRC","type":"STAMP","text":"...DELHI... Feb 2008...","page_control":"000017"},{"organization":"Blue Star Ltd.","type":"STAMP","text":"...BLUE STAR..."}]` | Optional (`[]` if none on page) |
| pdf_page | integer | Extracted | `pages[].pdf_page` | Physical page number in the source PDF | 1 | Mandatory |
| printed_page | string | Extracted | `pages[].printed_page` | Printed/control page number stamped on the page | "000017" | Optional (often "") |
| continues_previous | boolean | Extracted | `clauses[].continues_previous` | Whether this clause's text continues from the prior page/clause | false | Mandatory |
| low_confidence | boolean | Extracted/Default | `clauses[].low_confidence` | OCR/transcription confidence flag; defaults to `false` if key absent | true | Mandatory (defaulted) |

### D. Retrieval Metadata *(generated during preprocessing — one unique value per chunk; also used, unchanged, by the BOQ pipeline — see §2.F)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| chunk_id | string | Generated | — | Globally unique chunk identifier (deterministic, built from document_id + page + clause/item ref + sequence) | "DMRC-BE12BE14-VOL2-CH3-p1-c6.7.1-000" | Mandatory |
| chunk_number | integer | Generated | — | Only incremented for `chunk_type == "clause"` entries. Non-clause chunks (cover_page/index/marginalia) inherit whatever value chunk_number last held, so this is effectively "clause chunks emitted so far," not a distinct sequential index within every chunk type | 42 | Mandatory |
| document_sequence | integer | Generated | — | Absolute sequential position of the chunk across the *entire source document*, 1..N, regardless of chunk_type | 57 | Mandatory |
| chunk_hash | string | Generated | — | Content hash for deduplication/change-detection (first 12 hex chars of a SHA-256 digest of the chunk text); recomputed whenever a continuation is merged into an existing chunk | "a91f3e2b1c4d" | Optional |
| language | string | Default | — | Language of the chunk text (ISO 639-1) | "en" | Mandatory |

### E. Version Metadata *(governance/lifecycle — constant or slowly changing)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| approval_status | string (enum) | Default | — | Governance status of the ingested content | "draft" \| "approved" \| "superseded" | Mandatory (default "approved") |

> **Not implemented:** a per-chunk `revision` field was specified at design time (sourced from a date parsed out of STAMP text) but no such field is ever added to a chunk's metadata in the final pipeline — the raw stamp text is preserved verbatim in `stamps[].text` (§2.C) instead of being parsed into a structured revision date. `approval_status` alone is what's actually implemented and filterable today (e.g. `--filter approval_status=approved` in `query.py`'s CLI).

### F. BOQ-Specific Metadata *(chunk_type == "boq" only — introduced during implementation; see §1B for the source shape)*

BOQ chunks reuse the Retrieval Metadata (§2.D) and Version Metadata (§2.E) groups unchanged (`chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`, `language`, `approval_status` are all present with the same meaning). They do **not** carry the clause-pipeline's Document Metadata group (§2.B) or Project Metadata group (§2.A) — instead, every scalar field present in the source file's `document_metadata` block (§1B) is spread onto each chunk dynamically (e.g. `contract`, `schedule`, `discipline`, `project_title`, `lot`).

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| *(dynamic project/document fields)* | mixed | Extracted, spread verbatim | `document_metadata` (top level, scalar keys only) | Whatever per-file fields the BOQ transcription declares — no fixed field list | `contract: "BE-12 LOT3"`, `schedule: "Schedule B"` | Varies |
| source_file | string | Generated | `pages[].items[].source_file` (per page) | Per-page scan filename this item was transcribed from | "page_007.png" | Mandatory |
| page_label | string | Extracted | `pages[].page_label` | Page label from the BOQ transcription | "B-14" | Optional |
| item_type | string (enum) | Extracted | `items[].item_type` | BOQ row's structural role | "boq_item" \| "sub_item" \| "general_note" | Mandatory |
| s_no | string | Extracted | `items[].s_no` | The BOQ row's own identifying serial number — this is the field a BOQ item's "item number" is read from (not `item_number`, which is a different, clause-only cross-reference field) | "4.2" | Optional |
| parent | string | Extracted | `items[].parent` | Parent item reference for a sub-item | "4" | Optional |
| section_no / section_description | string | Extracted, carried forward | `items[].s_no` / `.description` where `item_type == "section_header"` | The most recent section header seen on the page, folded into every item beneath it | "4" / "Cooling Towers" | Optional |
| item_header_no / item_header_description | string | Extracted, carried forward | `items[].s_no` / `.description` where `item_type == "boq_item_header"` | The most recent BOQ item header seen on the page | "4.2" / "Supply of Cooling Towers" | Optional |
| panel_reference | string | Extracted, carried forward | `items[].panel_reference` where `item_type == "boq_item_header"` | Panel drawing reference tied to the current item header | "P-04" | Optional |
| subsection_no / subsection_description | string | Extracted, carried forward | `items[].s_no` / `.description` where `item_type == "sub_section_header"` | The most recent sub-section header seen on the page | "4.2.1" / "6 Nos." | Optional |
| unit | string | Extracted | `items[].unit` | Unit of measure | "Nos." | Optional |
| quantities | object | Extracted | `items[].quantities` | Per-station quantity breakdown, stored as a nested object (not flattened into separate keys) | `{"Hauz Khas": 2, "Saket": 1}` | Optional |
| rate_in_inr | number | Extracted | `items[].rate_in_inr` | Unit rate in INR | 125000.00 | Optional (omitted from stored metadata, not stored as `""`, when absent — see §5) |
| rate_in_foreign_currency | number | Extracted | `items[].rate_in_foreign_currency` | Unit rate in foreign currency, if applicable | null | Optional (omitted when absent) |
| amount_in_inr | number | Extracted | `items[].amount_in_inr` | Total amount in INR | 750000.00 | Optional (omitted when absent) |
| amount_in_foreign_currency | number | Extracted | `items[].amount_in_foreign_currency` | Total amount in foreign currency, if applicable | null | Optional (omitted when absent) |
| has_amendment | boolean | Extracted | `items[].has_amendment` | Whether this row has since been amended | false | Mandatory (default `false`) |
| amendments | list[object] | Extracted | `items[].amendments` | Amendment history entries, if any | `[]` | Optional (`[]` if none) |
| stamps | list[object] | Extracted | `pages[].stamps_seals` | Page-level stamps/seals for a BOQ page — a differently-sourced but same-shaped reuse of the `stamps` field name from §2.C (source is `stamps_seals`, not clause STAMP entries) | `[]` | Optional (`[]` if none) |

> **Storage-layer note (see `storage.py`):** the four numeric fields above (`rate_in_inr`, `rate_in_foreign_currency`, `amount_in_inr`, `amount_in_foreign_currency`) are the only metadata fields with dedicated `None`-handling: a missing value is **omitted from the stored record entirely**, rather than coerced to `""` the way other absent fields are, so the field never mixes numeric and string types across the collection (which would otherwise break ChromaDB's numeric `where` filters).

> **Not embedded:** identifiers and pipeline bookkeeping fields (`chunk_id`, `chunk_hash`, `chunk_number`, `document_sequence`, `source_file`, `page_label`, `language`, `approval_status`, `stamps`) and all four numeric fields above are deliberately excluded from the text sent to the embedding model — see §5.

> **Not implemented for BOQ:** sub-items are always emitted as their own chunk (never rolled into their parent `boq_item`); section/item-header/sub-section header rows are never chunked on their own (they exist only to populate the "carried forward" fields above on the items beneath them), the same exclusion pattern used for empty clause header stubs in §2.C.

> **Removed from per-chunk metadata:** `embedding_model`, `vector_store`, `token_count`, `ingestion_timestamp`, `ingestion_pipeline_version`, `schema_version`. These are pipeline/infrastructure properties, not properties of the content — stapling them to every one of thousands of chunks is redundant and couples the metadata to a specific backend or run. See §7A for where this information should live instead.

---

## 3. Why These Fields Matter for Retrieval

- **`chunk_type`** is the single highest-value derived field. Without it, "COVER PAGE," "INDEX," and "MARGINALIA" entries — which are transcription noise, not contract content — pollute similarity search results. Filtering `chunk_type == "clause"` before/after retrieval dramatically improves precision. Stamps no longer get their own `chunk_type` — see §2.C.
- **`stamps`** keeps clause text clean for embedding while preserving provenance information (who sealed the page, confidentiality markings, page-control numbers, and the full raw stamp text). A stamp was never semantically independent content — it's an annotation *about* the page it sits on — so folding it into the metadata of the real clause it accompanies (rather than emitting it as its own retrievable chunk) both improves embedding quality and keeps the information available for provenance/audit queries.
- **`clause_no` / `clause_no_normalized` / `parent_clause` / `hierarchy_level`** together enable exact-match and hierarchical lookups ("what does clause 6.8.1 say?", "show me everything under 6.7.2") that pure vector similarity handles poorly — this is where metadata filtering beats embeddings. `parent_clause` also lets you retrieve sibling/parent context around a matched chunk even when the match itself was narrow, and is what powers the sibling-clause-family expansion built on top of reranking.
- **`continues_previous`** lets the retriever stitch together a clause that was split across a page boundary, avoiding truncated/incomplete answers.
- **`pdf_page` / `printed_page`** support citation — telling the user exactly where in the physical contract an answer came from, which matters heavily in a legal/engineering-contract context.
- **`document_type` / `volume` / `chapter`** allow scoping a query to "only Specifications" or "only Volume 2" without re-embedding — essential once the corpus grows beyond these 3 chapters to the full multi-volume contract.
- **`low_confidence`** allows the system to warn users ("this clause was transcribed with lower OCR confidence — verify against the original") rather than presenting uncertain OCR output as fact.
- **`approval_status`** matters because engineering contracts get amended; once addenda are ingested, this field lets the system prefer current, approved content over superseded content.
- **`document_sequence`** preserves reading order independent of how chunks were typed or grouped — useful for reconstructing "show me this section of the document as originally laid out" even after chunk_type-based filtering has been applied.
- **`s_no` / `parent` / `section_no`** (BOQ) provide the same role for Bill-of-Quantities items that `clause_no` / `parent_clause` provide for clauses — an exact item-number lookup and section/parent grouping that vector similarity alone would handle poorly for a query like "what is BOQ item 4.2?".

## 4. Fields to Index for Metadata Filtering (ChromaDB payload indexes)

Prioritize indexing (i.e., structured/filterable, not just stored) on:
- `document_id`, `chapter`, `document_type`, `volume` (scoping queries to a document subset)
- `chunk_type` (excluding marginalia/cover/index noise from retrieval, and separating `"boq"` from clause content)
- `clause_no_normalized`, `parent_clause`, `hierarchy_level` (direct clause lookup and hierarchical navigation)
- `pdf_page` (page-range filtering, e.g. "show me pages 10–20")
- `approval_status` (retrieving only current/approved content)
- `low_confidence` (optionally excluding uncertain transcriptions from high-stakes answers)
- `s_no`, `parent`, `section_no`, `item_type` (BOQ item lookup and section/parent grouping, mirroring `clause_no`/`parent_clause` for the BOQ pipeline)

`stamps` is stored but typically not indexed for filtering unless you expect queries like "show me every page stamped confidential" — in that case index `stamps[].organization` / `stamps[].type`.

## 5. Fields That Should Never Be Embedded (Stored Alongside the Vector Only)

None of the metadata fields above should be part of the text sent to the embedding model — only the clause/item `text`/`description` (optionally prefixed with structural context for retrieval-relevant meaning) should be embedded.

**Clause pipeline:** only `clause_no` + `heading` + `section_heading` are lightly prepended to the embedded text as context. Fields that are purely payload/metadata and must **never** enter the embedding input:
`chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`, `document_id`, `source_file`, `pdf_page`, `printed_page`, `stamps`, `approval_status`, `continues_previous`, `parent_clause`, `hierarchy_level`, `low_confidence`.

**BOQ pipeline:** `contract`, `schedule`, `section_description`, `item_header_description`, `subsection_description`, `panel_reference`, `item_type`, `s_no`, and `parent` are lightly prepended as context (mirroring the clause pipeline's approach). Fields that must **never** enter the embedding input: identifiers/bookkeeping (`chunk_id`, `chunk_hash`, `chunk_number`, `document_sequence`, `source_file`, `page_label`, `language`, `approval_status`, `stamps`) and all numeric fields (`quantities`, `rate_in_inr`, `amount_in_inr`, `rate_in_foreign_currency`, `amount_in_foreign_currency`) — kept metadata-only so the embedded text stays concise and free of retrieval-irrelevant noise.

## 6. Constant vs. Per-Chunk Fields

**Constant across every clause chunk in the whole corpus (Project Metadata):**
`project_name`, `contract_number`, `employer_name`, `contractor_name`

**Constant per source file (Document Metadata, clause pipeline only):**
`document_id`, `document_name`, `document_type`, `volume`, `chapter`, `source_file`

**Changes for every clause chunk (Structural + Retrieval Metadata):**
`section_heading`, `clause_no`, `clause_no_normalized`, `parent_clause`, `hierarchy_level`, `heading`, `chunk_type`, `stamps`, `pdf_page`, `printed_page`, `continues_previous`, `low_confidence`, `chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`

**Constant per BOQ source file (dynamic, from `document_metadata`):**
whatever scalar fields that file declares (e.g. `contract`, `schedule`, `discipline`, `project_title`, `lot`)

**Changes for every BOQ chunk:**
`source_file`, `page_label`, `item_type`, `s_no`, `parent`, `section_no`, `section_description`, `item_header_no`, `item_header_description`, `panel_reference`, `subsection_no`, `subsection_description`, `unit`, `quantities`, `rate_in_inr`, `rate_in_foreign_currency`, `amount_in_inr`, `amount_in_foreign_currency`, `has_amendment`, `amendments`, `stamps`, `chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`

## 7A. Where the Removed Fields Actually Belong

`embedding_model` and `vector_store` shouldn't ride on every chunk, but you still need to know, at the collection level, which model produced the vectors currently in a given Chroma collection — otherwise a future re-embed with a different model silently mixes incompatible vector spaces. This is implemented in `storage.py`: `get_collection()` sets `metadata={"embedding_model": "BAAI/bge-m3", "chunking_strategy": "clause-level"}` directly on the ChromaDB collection object itself — **one record per collection**, not per chunk:
```json
{ "collection_name": "dmrc_be12be14_ecs", "embedding_model": "BAAI/bge-m3", "vector_store": "chromadb", "chunking_strategy": "clause-level" }
```
`token_count` — if you want it, compute it at chunking time and log it in your ETL run log for chunk-size tuning; it's a build-time diagnostic, not a durable property of the content. `ingestion_timestamp` / `ingestion_pipeline_version` — same treatment: useful in a pipeline run log, not dissertation-relevant as per-chunk payload.

---

## 7. Complete Unified Metadata Schema (JSON, Placeholder Values)

### Clause chunk example

```json
{
  "project_name": "Phase-II Delhi MRTS Project – CS to Qutab Minar & CS to Badarpur Corridors",
  "contract_number": "BE-12 LOT3 & BE-14 LOT3",
  "employer_name": "Delhi Metro Rail Corporation Ltd. (DMRC)",
  "contractor_name": "M/s Blue Star Limited",

  "document_id": "DMRC-BE12BE14-VOL2-CH1",
  "document_name": "Scope of Work – ECS (Chapter 1)",
  "document_type": "Scope of Work",
  "volume": "Volume 2",
  "chapter": "Chapter 1",
  "source_file": "DMRC_Chapter1_transcription.json",

  "section_heading": "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS",
  "clause_no": "6.7.2-1",
  "clause_no_normalized": "6.7.2.1",
  "parent_clause": "6.7.2",
  "hierarchy_level": 4,
  "heading": "Competency of Personnel",

  "pdf_page": 1,
  "printed_page": "000017",

  "chunk_id": "DMRC-BE12BE14-VOL2-CH1-p1-c6.7.2.1-000",
  "chunk_number": 42,
  "document_sequence": 57,
  "chunk_hash": "a91f3e2b1c4d",
  "language": "en",

  "chunk_type": "clause",
  "continues_previous": false,
  "low_confidence": false,

  "stamps": [
    {
      "organization": "DMRC",
      "type": "STAMP",
      "text": "GOVT OF NCT OF DELHI ... Feb 2008",
      "page_control": "000017"
    },
    {
      "organization": "Blue Star Ltd.",
      "type": "STAMP",
      "text": "BLUE STAR LIMITED - Company Seal"
    }
  ],

  "approval_status": "approved"
}
```

### BOQ chunk example (`chunk_type == "boq"`)

```json
{
  "contract": "BE-12 LOT3",
  "schedule": "Schedule B",
  "project_title": "Phase-II Delhi MRTS Project",

  "source_file": "page_007.png",
  "page_label": "B-14",

  "item_type": "boq_item",
  "s_no": "4.2",
  "parent": "4",
  "section_no": "4",
  "section_description": "Cooling Towers",
  "item_header_no": "4.2",
  "item_header_description": "Supply of Cooling Towers",
  "panel_reference": "",
  "subsection_no": "",
  "subsection_description": "",
  "unit": "Nos.",
  "quantities": { "Hauz Khas": 2, "Saket": 1 },
  "rate_in_inr": 125000.00,
  "amount_in_inr": 375000.00,
  "has_amendment": false,
  "amendments": [],

  "chunk_id": "BOQ-BE-12-LOT3-SCHEDULE-B-page-007-i4-2-001",
  "chunk_number": 12,
  "document_sequence": 12,
  "chunk_hash": "7c2a91de44f0",
  "language": "en",

  "chunk_type": "boq",
  "approval_status": "approved",
  "stamps": []
}
```

Field sets shown above reflect what the pipeline actually populates. `rate_in_foreign_currency` / `amount_in_foreign_currency` are omitted from the BOQ example (rather than shown as `null`) to illustrate the storage-layer behavior described in §2.F: absent numeric values are dropped from the stored record, not written as empty/placeholder values.