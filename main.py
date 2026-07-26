"""
main.py

Orchestrates Chapter 7's embedding generation pipeline end-to-end:

  Parsed JSON -> Chunk Objects -> Metadata Loading -> Text Normalization
  -> BGE-M3 Encoding -> Embedding Vector -> Embedding + Metadata
  -> Vector Database (ChromaDB)

Usage:
    python main.py --input data/DMRC_Chapter1_transcription.json
    python main.py --input-dir data/
"""

# --- Standard library --------------------------------------------------
import argparse
import json
import os
import sys
from typing import Any, List, Optional, Tuple

# --- Path setup: allow the local `src/` pipeline modules to be imported
# below without this script needing to live inside a proper installed
# package. Must run before the local imports that follow it.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# --- Local pipeline modules (Chapter 7) --------------------------------
from metadata_loader import build_chunk_records, build_boq_chunk_records, is_boq_json
from text_normalization import build_embedding_input, build_boq_embedding_input
from batch_embed import embed_batch
from storage import store_chunks

# chunk_type values eligible for embedding. "clause" (Chapter text) and
# "boq" (Bill-of-Quantities items) both carry real embeddable content;
# cover_page/index/marginalia/stamp remain excluded per the Chapter 7
# Chunking Strategy.
EMBEDDABLE_CHUNK_TYPES = {"clause", "boq"}


def _load_records(
    filepath: str, carry_over_record: Optional[Any] = None
) -> Tuple[str, List[Any], Optional[Any]]:
    """Parse one JSON file and build its chunk records. carry_over_record
    is the last real-clause chunk left open by the PREVIOUS file (see
    metadata_loader.build_chunk_records) so a continuation fragment at
    the very start of this file can be merged into it -- e.g. the
    Chapter 1 -> Chapter 2 boundary in this corpus. Returns (filename,
    records, open_record) where open_record should be threaded into the
    next file's call.
    """
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            parsed_json = json.load(f)
    except FileNotFoundError:
        # Re-raised as-is (same exception type/exit behavior as before)
        # after naming the offending file -- in --input-dir mode there
        # can be dozens of files in flight, and a bare traceback alone
        # doesn't say which one failed to open.
        print(f"[ERROR] Input file not found: {filepath}", file=sys.stderr)
        raise
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {filepath}: {exc}", file=sys.stderr)
        raise

    if is_boq_json(parsed_json):
        # BOQ items carry no continues_previous flag (nothing spans a
        # page/file boundary the way a clause can), so there is no new
        # open_record to produce -- pass carry_over_record straight
        # through unchanged so a later clause file in the same
        # --input-dir run still sees whatever clause was left open by an
        # earlier clause file.
        records = build_boq_chunk_records(parsed_json, filename)
        return filename, records, carry_over_record

    records, open_record = build_chunk_records(parsed_json, filename, carry_over_record=carry_over_record)
    return filename, records, open_record


def _build_embedding_text(record: Any) -> str:
    """Routes embedding-text construction by chunk_type so the existing
    Chapter (clause) pipeline is untouched while BOQ chunks get their
    own appropriately-shaped input built from BOQ metadata.
    """
    if record.metadata["chunk_type"] == "boq":
        return build_boq_embedding_input(record.metadata, record.text)

    return build_embedding_input(
        record.metadata["clause_no"],
        record.metadata["heading"],
        record.metadata["section_heading"],
        record.text,
    )


def _embed_and_store(filename: str, records: List[Any]) -> int:
    """Embed and persist the chunks already finalized for one file.

    Called only AFTER cross-file continuation merges are resolved, so a
    chunk is never embedded before text that a later file might still
    append to it (embedding a chunk, then mutating its text via a
    cross-file merge, would leave the stored vector stale).

    chunk_type in EMBEDDABLE_CHUNK_TYPES ("clause" and "boq") is eligible
    for embedding. Continuation fragments no longer exist as their own
    chunk_type -- they were merged into their parent clause by
    build_chunk_records -- and cover_page/index/marginalia/stamp entries
    are intentionally excluded per the Chapter 7 Chunking Strategy.
    """
    embeddable = [r for r in records if r.metadata["chunk_type"] in EMBEDDABLE_CHUNK_TYPES]
    n_clause = sum(1 for r in embeddable if r.metadata["chunk_type"] == "clause")
    n_boq = sum(1 for r in embeddable if r.metadata["chunk_type"] == "boq")

    print(f"[{filename}] {len(records)} chunks parsed, "
          f"{len(embeddable)} eligible for embedding "
          f"({n_clause} clause, {n_boq} boq; "
          f"{len(records) - len(embeddable)} filtered: cover/index/marginalia/stamp).")

    if not embeddable:
        return 0

    texts_for_embedding = [_build_embedding_text(r) for r in embeddable]

    vectors = embed_batch(texts_for_embedding, batch_size=32)

    chunk_ids = [r.metadata["chunk_id"] for r in embeddable]
    raw_texts = [r.text for r in embeddable]
    metadatas = [r.metadata for r in embeddable]

    n_stored = store_chunks(chunk_ids, raw_texts, vectors, metadatas)
    print(f"[{filename}] Stored {n_stored} embeddings to ChromaDB.")
    return n_stored


def process_file(filepath: str) -> int:
    """Single-file mode. No other file is available to carry a leading
    continuation fragment back into, so carry_over_record is None.
    """
    filename, records, _ = _load_records(filepath)
    return _embed_and_store(filename, records)


def process_directory(input_dir: str) -> int:
    """Batch mode: build chunk records for every file FIRST (in filename
    order, threading carry_over_record between calls so cross-file
    continuations are correctly merged), and only THEN embed and store --
    see _embed_and_store for why storage must wait until all merges
    across the whole document sequence are resolved.
    """
    try:
        json_filenames = sorted(fname for fname in os.listdir(input_dir) if fname.endswith(".json"))
    except FileNotFoundError:
        print(f"[ERROR] --input-dir path not found: {input_dir}", file=sys.stderr)
        raise

    filepaths = [os.path.join(input_dir, fname) for fname in json_filenames]

    built = []
    carry_over_record = None
    for filepath in filepaths:
        filename, records, carry_over_record = _load_records(filepath, carry_over_record)
        built.append((filename, records))

    total = 0
    for filename, records in built:
        total += _embed_and_store(filename, records)
    return total


def main() -> None:
    """CLI entry point: parse args, run the requested mode(s), report
    the total number of embeddings written. See module docstring for
    the two supported invocations (--input / --input-dir).
    """
    parser = argparse.ArgumentParser(description="DMRC Contract Embedding Generation (BGE-M3)")
    parser.add_argument("--input", type=str, help="Path to a single parsed JSON file")
    parser.add_argument("--input-dir", type=str, help="Directory of parsed JSON files")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide --input <file> or --input-dir <dir>")

    total = 0
    if args.input:
        total += process_file(args.input)
    if args.input_dir:
        total += process_directory(args.input_dir)

    print(f"\nDone. {total} clause/boq embeddings written to ./chroma_db")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Additive only: prints a clear one-line summary before letting
        # the original exception propagate, so the full traceback (and
        # the same process exit code Python already used) is preserved
        # exactly as before -- this just makes pipeline failures easy to
        # spot at the top of a long batch-mode log instead of having to
        # scroll to the bottom of a traceback to find out what broke.
        print(f"[ERROR] Embedding pipeline failed: {exc}", file=sys.stderr)
        raise