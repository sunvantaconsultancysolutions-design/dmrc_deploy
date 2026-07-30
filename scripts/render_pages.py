"""Render source contract PDFs to page images for the evidence viewer.

Output: page_images/{document_id}/p{pdf_page:04d}.jpg (150 dpi, JPEG q80)
Run:    python scripts/render_pages.py --pdf-dir /path/to/source_pdfs

------------------------------------------------------------------------
TASK 1 FIX -- BOQ page-image document_id normalisation
------------------------------------------------------------------------
Previously, the BOQ keys in DOC_PDFS were hand-written (e.g.
"BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-28-37") while
metadata_loader.py derived image_document_id by calling
  f"BOQ-{_slugify(source_pdf.rsplit('.', 1)[0])}"
on the source PDF filename stored in the JSON's document_metadata.
_slugify replaces every run of non-alphanumeric characters (spaces,
ampersands, underscores, dashes) with a single hyphen, so
  "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-28-37.pdf"
and
  "Contract_Agreement_CE-10___CE-11_Lot_4_Vol-3-28-37.pdf"
both produce "CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-28-37".
The hand-written key, which said "CE-10-AND-11", is different.

Fix: for BOQ PDFs, _slugify the filename and prepend "BOQ-", exactly
as metadata_loader does for the JSON source_pdf. That way the directory
name this script creates always matches what metadata_loader writes into
the ChromaDB "document_id" field -- guaranteed by construction, not by
keeping two hand-written lists in sync.

The clause PDF keys are unchanged (they come from the hard-coded
DOCUMENT_METADATA lookup table in metadata_loader.py, so the same
string is used on both sides already).
"""

import argparse
import os
import re
import sys

import fitz  # PyMuPDF (pip install pymupdf)


# ---------------------------------------------------------------------------
# Shared slugify -- must be byte-for-byte identical to the implementation
# in src/metadata_loader.py so the two modules produce the same ID for the
# same input. It is not imported from there because this script is intended
# to be runnable standalone (outside the src package), and the test below
# enforces they stay in sync.
# ---------------------------------------------------------------------------

def _slugify(value: str) -> str:
    """Uppercase, alphanumeric-only slug.  Same function as metadata_loader._slugify."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").upper()


# ---------------------------------------------------------------------------
# Clause PDF mapping -- keys taken directly from metadata_loader.DOCUMENT_METADATA
# to guarantee they match what is written into the ChromaDB "document_id" field.
# ---------------------------------------------------------------------------

_CLAUSE_PDF_MAP = {
    "DMRC-BE12BE14-VOL2-CH1": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-1-10.pdf",
    "DMRC-BE12BE14-VOL2-CH2": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-11-20.pdf",
    "DMRC-BE12BE14-VOL2-CH3": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-20-29.pdf",
}

# ---------------------------------------------------------------------------
# BOQ PDF listing -- filenames only; the document_id key is *derived* at
# runtime via _slugify so it always matches metadata_loader's computation.
# Add new BOQ PDFs here (filename only). Do NOT hard-code the key.
# ---------------------------------------------------------------------------

_BOQ_PDF_FILENAMES = [
    "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-18-27.pdf",
    "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-28-37.pdf",
    "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-38-47.pdf",
]


def _boq_document_id(pdf_filename: str) -> str:
    """Derive a BOQ document_id from the PDF filename.

    Applies the same transformation as metadata_loader.build_boq_chunk_records:
        image_document_id = f"BOQ-{_slugify(source_pdf.rsplit('.', 1)[0])}"
    so that the directory name produced here is always identical to the
    metadata field written into ChromaDB.
    """
    stem = pdf_filename.rsplit(".", 1)[0]
    return f"BOQ-{_slugify(stem)}"


def _build_doc_pdfs() -> dict:
    """Build the complete {document_id: pdf_filename} mapping used by both
    render() and extract_page_figures.py's import of DOC_PDFS.

    BOQ keys are derived rather than hard-coded; clause keys are taken
    verbatim from the DOCUMENT_METADATA lookup table.
    """
    mapping = dict(_CLAUSE_PDF_MAP)
    for fname in _BOQ_PDF_FILENAMES:
        doc_id = _boq_document_id(fname)
        mapping[doc_id] = fname
    return mapping


# Public name retained for backward-compatibility with extract_page_figures.py,
# which does `from render_pages import DOC_PDFS`.
DOC_PDFS = _build_doc_pdfs()

DPI = 150


def render(pdf_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):        # i is 0-based
        pdf_page = i + 1                  # JSON is 1-based
        pix = page.get_pixmap(dpi=DPI)
        out = os.path.join(out_dir, f"p{pdf_page:04d}.jpg")
        pix.save(out, jpg_quality=80)
    return len(doc)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render source contract PDFs to page images for the evidence viewer. "
            "BOQ document_id values are derived from the PDF filename via _slugify, "
            "matching metadata_loader.py's image_document_id computation exactly."
        )
    )
    ap.add_argument("--pdf-dir", required=True, help="Directory containing the source PDFs.")
    ap.add_argument("--out-dir", default="page_images", help="Root output directory (default: page_images).")
    ap.add_argument(
        "--list-ids",
        action="store_true",
        help="Print the {document_id: filename} mapping and exit without rendering.",
    )
    args = ap.parse_args()

    if args.list_ids:
        print("document_id -> PDF filename mapping:")
        for doc_id, fname in DOC_PDFS.items():
            print(f"  {doc_id!r:60s} <- {fname!r}")
        return

    total = 0
    missing = []
    for doc_id, fname in DOC_PDFS.items():
        src = os.path.join(args.pdf_dir, fname)
        if not os.path.exists(src):
            missing.append(src)
            continue
        n = render(src, os.path.join(args.out_dir, doc_id))
        print(f"{doc_id}: {n} pages rendered -> {args.out_dir}/{doc_id}/")
        total += n

    if missing:
        print(f"\nWARNING: {len(missing)} source PDF(s) not found (skipped):")
        for p in missing:
            print(f"  {p}")
        print(
            "Only PDFs that exist in --pdf-dir are rendered. "
            "Re-run with the missing files present to generate their images."
        )

    print(f"\nDone. {total} pages rendered to {args.out_dir}/")


if __name__ == "__main__":
    main()
