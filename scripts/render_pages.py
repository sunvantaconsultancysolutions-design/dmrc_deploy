"""Render source contract PDFs to page images for the evidence viewer.

Output: page_images/{document_id}/p{pdf_page:04d}.jpg (150 dpi, JPEG q80)
Run:    python scripts/render_pages.py --pdf-dir /path/to/source_pdfs
"""

import argparse, os, sys
import fitz  # PyMuPDF (pip install pymupdf)

# document_id -> source PDF filename. Keep ids identical to
# src/metadata_loader.py DOCUMENT_METADATA for clause docs. For BOQ docs,
# the key MUST exactly match what metadata_loader.py's image_document_id
# computes: f"BOQ-{_slugify(source_pdf_without_extension)}" -- these three
# keys were generated the same way, from each boq_part*.json's own
# document_metadata.source_pdf value. If you rename a source PDF on disk,
# update BOTH the filename here AND source_pdf in that part's JSON, or
# the two will silently stop matching and images won't resolve.
DOC_PDFS = {
    "DMRC-BE12BE14-VOL2-CH1": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-1-10.pdf",
    "DMRC-BE12BE14-VOL2-CH2": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-11-20.pdf",
    "DMRC-BE12BE14-VOL2-CH3": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-20-29.pdf",
    # BOQ (Volume 3) -- CE-10 & CE-11 Lot 4. boq_part1.json's source_pdf
    # is "Contract_Agreement_CE-10___CE-11_Lot_4_Vol-3-18-27.pdf" and
    # boq_part2.json's is "..._Vol-3-28-37.pdf" -- rename your local PDF
    # files to match those exact strings (underscores and all), or edit
    # source_pdf in the JSON to match your local filenames instead.
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-18-27": "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-18-27.pdf",
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-28-37": "Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-28-37.pdf",
    # Vol-3-38-47 -- covers "5 of 42R" to "14 of 42R" of the ADDENDUM
    # schedule, overlapping boq_part3.json's "8-14 of 42R". Rendered so
    # images exist, but boq_part3.json's chunks won't link to them yet:
    # that JSON has no source_pdf field, so image_document_id falls back
    # to the shared BOQ id (see metadata_loader.py comment) until you
    # either add source_pdf to boq_part3.json's document_metadata or
    # confirm the 194-offset stamp inference and re-key it explicitly.
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-38-47":"Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-38-47.pdf",
}

DPI = 150


def render(pdf_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):  # i is 0-based
        pdf_page = i + 1  # our JSON is 1-based
        pix = page.get_pixmap(dpi=DPI)
        out = os.path.join(out_dir, f"p{pdf_page:04d}.jpg")
        pix.save(out, jpg_quality=80)
    return len(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out-dir", default="page_images")
    args = ap.parse_args()

    total = 0
    for doc_id, fname in DOC_PDFS.items():
        src = os.path.join(args.pdf_dir, fname)
        if not os.path.exists(src):
            sys.exit(f"missing source PDF: {src}")
        n = render(src, os.path.join(args.out_dir, doc_id))
        print(f"{doc_id}: {n} pages rendered")
        total += n
    print(f"Done. {total} pages in {args.out_dir}/")


if __name__ == "__main__":
    main()