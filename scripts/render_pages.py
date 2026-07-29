"""Render source contract PDFs to page images for the evidence viewer.

Output: page_images/{document_id}/p{pdf_page:04d}.jpg (150 dpi, JPEG q80)
Run:    python scripts/render_pages.py --pdf-dir /path/to/source_pdfs
"""

import argparse, os, sys
import fitz  # PyMuPDF (pip install pymupdf)

# document_id -> source PDF filename. Keep ids identical to
# src/metadata_loader.py DOCUMENT_METADATA. Extend this dict when the
# BOQ volumes (Vol-3) are added.
DOC_PDFS = {
    "DMRC-BE12BE14-VOL2-CH1": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-1-10.pdf",
    "DMRC-BE12BE14-VOL2-CH2": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-11-20.pdf",
    "DMRC-BE12BE14-VOL2-CH3": "Contract Agreement BE-12 & BE-14 Lot 3 Vol-2-20-29.pdf",
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
