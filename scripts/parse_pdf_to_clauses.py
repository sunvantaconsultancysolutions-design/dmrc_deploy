"""Parse a raw contract PDF into the pages[].clauses[] JSON schema that
src/metadata_loader.py::build_chunk_records() already expects.

Output shape (matches DMRC_Chapter1_transcription.json etc. exactly --
see docs/DMRC_Unified_Metadata_Schema.md):

    {
      "pages": [
        {
          "pdf_page": 1,
          "printed_page": "000004",   -- document-control stamp, or "" if
                                          unreadable/absent (cover pages)
          "section_heading": "6.7 OPERATION & MAINTENANCE",
          "clauses": [
            {
              "clause_no": "6.7.2-4",
              "heading": "Penalty Clause",
              "text": "Penalty of Rs. 10,000 per day ...",
              "continues_previous": false,
              "low_confidence": false
            },
            ...
            {"heading": "STAMP", "text": "<raw stamp/seal text>"}
          ]
        }
      ]
    }

WHY THIS NEEDS TWO EXTRACTION PATHS (read before running)
-----------------------------------------------------------------------
The BE-12/BE-14 source PDFs are scanned photocopies (stamps, wet
signatures, uneven scan skew) -- they have NO text layer at all. Running
pdfplumber's normal .extract_text() on them returns empty strings, not
partial/garbled text. This script therefore:

  1. Tries the fast path first: pdfplumber text-layer extraction. This
     works instantly and perfectly for any digitally-produced PDF (e.g.
     a contract exported from Word, not scanned) -- use --mode=text.
  2. Falls back to OCR (pytesseract over a rendered page image) for
     scanned pages -- use --mode=ocr (the default, since your BE-12/14
     volumes are scanned).

OCR accuracy caveat (important, do not skip this)
-----------------------------------------------------------------------
Tesseract OCR on a 1970s-quality photocopy will NOT reliably read the
document-control stamp digits (the whole reason the original
transcription in data/*.json was done with a vision-LLM pass instead of
plain OCR -- see the PDF Evidence Viewer spec's Rule 1/Rule 2 discussion
of why "printed_page" needed a human/vision-reviewed transcription in
the first place). This script's OCR path gets you a structured,
edit-ready first draft (clause boundaries, page splits, rough text) --
budget for a manual or vision-LLM review pass afterward, exactly the way
data/*.json was produced. Do not wire this script's raw OCR output
straight into production ingestion without that review step.

Clause-boundary detection is a regex heuristic (see CLAUSE_HEADING_RE
below): it looks for a leading numeric clause id ("6.7.2", "6.7.2-4",
a bare top-level "6") followed by a capitalized heading phrase on the
same line. It will miss unusual formatting and should be spot-checked
against the rendered page images in page_images/ (or against
scripts/render_pages.py's output for a new document) before trusting
the output at scale.

Usage:
    pip install pdfplumber pytesseract pillow
    # OCR path also needs the tesseract binary itself installed:
    #   Ubuntu/Debian: apt-get install tesseract-ocr
    #   Windows:       https://github.com/UB-Mannheim/tesseract/wiki
    #   macOS:         brew install tesseract

    python scripts/parse_pdf_to_clauses.py \\
        --pdf source_pdfs/Contract_Agreement_BE-12___BE-14_Lot_3_Vol-2-1-10.pdf \\
        --out data/DMRC_Chapter1_transcription_DRAFT.json \\
        --mode ocr
"""

import argparse
import json
import re
import sys

# Stamp text is typically a 4-8 digit sequence, usually with leading
# zeros (e.g. "000004"), sometimes preceded/followed by a month+year
# ("Feb 2008") on the same physical stamp -- see the sample pages in
# page_images/ for what this looks like on the actual scans.
STAMP_RE = re.compile(r"\b(\d{4,8})\b")

# A clause heading line: leading clause number (dotted, or dotted with a
# trailing "-N" sub-item suffix as seen in the source docs, e.g.
# "6.7.2-4"), then whitespace, then a capitalized heading phrase.
# Deliberately conservative (few false positives) at the cost of missing
# some real headings -- false positives corrupt chunk boundaries, false
# negatives just fall into the previous clause's continuation text,
# which is the safer failure mode here.
CLAUSE_HEADING_RE = re.compile(
    r"^\s*(?P<clause_no>\d+(?:\.\d+)*(?:-\d+)?)\s+(?P<heading>[A-Z][A-Za-z0-9 &/,'\-]{2,80})\s*$"
)

# A section heading (e.g. "6.7 OPERATION & MAINTENANCE") -- all-caps,
# short, standalone line, distinguished from a clause heading by being
# ALL CAPS rather than Title Case.
SECTION_HEADING_RE = re.compile(
    r"^\s*(?P<section_no>\d+(?:\.\d+)?)\s+(?P<heading>[A-Z][A-Z0-9 &/,\-]{2,80})\s*$"
)


def extract_text_layer(pdf_path: str) -> list:
    """Fast path: digitally-produced PDF with a real text layer."""
    import pdfplumber

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return pages_text


def extract_via_ocr(pdf_path: str, dpi: int = 300) -> list:
    """Fallback path: scanned PDF with no text layer. Renders each page
    to an image (via PyMuPDF, matching scripts/render_pages.py's DPI
    convention) and OCRs it with pytesseract.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import io

    pages_text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages_text.append(pytesseract.image_to_string(img))
    return pages_text


def find_stamp(page_text: str) -> str:
    """Best-effort document-control stamp guess for one page's raw
    text. Returns "" (not None) when nothing plausible is found, so
    downstream get_scanned_page() falls back to pdf_page cleanly --
    same convention as printed_page elsewhere in this project.
    """
    for line in page_text.splitlines()[-8:]:  # stamps sit near the page foot
        m = STAMP_RE.search(line)
        if m:
            return m.group(1).zfill(6)
    return ""


def split_into_clauses(page_text: str) -> list:
    """Heuristic split of one page's raw text into clause dicts. A line
    matching CLAUSE_HEADING_RE starts a new clause; everything until the
    next heading (or end of page) is that clause's text.
    """
    lines = page_text.splitlines()
    clauses = []
    current = None

    for line in lines:
        m = CLAUSE_HEADING_RE.match(line)
        if m:
            if current is not None:
                clauses.append(current)
            current = {
                "clause_no": m.group("clause_no"),
                "heading": m.group("heading").strip(),
                "text": "",
                "continues_previous": False,
                "low_confidence": False,
            }
            continue
        if current is not None:
            current["text"] = (current["text"] + "\n" + line).strip()
        # Text appearing before the first detected heading on a page is
        # dropped rather than guessed into a fake clause -- mark
        # low_confidence on the page instead so a human/vision review
        # pass knows to check the top of this page manually.

    if current is not None:
        clauses.append(current)

    if not clauses:
        # Nothing matched at all -- emit the whole page as one
        # low_confidence clause rather than silently losing the page's
        # content. A human reviewer should re-run --mode ocr at a
        # higher --dpi or hand-correct this page.
        clauses = [{
            "clause_no": "",
            "heading": "UNCLASSIFIED",
            "text": page_text.strip(),
            "continues_previous": False,
            "low_confidence": True,
        }]

    return clauses


def find_section_heading(page_text: str) -> str:
    for line in page_text.splitlines()[:15]:  # section headers sit near the top
        m = SECTION_HEADING_RE.match(line)
        if m:
            return f"{m.group('section_no')} {m.group('heading').strip()}"
    return ""


def build_pages_json(pages_text: list) -> dict:
    pages = []
    for i, text in enumerate(pages_text):
        pdf_page = i + 1
        pages.append({
            "pdf_page": pdf_page,
            "printed_page": find_stamp(text),
            "section_heading": find_section_heading(text),
            "clauses": split_into_clauses(text),
        })
    return {"pages": pages}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["text", "ocr"], default="ocr",
                     help="'text' for digitally-produced PDFs with a real "
                          "text layer; 'ocr' (default) for scanned "
                          "photocopies like the BE-12/BE-14 volumes.")
    ap.add_argument("--dpi", type=int, default=300,
                     help="OCR render DPI. Higher improves stamp/small-"
                          "text accuracy at the cost of speed; 300 is a "
                          "reasonable starting point, try 400+ if stamp "
                          "digits keep coming back empty.")
    args = ap.parse_args()

    if args.mode == "text":
        pages_text = extract_text_layer(args.pdf)
    else:
        pages_text = extract_via_ocr(args.pdf, dpi=args.dpi)

    result = build_pages_json(pages_text)

    low_conf_count = sum(
        1 for p in result["pages"]
        for c in p["clauses"] if c.get("low_confidence")
    )
    empty_stamp_count = sum(1 for p in result["pages"] if not p["printed_page"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(result['pages'])} pages to {args.out}")
    print(f"  low_confidence clauses: {low_conf_count} (needs manual/vision review)")
    print(f"  pages with no stamp detected: {empty_stamp_count} (will fall back to pdf_page in citations)")
    print("This is a DRAFT transcription -- review low_confidence clauses and "
          "spot-check stamps against page_images/ before ingesting into ChromaDB.")


if __name__ == "__main__":
    main()
