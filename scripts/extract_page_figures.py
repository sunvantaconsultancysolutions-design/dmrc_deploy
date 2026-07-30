"""Extract embedded figures/diagrams from source PDFs -- distinct from
scripts/render_pages.py, which renders the WHOLE page as one image for
the evidence viewer. This script pulls out individual embedded raster
objects (SLDs, wiring diagrams, equipment photos) so they can be shown
inline next to an answer, separately from the full scanned page.

Output: figure_images/{document_id}/p{pdf_page:04d}_fig{n}.{ext}
        figure_images/manifest.json  -- {(document_id, pdf_page): [filenames]}

WHY THIS IS A SEPARATE STEP FROM render_pages.py (read before running)
-----------------------------------------------------------------------
For a fully-scanned photocopy page (which is what most of BE-12/BE-14
Vol-2 is), the ENTIRE page is one embedded image -- PyMuPDF's
page.get_images() will return that single full-page scan as "an
embedded image", which is not a distinct figure at all, it's just the
page render_pages.py already produces. Extracting it again here would
be redundant and would flood the manifest with one useless "figure" per
page.

This script guards against that with AREA_RATIO_THRESHOLD: an embedded
image covering more than that fraction of the page's total area is
assumed to BE the scanned page itself and is skipped, not extracted as
a figure. Only smaller embedded images -- the actual case where a
diagram/photo/SLD is embedded within an otherwise digitally-produced
page (e.g. tender drawing volumes, datasheets) -- are kept.

Tune AREA_RATIO_THRESHOLD per corpus: if you run this against a volume
that turns out to have genuinely large embedded diagrams that fill most
of a page, lower it cautiously and manually check the output count
first -- an unexpectedly large number of extracted files is the signal
something is misconfigured, not a signal to raise the threshold further.

Usage:
    pip install pymupdf

    python scripts/extract_page_figures.py \\
        --pdf-dir source_pdfs \\
        --out-dir figure_images
"""

import argparse
import json
import os
import sys

import fitz  # PyMuPDF

# Same document_id -> filename mapping convention as render_pages.py.
# Duplicated here rather than imported, so this script can run standalone
# against a single new PDF without requiring render_pages.py's exact
# DOC_PDFS dict to already contain that file.
from render_pages import DOC_PDFS  # type: ignore

AREA_RATIO_THRESHOLD = 0.6  # skip any embedded image covering >60% of the page
MIN_DIMENSION_PX = 80       # skip tiny embedded images (icons, logos, noise)


def extract_figures(pdf_path: str, document_id: str, out_dir: str) -> list:
    """Returns a list of {pdf_page, filename} dicts for this one PDF."""
    doc = fitz.open(pdf_path)
    doc_out_dir = os.path.join(out_dir, document_id)
    os.makedirs(doc_out_dir, exist_ok=True)

    entries = []
    for i, page in enumerate(doc):
        pdf_page = i + 1
        page_area = page.rect.width * page.rect.height
        fig_n = 0

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_w, img_h = base_image.get("width", 0), base_image.get("height", 0)

            if img_w < MIN_DIMENSION_PX or img_h < MIN_DIMENSION_PX:
                continue  # too small to be a meaningful figure

            # Estimate this image's on-page area via its placement rects
            # (an xref can appear more than once per page in rare cases;
            # use the largest placement to decide whether it's "the
            # whole page" or a genuine embedded figure).
            rects = page.get_image_rects(xref)
            placed_area = max((r.width * r.height for r in rects), default=0)
            if page_area > 0 and (placed_area / page_area) > AREA_RATIO_THRESHOLD:
                continue  # this embedded image IS the scanned page itself

            fig_n += 1
            ext = base_image.get("ext", "png")
            fname = f"p{pdf_page:04d}_fig{fig_n}.{ext}"
            out_path = os.path.join(doc_out_dir, fname)
            with open(out_path, "wb") as f:
                f.write(base_image["image"])

            entries.append({"pdf_page": pdf_page, "filename": fname})

    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out-dir", default="figure_images")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest: dict = {}
    total = 0
    for doc_id, fname in DOC_PDFS.items():
        src = os.path.join(args.pdf_dir, fname)
        if not os.path.exists(src):
            print(f"skipping {doc_id}: missing source PDF {src}", file=sys.stderr)
            continue
        entries = extract_figures(src, doc_id, args.out_dir)
        if entries:
            manifest[doc_id] = entries
        print(f"{doc_id}: {len(entries)} figures extracted")
        total += len(entries)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {total} figures across {len(manifest)} documents.")
    print(f"Manifest written to {manifest_path}")
    if total == 0:
        print(
            "0 figures found -- expected if your source PDFs are pure "
            "full-page scans with no separately embedded diagrams "
            "(check AREA_RATIO_THRESHOLD if you expected otherwise)."
        )


if __name__ == "__main__":
    main()
