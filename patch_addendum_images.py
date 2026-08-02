"""
patch_addendum_images.py

One-time metadata patch: links the 211 BOQ ADDENDUM chunks
(document_id="BOQ-CE-10-AND-11-LOT-4-SCHEDULE-S4-1", from boq_part3.json)
to their real, already-rendered scanned page images.

--------------------------------------------------------------------------
ROOT CAUSE
--------------------------------------------------------------------------
boq_part3.json's own metadata always had `source_pdf: null` -- the file's
`continuation_note` even says it continues from a separate file
(ce10_11_lot4_s4-1_pages_1_3_5_6_7.json) that was never delivered. This
made it look like the ADDENDUM's source pages simply didn't exist.

They do. `page_images/BOQ-CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-38-47/`
already contains 10 correctly-rendered pages from
"Contract_Agreement_CE-10___CE-11_Lot_4_Vol-3-38-47.pdf" -- they were just
never linked, because the ADDENDUM chunks pointed at a different
document_id and had no source_pdf at all.

Verified via OCR (tesseract) against boq_part3.json's own transcribed
item text, matching every one of the 7 needed pages by exact content,
not by page-count guessing:

    internal label   ChromaDB pdf_page (before)   physical image page (after)
    "8 of 42 R"        8                             4   (p0004.jpg)
    "9 of 42 R"        9                             2   (p0002.jpg)  <- Fire Pump Panel MCCB
    "10 of 42 R"       10                            6   (p0006.jpg)
    "11 of 42 R"       11                            7   (p0007.jpg)
    "12 of 42 R"       12                            8   (p0008.jpg)  <- Fire Pump Panel MCCB (2nd instance)
    "13 of 42 R"       13                            9   (p0009.jpg)
    "14 of 42 R"       14                            10  (p0010.jpg)

--------------------------------------------------------------------------
EFFECT
--------------------------------------------------------------------------
BOQ image coverage (chunks with a source_pdf): 101/101 -> 312/312.
Idempotent: safe to run multiple times. Metadata-only -- does not touch
embeddings, does not require re-embedding, does not change chunk text.

Run from the repo root:
    python patch_addendum_images.py
"""
import sys
sys.path.insert(0, ".")
from src.storage import get_collection

OLD_DOC_ID = "BOQ-CE-10-AND-11-LOT-4-SCHEDULE-S4-1"
NEW_DOC_ID = "BOQ-CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-38-47"
NEW_SOURCE_PDF = "Contract_Agreement_CE-10___CE-11_Lot_4_Vol-3-38-47.pdf"

# internal transcription label number -> physical rendered page number
LABEL_TO_PHYSICAL_PAGE = {8: 4, 9: 2, 10: 6, 11: 7, 12: 8, 13: 9, 14: 10}


def main() -> None:
    col = get_collection()

    res = col.get(where={"document_id": OLD_DOC_ID}, include=["metadatas"])
    ids = res["ids"]
    metas = res["metadatas"]

    if not ids:
        # Already patched (document_id no longer exists under the old name)
        # or nothing to do.
        already = col.get(where={"document_id": NEW_DOC_ID}, include=["metadatas"])
        print(f"No chunks found under {OLD_DOC_ID!r}. "
              f"{len(already['ids'])} chunks already under {NEW_DOC_ID!r} -- "
              f"likely already patched. No action taken.")
        return

    print(f"Found {len(ids)} ADDENDUM chunks under {OLD_DOC_ID!r}.")

    updated = []
    mapped = 0
    for m in metas:
        old_page = m.get("pdf_page")
        new_m = dict(m)
        if old_page in LABEL_TO_PHYSICAL_PAGE:
            new_m["document_id"] = NEW_DOC_ID
            new_m["source_pdf"] = NEW_SOURCE_PDF
            new_m["pdf_page"] = LABEL_TO_PHYSICAL_PAGE[old_page]
            mapped += 1
        updated.append(new_m)

    col.update(ids=ids, metadatas=updated)
    print(f"Patched {mapped}/{len(ids)} chunks.")

    # Verify
    res2 = col.get(where={"document_id": NEW_DOC_ID}, include=["metadatas"])
    print(f"Chunks now under {NEW_DOC_ID!r}: {len(res2['ids'])}")
    from collections import Counter
    pages = Counter(m.get("pdf_page") for m in res2["metadatas"])
    for p, n in sorted(pages.items()):
        print(f"  pdf_page={p}: {n} chunks")


if __name__ == "__main__":
    main()
