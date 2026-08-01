"""
patch_ch3_pdf_page.py

One-time metadata patch: fixes the CH3 (DMRC-BE12BE14-VOL2-CH3) pdf_page
off-by-one bug found in the Phase 1 audit.

Root cause: physical page 1 of "Contract Agreement BE-12 & BE-14 Lot 3
Vol-2-20-29.pdf" duplicates the last page of Vol-2-11-20.pdf (the filename
ranges overlap inclusively at page "20"). The chapter3.json transcription
correctly skipped re-transcribing that duplicate page but mislabeled its
own pdf_page counter to restart at 1 instead of continuing at 2, so every
clause in Chapter 3 pointed to the scanned-page image one page before its
actual physical location (e.g. the Penalty Clause 6.7.2-4 showed page 2's
image instead of page 3's).

This script is idempotent and safe to run multiple times: it only proceeds
if the current CH3 pdf_page range is 1-9 (the broken state); if it's
already 2-10 (already patched) it does nothing.

Run from the repo root after `pip install chromadb`:
    python patch_ch3_pdf_page.py
"""
import sys
sys.path.insert(0, ".")
from src.storage import get_collection


def main() -> None:
    col = get_collection()
    res = col.get(where={"document_id": "DMRC-BE12BE14-VOL2-CH3"}, include=["metadatas"])
    ids = res["ids"]
    metas = res["metadatas"]

    if not ids:
        print("No DMRC-BE12BE14-VOL2-CH3 chunks found -- nothing to patch.")
        return

    pages = sorted(set(int(m["pdf_page"]) for m in metas))
    print(f"Found {len(ids)} CH3 chunks. Current pdf_page range: "
          f"{min(pages)}-{max(pages)}")

    if min(pages) >= 2:
        print("Already patched (range starts at 2 or higher). No action taken.")
        return

    updated_metas = []
    for m in metas:
        new_m = dict(m)
        new_m["pdf_page"] = int(m["pdf_page"]) + 1
        updated_metas.append(new_m)

    col.update(ids=ids, metadatas=updated_metas)

    # Verify
    res2 = col.get(where={"document_id": "DMRC-BE12BE14-VOL2-CH3"}, include=["metadatas"])
    new_pages = sorted(set(int(m["pdf_page"]) for m in res2["metadatas"]))
    print(f"Patched. New pdf_page range: {min(new_pages)}-{max(new_pages)} "
          f"(expected 2-10)")

    penalty = col.get(
        where={"$and": [{"document_id": "DMRC-BE12BE14-VOL2-CH3"},
                         {"clause_no": "6.7.2-4"}]},
        include=["metadatas"],
    )
    if penalty["metadatas"]:
        print(f"Penalty clause (6.7.2-4) pdf_page is now: "
              f"{penalty['metadatas'][0]['pdf_page']} (expected 3)")


if __name__ == "__main__":
    main()
