"""One-time backfill: derive pdf_page from chunk_id's embedded page_ref
(e.g. '...-page_007-...' -> 7) and write it into BOQ chunk metadata.

Why this is needed: src/metadata_loader.py::build_boq_chunk_records()
computes a per-page reference (page_ref, e.g. "page_007") to keep
chunk_id globally unique, but historically never wrote it into the
chunk's metadata as an integer pdf_page field. That field has now been
added to metadata_loader.py for future ingestions -- this script
backfills the same field onto chunks that were already embedded and
stored in ChromaDB before that fix existed.

Metadata-only update via collection.update() -- embeddings are
untouched, matching the same non-destructive pattern used by
patch_boq_stamps.py. Run once, BEFORE patch_boq_stamps.py:

    python scripts/patch_boq_pdf_page.py
    python scripts/patch_boq_stamps.py
"""
import re
import sys

sys.path.insert(0, ".")
from src.storage import get_collection

PAGE_REF_RE = re.compile(r"-page_(\d+)-")


def main() -> None:
    col = get_collection()
    res = col.get(where={"chunk_type": "boq"}, include=["metadatas"])
    ids, metas, patched = [], [], 0
    for cid, md in zip(res["ids"], res["metadatas"]):
        m = PAGE_REF_RE.search(md.get("chunk_id", ""))
        if not m:
            continue
        pdf_page = int(m.group(1))
        if md.get("pdf_page") != pdf_page:
            md["pdf_page"] = pdf_page
            ids.append(cid)
            metas.append(md)
            patched += 1
    if ids:
        col.update(ids=ids, metadatas=metas)
    print(f"pdf_page backfilled: {patched} / {len(res['ids'])}")
    if patched == 0:
        print(
            "NOTE: 0 patched -- check that chunk_id actually contains a "
            "'-page_NNN-' segment (inspect a sample chunk_id to confirm "
            "the pattern before assuming this script is broken)."
        )


if __name__ == "__main__":
    main()
