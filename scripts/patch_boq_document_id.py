"""One-time backfill: derive document_id (image-lookup key, Rule 2) for
BOQ chunks already embedded in ChromaDB, from each chunk's own
source_pdf metadata field.

Why this is needed: build_boq_chunk_records() computed document_id
locally (for chunk_id) but never wrote it into the chunk's metadata, so
_build_sources()'s viewer lookup (metadata.get("document_id")) has
always returned None for BOQ chunks. This backfills a per-source-PDF
image_document_id so it doesn't collide across boq_part1/2/3 sharing
one logical document_id (see metadata_loader.py comment for why).

Chunks with no source_pdf on file (e.g. the ADDENDUM part) are left
without a document_id -- they degrade gracefully to "no view page"
once the viewer ships, same as any other unrendered page.

Metadata-only update via collection.update() -- embeddings untouched.
Run once, alongside the existing patch scripts:

    python scripts/patch_boq_pdf_page.py
    python scripts/patch_boq_document_id.py
    python scripts/patch_boq_stamps.py
"""
import re
import sys

sys.path.insert(0, ".")
from src.storage import get_collection


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").upper()


def main() -> None:
    col = get_collection()
    res = col.get(where={"chunk_type": "boq"}, include=["metadatas"])
    ids, metas, patched, skipped = [], [], 0, 0
    for cid, md in zip(res["ids"], res["metadatas"]):
        source_pdf = md.get("source_pdf")
        if not source_pdf:
            skipped += 1
            continue
        image_document_id = f"BOQ-{_slugify(source_pdf.rsplit('.', 1)[0])}"
        if md.get("document_id") != image_document_id:
            md["document_id"] = image_document_id
            ids.append(cid)
            metas.append(md)
            patched += 1
    if ids:
        col.update(ids=ids, metadatas=metas)
    print(f"document_id backfilled: {patched} / {len(res['ids'])}")
    print(f"skipped (no source_pdf on file, e.g. ADDENDUM part): {skipped}")


if __name__ == "__main__":
    main()
