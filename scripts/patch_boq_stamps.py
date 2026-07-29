"""Backfill stamp_number onto BOQ chunk metadata in ChromaDB.

Matches BOQ chunks to boq_part*.json pages by (source_file, page key)
and writes the page's stamp_number into chunk metadata. Embeddings are
untouched. Run once: python scripts/patch_boq_stamps.py

CAVEAT (as of this branch): the BOQ chunk metadata currently produced
by src/metadata_loader.py::build_boq_chunk_records() does not carry a
"pdf_page" or "page_number" field at all -- only "page_label",
"source_file" (the boq_part*.json filename, not the per-page image),
and "stamps_seals". The page_key() matching below mirrors what
get_boq_page_number() in prompt_engineering.py already reads, so this
script is correct *once* pdf_page/page_number is added to BOQ chunk
metadata at ingestion time. Until then this script will report 0
patched chunks -- that is expected, not a bug in this script. See the
PR description / implementation notes for the ingestion-side fix
needed first.
"""

import glob, json, sys
sys.path.insert(0, ".")
from src.storage import get_collection


def page_key(p):
    # BOQ page objects may carry pdf_page or page_number depending on
    # the part file; mirror get_boq_page_number()'s preference order.
    return p.get("pdf_page") or p.get("page_number")


def build_stamp_map():
    stamp_map = {}
    for path in sorted(glob.glob("data/boq_part*.json")):
        data = json.load(open(path, encoding="utf-8"))
        src_names = {path.split("/")[-1]}
        meta_src = (data.get("document_metadata") or {}).get("source_file")
        if meta_src:
            src_names.add(meta_src)
        for page in data.get("pages", []):
            stamp = page.get("stamp_number")
            key = page_key(page)
            if stamp and key is not None:
                for s in src_names:
                    stamp_map[(s, int(key))] = str(stamp)
    return stamp_map


def main():
    stamp_map = build_stamp_map()
    col = get_collection()
    res = col.get(where={"chunk_type": "boq"}, include=["metadatas"])
    ids, metas, patched = [], [], 0
    for cid, md in zip(res["ids"], res["metadatas"]):
        key = md.get("pdf_page") or md.get("page_number")
        src = md.get("source_file")
        if key is None or src is None:
            continue
        stamp = stamp_map.get((src, int(key)))
        if stamp and md.get("stamp_number") != stamp:
            md["stamp_number"] = stamp
            ids.append(cid); metas.append(md); patched += 1
    if ids:
        col.update(ids=ids, metadatas=metas)
    print(f"BOQ chunks patched: {patched} / {len(res['ids'])}")
    if patched == 0:
        print(
            "NOTE: 0 patched is expected today -- BOQ chunk metadata has no "
            "pdf_page/page_number field yet (see this file's module docstring)."
        )


if __name__ == "__main__":
    main()
