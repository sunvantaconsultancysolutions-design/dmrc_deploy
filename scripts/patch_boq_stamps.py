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

import glob, json, re, sys
sys.path.insert(0, ".")
from src.storage import get_collection

_SOURCE_PAGE_RE = re.compile(r"page_(\d+)")


def page_key(p):
    # BOQ page objects never carried a bare pdf_page/page_number field --
    # only source_file (e.g. "page_007.png"). Derive the integer page
    # index from that filename instead, matching the same convention
    # metadata_loader.py::build_boq_chunk_records() and
    # patch_boq_pdf_page.py already use for the chunk side.
    m = _SOURCE_PAGE_RE.search(p.get("source_file") or "")
    if m:
        return int(m.group(1))
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
            # Reject non-numeric stamp text (e.g. "illegible", "not legible
            # in text layer (image-based)") -- writing that into
            # stamp_number would surface as a broken citation like "Page
            # not legible in text layer...". Only a real stamped number
            # (digits, possibly with leading zeros) is usable.
            if stamp and key is not None and re.fullmatch(r"\d+", str(stamp).strip()):
                for s in src_names:
                    stamp_map[(s, int(key))] = str(stamp).strip()
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
        existing = md.get("stamp_number")
        if stamp and existing != stamp:
            md["stamp_number"] = stamp
            ids.append(cid); metas.append(md); patched += 1
        elif not stamp and existing and not re.fullmatch(r"\d+", str(existing).strip()):
            # Clear a leftover non-numeric value from a prior run (e.g.
            # "not legible in text layer (image-based)") -- get_scanned_page()
            # falls back to pdf_page cleanly once stamp_number is absent.
            md["stamp_number"] = None
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
