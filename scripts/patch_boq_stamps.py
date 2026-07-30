"""Backfill stamp_number onto BOQ chunk metadata in ChromaDB.

Matches BOQ chunks to boq_part*.json pages by (source_file, page key)
and writes the page's stamp_number into chunk metadata. Embeddings are
untouched. Run once: python scripts/patch_boq_stamps.py

UPDATE: pdf_page is now populated at ingestion time by
src/metadata_loader.py::build_boq_chunk_records() (confirmed present in
production chunk metadata, e.g. via src/validate_db.py's sample
output), so the earlier "0 patched, pdf_page doesn't exist yet" caveat
below no longer applies -- page_key() matching works correctly today.

The real remaining gap this script handles: boq_part*.json's
stamp_number field is NOT always a bare digit string. Real formats
found in the data include "illegible", "not legible in text layer
(image-based)", "top: 000013; bottom: 0019" (two separate physical
stamps per page), and "top: illegible (approx. 000016); bottom:
illegible (approx. 000022)" (an OCR guess, explicitly marked as such).
extract_stamp() below rejects any segment containing "illegible",
"approx", or "not legible" (all explicit non-reads) and otherwise
prefers the "bottom" stamp over "top" when a page has both -- per Rule
2 (a wrong stamp must only ever produce a wrong LABEL, never a wrong
page image), a guessed digit is worse than no stamp at all, so
ambiguous values are skipped rather than guessed.
"""

import glob, json, os, re, sys
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


def _segment_ok(segment: str) -> bool:
    s = segment.lower()
    return "illegible" not in s and "approx" not in s and "not legible" not in s


def _segment_digits(segment: str):
    if not _segment_ok(segment):
        return None
    m = re.search(r"\d{4,8}", segment)
    return m.group(0) if m else None


def extract_stamp(raw):
    """Recover a usable stamp digit string from stamp_number's free-text
    value, or None if nothing in it is confidently readable. See the
    module docstring above for the real-data formats this handles.
    """
    if not raw:
        return None
    raw_lower = str(raw).lower().strip()
    if raw_lower == "illegible" or "not legible" in raw_lower:
        return None

    segments = str(raw).split(";")
    if len(segments) == 1:
        return _segment_digits(segments[0])

    bottom_seg = next((s for s in segments if "bottom" in s.lower()), None)
    top_seg = next((s for s in segments if "top" in s.lower()), None)
    if bottom_seg:
        d = _segment_digits(bottom_seg)
        if d:
            return d
    if top_seg:
        d = _segment_digits(top_seg)
        if d:
            return d
    return None


def build_stamp_map():
    stamp_map = {}
    for path in sorted(glob.glob("data/boq_part*.json")):
        data = json.load(open(path, encoding="utf-8"))
        src_names = {os.path.basename(path)}
        meta_src = (data.get("document_metadata") or {}).get("source_file")
        if meta_src:
            src_names.add(meta_src)
        for page in data.get("pages", []):
            raw_stamp = page.get("stamp_number")
            key = page_key(page)
            stamp = extract_stamp(raw_stamp)
            if stamp and key is not None:
                for s in src_names:
                    stamp_map[(s, int(key))] = stamp
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
        elif not stamp and existing and not re.fullmatch(r"\d{4,8}", str(existing).strip()):
            # Clear a leftover free-text value from a prior run (e.g.
            # "not legible in text layer (image-based)") -- get_scanned_page()
            # falls back to pdf_page cleanly once stamp_number is absent.
            md["stamp_number"] = None
            ids.append(cid); metas.append(md); patched += 1
    if ids:
        col.update(ids=ids, metadatas=metas)
    print(f"BOQ chunks patched: {patched} / {len(res['ids'])}")
    if patched == 0:
        print(
            "NOTE: 0 patched -- either every BOQ chunk already has the "
            "correct stamp_number (re-run is a no-op), or none of the "
            "boq_part*.json pages backing these chunks have a recoverable "
            "stamp (check extract_stamp() output per page if this is "
            "unexpected)."
        )


if __name__ == "__main__":
    main()