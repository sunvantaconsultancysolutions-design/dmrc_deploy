"""
diag_stamp_check.py

MAINTAINABILITY FIX: this was previously committed as `diag.py` at the
repo ROOT (not in scripts/, alongside every other CLI utility), a stray
one-off debug script left over from stamp-extraction work.

Ad-hoc diagnostic: prints scripts/patch_boq_stamps.py's stamp_map
alongside the current stamp_number state actually stored in
chroma_db/chroma.sqlite3. Queries the SQLite file directly with raw
SQL rather than going through storage.get_collection() -- this couples
the script to ChromaDB's internal on-disk schema (the
`embedding_metadata` key/string_value/int_value table layout), which
is exactly the kind of coupling that has already broken once in this
project (an unpinned chromadb install silently rewrote this schema,
see requirements.txt's chromadb==0.5.5 pin comment). Kept as-is here
(not rewritten to use the collection API) since its only purpose is
inspecting the raw on-disk state for debugging, not production
retrieval -- but be aware a chromadb version bump can break this
script's queries even if the app itself still works.

Run from the repo root:
    python scripts/diag_stamp_check.py
"""
import sys, sqlite3
sys.path.insert(0, ".")
from scripts.patch_boq_stamps import build_stamp_map

sm = build_stamp_map()
print("=== stamp_map (should have NO data\\ prefix now) ===")
print("size:", len(sm))
for k, v in list(sm.items())[:6]:
    print(k, "->", v)

con = sqlite3.connect("chroma_db/chroma.sqlite3")
cur = con.cursor()

print()
print("=== current stamp_number state in your chroma_db ===")
cur.execute("select count(*) from embedding_metadata where key='stamp_number'")
print("rows with stamp_number set:", cur.fetchone())

print()
print("=== sample of actual chunk (source_file, pdf_page) pairs ===")
cur.execute("""
select s.string_value, p.int_value from embedding_metadata s
join embedding_metadata p on p.id = s.id and p.key='pdf_page'
join embedding_metadata t on t.id = s.id and t.key='chunk_type' and t.string_value='boq'
where s.key='source_file' limit 6
""")
for row in cur.fetchall():
    print(row)