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