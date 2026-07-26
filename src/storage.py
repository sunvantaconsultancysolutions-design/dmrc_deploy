"""
storage.py

Persists embedding vectors + finalized metadata + chunk text into a
local ChromaDB collection. Covers Chapter 7.12 (Embedding Storage).
No retrieval logic lives here -- this module only writes.
"""

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "dmrc_be12be14_ecs"

# Metadata keys that must remain numeric (float/int) across every record.
# A missing value for one of these fields is omitted from the stored
# metadata rather than coerced to a placeholder, so the key never mixes
# numeric and string types within the collection (which would break
# Chroma's numeric `where` filters, e.g. $gt / $lt / $gte / $lte).
NUMERIC_METADATA_FIELDS = {
    "rate_in_inr",
    "rate_in_foreign_currency",
    "amount_in_inr",
    "amount_in_foreign_currency",
}


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": "BAAI/bge-m3",
            "chunking_strategy": "clause-level",
        },
    )
    return collection


def sanitize_metadata(metadata: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- lists and
    nested objects (e.g. stamps[]) are JSON-encoded to strings.

    None handling:
      - For known numeric fields (see NUMERIC_METADATA_FIELDS), a None
        value means "not available" and the key is OMITTED entirely,
        so the field never mixes numeric and string types across the
        collection and remains safe for numeric range queries.
      - For all other fields, None is coerced to "" as before, since an
        empty string is a reasonable "no value" sentinel for text/
        categorical metadata.
    """
    import json
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (list, dict)):
            clean[k] = json.dumps(v)
        elif v is None:
            if k in NUMERIC_METADATA_FIELDS:
                # Omit rather than store "" -- keeps this field purely
                # numeric across all records in the collection.
                continue
            clean[k] = ""
        else:
            clean[k] = v
    return clean


def store_chunks(chunk_ids, texts, embeddings, metadatas):
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        metadatas=[sanitize_metadata(m) for m in metadatas],
    )
    return len(chunk_ids)