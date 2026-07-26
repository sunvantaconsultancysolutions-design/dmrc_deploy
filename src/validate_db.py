"""
validate_db.py

Read-only validation script for the DMRC ChromaDB collection.
Connects to the existing collection populated by the Chapter 7
embedding pipeline and confirms that it is intact and queryable.
Covers Chapter 8.9 (Vector Indexing Workflow validation) / 8.12
(ChromaDB Implementation). This script never inserts, updates, or
deletes any data in the collection.
"""

import sys

from storage import get_collection

# BGE-M3 embeddings are expected to be 1024-dimensional. Kept as a named
# constant so the check below is self-explanatory and easy to update.
EXPECTED_EMBEDDING_DIM = 1024


def validate_collection() -> None:
    """Connect to the existing ChromaDB collection and print a summary.

    Exits with status 0 on success (including the "empty collection"
    case, which is a warning rather than a failure) and status 1 if the
    collection could not be reached or validated.
    """
    print("=" * 70)
    print("DMRC ChromaDB Collection Validation")
    print("=" * 70)

    # --- Connect and open the collection -----------------------------
    try:
        collection = get_collection()
    except Exception as exc:  # noqa: BLE001 - surface any connection issue clearly
        print(f"[FAILED] Could not connect to ChromaDB / open collection: {exc}")
        sys.exit(1)

    # The actual storage location is owned by storage.get_collection();
    # we only confirm that a connection was successfully established.
    print("[OK] Connected to ChromaDB")
    print(f"[OK] Collection opened: '{collection.name}'")
    print(f"     Collection metadata : {collection.metadata}")

    # --- Total vector count --------------------------------------------
    total_vectors = collection.count()
    print(f"\n[OK] Total vectors stored : {total_vectors}")

    if total_vectors == 0:
        print("\n[WARNING] Collection is empty. Nothing further to validate.")
        print("          Run the Chapter 7 embedding pipeline (main.py) first.")
        sys.exit(0)

    # --- Sample record (read-only) --------------------------------------
    try:
        sample = collection.get(
            limit=1,
            include=["documents", "metadatas", "embeddings"],
        )

        if not sample.get("ids"):
            print("\n[FAILED] Collection reports a non-zero count but returned no "
                  "sample records.")
            sys.exit(1)

        sample_id = sample["ids"][0]
        sample_document = sample["documents"][0]
        sample_metadata = sample["metadatas"][0]
        sample_embedding = sample["embeddings"][0]
    except Exception as exc:  # noqa: BLE001 - surface any retrieval issue clearly
        print(f"\n[FAILED] Could not retrieve sample record: {exc}")
        sys.exit(1)

    print("\n" + "-" * 70)
    print("Sample Record")
    print("-" * 70)
    print(f"Chunk ID : {sample_id}")

    print("\nSample document text (first 200 characters):")
    preview = sample_document[:200]
    print(preview + ("..." if len(sample_document) > 200 else ""))

    print("\nSample metadata:")
    if sample_metadata:
        for key, value in sample_metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  (no metadata)")

    if sample_embedding is not None and len(sample_embedding) > 0:
        embedding_dim = len(sample_embedding)
        print(f"\nSample embedding (first 5 values) : {list(sample_embedding[:5])}")
        print(f"[OK] Embedding dimension : {embedding_dim}")

        if embedding_dim != EXPECTED_EMBEDDING_DIM:
            print(f"[WARNING] Expected BGE-M3 dimension {EXPECTED_EMBEDDING_DIM}, "
                  f"got {embedding_dim}.")
    else:
        print("\n[WARNING] Sample record has no embedding to inspect.")

    print("\n" + "=" * 70)
    print(f"Validation complete. Collection '{collection.name}' is READY for querying.")
    print("=" * 70)

    sys.exit(0)


if __name__ == "__main__":
    validate_collection()