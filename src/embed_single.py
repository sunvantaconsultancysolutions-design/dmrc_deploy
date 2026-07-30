"""
embed_single.py

Encodes a single clause chunk into a dense embedding vector using the
BAAI/bge-m3 model. This module covers Chapter 7.9 (Local Python
Implementation) of the Software Design Report.

Note: this module does NOT load its own copy of the BGE-M3 model. It
reuses the lazily-loaded, process-wide singleton exposed by
batch_embed.get_model() (Chapter 7.10), so a process that performs both
single-chunk and batch embedding only ever holds one ~2.27GB model
instance in memory instead of two.
"""

from .batch_embed import get_model


def embed_chunk(text: str):
    """Encodes a single normalized clause string into a dense vector.

    normalize_embeddings=True L2-normalizes the output vector so that
    a plain dot product between two embeddings is equivalent to cosine
    similarity -- this is a storage/encoding-time property, independent
    of how retrieval later uses the vectors.
    """
    model = get_model()
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


if __name__ == "__main__":
    sample_text = (
        "Clause 2.1 | Scope of work | Contract BE-12 LOT3 & BE-14 LOT3 "
        "Scope of work - ECS\n"
        "The detailed scope of work for ENVIRONMENT CONTROL SYSTEM (ECS) "
        "is described in the specification."
    )
    vector = embed_chunk(sample_text)
    print(f"Embedding dimension: {vector.shape[0]}")
    print(f"L2 norm (should be ~1.0): {(vector ** 2).sum() ** 0.5:.6f}")
