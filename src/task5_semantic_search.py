"""
Task 5 - Semantic Search Module.

Viet module tim kiem ngu nghia (dense retrieval) tren vector store.

Yeu cau:
    - Input: query string + top_k
    - Output: danh sach chunks co score, sorted descending
    - Phai tuong thich voi embedding model va vector store o Task 4
"""

import os
from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)


# HyDE disabled: corpus is Vietnamese but HyDE generates English → embedding mismatch
# Using raw query directly gives better results for Vietnamese queries
ENABLE_HYDE = False

# Minimum content length to filter out boilerplate intro chunks
MIN_CONTENT_LENGTH = 10


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem ngu nghia su dung vector similarity.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,      # Noi dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or top_k <= 0:
        return []

    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []

    search_text = _generate_hypothetical_doc(query) if ENABLE_HYDE else query
    query_vector = _embed_query(search_text)
    if query_vector is None:
        return []

    # Retrieve a larger set of candidates so that post-filtering doesn't leave us empty-handed
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max(40, top_k * 4),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = _first_result_list(results.get("documents"))
    metadatas = _first_result_list(results.get("metadatas"))
    distances = _first_result_list(results.get("distances"))

    for doc, meta, dist in zip(documents, metadatas, distances):
        score = _cosine_distance_to_score(dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta or {},
        })

    output.sort(key=lambda item: item["score"], reverse=True)

    # Filter out boilerplate intro/header chunks that contain no useful game info
    output = [
        item for item in output
        if len(item.get("content", "").strip()) >= MIN_CONTENT_LENGTH
        and not _is_boilerplate(item.get("content", ""))
    ]

    return output[:top_k]


def _is_boilerplate(text: str) -> bool:
    """Detect document intro/header chunks that have no actual game information."""
    boilerplate_markers = [
        "This is a snapshot of the Uma Musume Reference doc",
        "This document assumes you've completed",
        "By Erzzy#1197",
        "Uma Musume EN Discord",
    ]
    return any(marker.lower() in text.lower() for marker in boilerplate_markers)


def _generate_hypothetical_doc(query: str) -> str:
    """
    HyDE local: tao mot doan tai lieu gia dinh ngan de embed thay vi embed query ngan.

    Ban day du co the thay helper nay bang LLM call, nhung ban local nay giu task
    chay duoc ma khong can API key.
    """
    return (
        "This Uma Musume Pretty Derby game guide answers the question: "
        f"{query}. It contains relevant details about training strategies, "
        "stats (Speed, Stamina, Power, Guts, Wisdom), skills, aptitudes, "
        "scenarios (URA, MANT, Grand Live, Aoharu), support cards, "
        "inheritance, PvP Stadium, and race mechanics."
    )


def _embed_query(text: str) -> list[float] | None:
    """
    Embed query using the same model/API used during indexing (Task 4).
    Falls back to SentenceTransformer for local models.
    """
    # OpenAI models (text-embedding-*) must use the OpenAI API
    if "text-embedding" in EMBEDDING_MODEL or EMBEDDING_MODEL.startswith("openai"):
        return _embed_openai(text)
    # Local sentence-transformer models
    model = _get_sentence_transformer()
    if model is None:
        return None
    return model.encode(text).tolist()


def _embed_openai(text: str) -> list[float] | None:
    """Call OpenAI Embeddings API - same as used in task4 indexing."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_collection():
    if not CHROMA_DIR.exists():
        return None

    try:
        import chromadb
    except ImportError:
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


def _first_result_list(value: Any) -> list:
    if not value:
        return []
    return value[0] or []


def _cosine_distance_to_score(distance: Any) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


if __name__ == "__main__":
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
