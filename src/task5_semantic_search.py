"""
Task 5 - Semantic Search Module.

Viet module tim kiem ngu nghia (dense retrieval) tren vector store.

Yeu cau:
    - Input: query string + top_k
    - Output: danh sach chunks co score, sorted descending
    - Phai tuong thich voi embedding model va vector store o Task 4
"""

from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


ENABLE_HYDE = True


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
    model = _get_embedding_model()
    if model is None:
        return []
    query_vector = model.encode(search_text).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
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
    return output[:top_k]


def _generate_hypothetical_doc(query: str) -> str:
    """
    HyDE local: tao mot doan tai lieu gia dinh ngan de embed thay vi embed query ngan.

    Ban day du co the thay helper nay bang LLM call, nhung ban local nay giu task
    chay duoc ma khong can API key.
    """
    return (
        "This university policy or student service document answers the question: "
        f"{query}. It contains relevant details, requirements, procedures, fees, "
        "eligibility conditions, deadlines, contact points, and official guidance."
    )


@lru_cache(maxsize=1)
def _get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
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
