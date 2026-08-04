"""
Task 6 - Lexical Search Module (BM25).

Mac dinh su dung BM25. Neu dung phuong phap khac (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hay giai thich co che trong buoi demo -> +5 bonus.

Cai dat:
    pip install rank-bm25

BM25 hoat dong the nao:
    - Term Frequency (TF): tu xuat hien nhieu trong document -> diem cao
    - Inverse Document Frequency (IDF): tu hiem -> quan trong hon
    - Document length normalization: document dai khong bi uu tien qua muc
    - Formula: score(q,d) = sum IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from collections import Counter
import math
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "university_services_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# List of {'content': str, 'metadata': dict}. Tests or notebooks can also set this manually.
CORPUS: list[dict] = []
_BM25_INDEX = None
_BM25_CORPUS_SIZE = 0


def build_bm25_index(corpus: list[dict]):
    """
    Xay dung BM25 index tu corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    if not any(tokenized_corpus):
        return None

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return _SimpleBM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem tu khoa su dung BM25.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or top_k <= 0:
        return []

    corpus = _get_corpus()
    if not corpus:
        return []

    bm25 = _get_bm25_index()
    if bm25 is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = list(bm25.get_scores(query_tokens))
    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": corpus[idx].get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _get_corpus() -> list[dict]:
    global CORPUS

    if CORPUS:
        return CORPUS

    # Prioritize loading from ChromaDB to ensure BM25 uses the exact same chunks and prefixes
    CORPUS = _load_corpus_from_chroma()
    if not CORPUS:
        try:
            from .task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            CORPUS = chunk_documents(docs)
        except ImportError:
            # Fallback to local custom load if task4 is not importable
            CORPUS = _load_corpus_from_markdown()
    return CORPUS


def _get_bm25_index():
    global _BM25_CORPUS_SIZE, _BM25_INDEX

    corpus = _get_corpus()
    if _BM25_INDEX is None or _BM25_CORPUS_SIZE != len(corpus):
        _BM25_INDEX = build_bm25_index(corpus)
        _BM25_CORPUS_SIZE = len(corpus)
    return _BM25_INDEX


def _load_corpus_from_markdown() -> list[dict]:
    if not STANDARDIZED_DIR.exists():
        return []

    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_parts = {part.lower() for part in md_file.relative_to(STANDARDIZED_DIR).parts}
        doc_type = "legal" if "legal" in relative_parts else "news" if "news" in relative_parts else "unknown"

        for chunk_index, chunk in enumerate(_chunk_text(content)):
            corpus.append({
                "content": chunk,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "doc_type": doc_type,
                    "chunk_index": chunk_index,
                },
            })
    return corpus


def _load_corpus_from_chroma() -> list[dict]:
    if not CHROMA_DIR.exists():
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        data = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    return [
        {"content": doc, "metadata": meta or {}}
        for doc, meta in zip(documents, metadatas)
        if doc
    ]


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class _SimpleBM25Okapi:
    """Small BM25 fallback used only when rank-bm25 is not installed."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.corpus_size = len(tokenized_corpus)
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.idf = self._calculate_idf()

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_len):
            score = 0.0
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / norm
            scores.append(score)
        return scores

    def _calculate_idf(self) -> dict[str, float]:
        doc_counts = Counter()
        for freqs in self.doc_freqs:
            doc_counts.update(freqs.keys())

        return {
            token: math.log(1 + (self.corpus_size - count + 0.5) / (count + 0.5))
            for token, count in doc_counts.items()
        }


if __name__ == "__main__":
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
