"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path
from .env_utils import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Markdown header splitting giữ nguyên context theo section (## / ###)
# Các section lớn hơn CHUNK_SIZE sẽ được split tiếp bằng RecursiveCharacterTextSplitter
CHUNK_SIZE = 800        # Tăng từ 500 → 800: mỗi section Uma Musume khá dài
CHUNK_OVERLAP = 150     # Tăng từ 50 → 150: giảm mất context ở ranh giới chunk
CHUNKING_METHOD = "markdown_header"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "text-embedding-3-small"  # Sử dụng OpenAI API nhẹ nhàng, nhanh chóng và chất lượng cao
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _extract_character_name(filename: str, content: str):
    """
    Extract character name from char_XXXX_NNN.md files.
    Parses the first '# Profile of ...' heading in the file.
    Returns None for non-character files.
    """
    import re
    if not filename.startswith("char_"):
        return None
    match = re.search(r"^#\s+(?:Profile of|Character Profile:)\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def load_documents() -> list[dict]:
    """
    Doc toan bo markdown files tu data/standardized/.
    Voi file char_XXXX_NNN.md, tu dong extract ten nhan vat vao metadata
    de BM25 va semantic search co the match theo ten.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'character': str?}}
    """
    documents = []
    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file) else "news"
            metadata = {"source": md_file.name, "type": doc_type}

            char_name = _extract_character_name(md_file.name, content)
            if char_name:
                metadata["character"] = char_name

            documents.append({"content": content, "metadata": metadata})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using MarkdownHeaderTextSplitter.

    Strategy:
    1. Split on ## and ### headings to keep each section self-contained.
    2. Sub-split any section > CHUNK_SIZE chars with RecursiveCharacterTextSplitter
       to avoid feeding the LLM an oversized context block.
    3. Heading text is added to chunk metadata for richer citations.
    """
    try:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except ImportError:
        # Graceful fallback to the original recursive approach
        return _fallback_chunk(documents)

    chunks = []
    for doc in documents:
        md_chunks = md_splitter.split_text(doc["content"])

        chunk_index = 0
        for md_chunk in md_chunks:
            text = md_chunk.page_content
            # Merge heading metadata from splitter with document metadata
            heading_meta = md_chunk.metadata or {}

            if len(text) > CHUNK_SIZE:
                sub_texts = sub_splitter.split_text(text)
            else:
                sub_texts = [text]

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                chunks.append({
                    "content": sub_text,
                    "metadata": {
                        **doc["metadata"],
                        **heading_meta,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1

    return chunks


def _fallback_chunk(documents: list[dict]) -> list[dict]:
    """Original recursive chunking — used when langchain-text-splitters is missing."""
    chunks = []
    for doc in documents:
        splits = _fallback_split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def _fallback_split_text(text: str) -> list[str]:
    """Local recursive-like splitter used when langchain-text-splitters is missing."""
    chunks = []
    current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str) -> list[str]:
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    texts = [c["content"] for c in chunks]
    
    # Batch texts to prevent hitting single-request size/rate limits
    batch_size = 200
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} ({len(batch_texts)} chunks)...")
        response = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM
        )
        for data in response.data:
            embeddings.append(data.embedding)
            
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Clean up existing collection if any to prevent mixing stale chunks
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared old collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB has a maximum batch size of 5461, so we split it into batches of 2000
    batch_size = 2000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        print(f"Upserting batch {i // batch_size + 1} (chunks {i} to {end_idx})...")
        collection.upsert(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nOK Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"OK Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"OK Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("OK Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
