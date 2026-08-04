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
# Japanese → English name mapping for all crawled characters
# Source: uma.guide romanizations
_JP_TO_EN: dict[str, str] = {
    "スペシャルウィーク": "Special Week",
    "サイレンススズカ": "Silence Suzuka",
    "トウカイテイオー": "Tokai Teio",
    "マルゼンスキー": "Maruzensky",
    "フジキセキ": "Fuji Kiseki",
    "オグリキャップ": "Oguri Cap",
    "ゴールドシップ": "Gold Ship",
    "ウオッカ": "Vodka",
    "ダイワスカーレット": "Daiwa Scarlet",
    "タイキシャトル": "Taiki Shuttle",
    "グラスワンダー": "Grass Wonder",
    "ヒシアマゾン": "Hishi Amazon",
    "メジロマックイーン": "Mejiro McQueen",
    "エルコンドルパサー": "El Condor Pasa",
    "テイエムオペラオー": "TM Opera O",
    "ナリタブライアン": "Narita Brian",
    "シンボリルドルフ": "Symboli Rudolf",
    "エアグルーヴ": "Air Groove",
    "アグネスデジタル": "Agnes Digital",
    "セイウンスカイ": "Seiun Sky",
    "タマモクロス": "Tamamo Cross",
    "ファインモーション": "Fine Motion",
    "ビワハヤヒデ": "Biwa Hayahide",
    "マヤノトップガン": "Mayano Top Gun",
    "マンハッタンカフェ": "Manhattan Cafe",
    "ミホノブルボン": "Mihono Bourbon",
    "メジロライアン": "Mejiro Ryan",
    "ヒシミラクル": "Hishi Miracle",
    "エイシンフラッシュ": "Eishin Flash",
    "カレンチャン": "Karen Chan",
    "コパノリッキー": "Copano Rickey",
    "ドリームジャーニー": "Dream Journey",
    "ゴールドシチー": "Gold City",
    "スターリングローズ": "Starling Rose",
    "フクキタル": "Fuku Kitaru",
    "ライスシャワー": "Rice Shower",
    "イクノディクタス": "Ikuno Dictus",
    "スマートファルコン": "Smart Falcon",
    "エルコンドルパサー": "El Condor Pasa",
    "サクラバクシンオー": "Sakura Bakushin O",
    "ナカヤマフェスタ": "Nakayama Festa",
    "ヴィルシーナ": "Virshina",
    "ヴィブロス": "Vivlos",
    "アドマイヤベガ": "Admire Vega",
    "ニシノフラワー": "Nishino Flower",
    "ハルウララ": "Haru Urara",
    "キタサンブラック": "Kitasan Black",
    "サトノダイヤモンド": "Satono Diamond",
    "シリウスシンボリ": "Sirius Symboli",
    "メジロアルダン": "Mejiro Ardan",
    "ロイスアンドロイス": "Royce and Royce",
    "アグネスタキオン": "Agnes Tachyon",
    "アドマイヤグルーヴ": "Admire Groove",
    "サクラローレル": "Sakura Laurel",
    "ツインターボ": "Twin Turbo",
    "テンポイント": "Tenpointö",
    "ニッポンテイオー": "Nippon Teio",
    "マーベラスサンデー": "Marvelous Sunday",
    "タニノギムレット": "Tanino Gimlet",
    "ゼンノロブロイ": "Zenno Rob Roy",
    "キングヘイロー": "King Halo",
    "ナリタトップロード": "Narita Top Road",
    "ユキノビジン": "Yukino Bijin",
    "トーセンジョーダン": "Tosen Jordan",
    "スーパークリーク": "Super Creek",
    "イナリワン": "Inari One",
    "カツラギエース": "Katsuragi Ace",
    "ダイイチルビー": "Daiichi Ruby",
    "ハッピーミーク": "Happy Meek",
    "ウインバリアシオン": "Win Variation",
    "メジロパーマー": "Mejiro Palmer",
    "ドゥラメンテ": "Duramente",
    "ラインクラフト": "Linecraft",
    "メジロドーベル": "Mejiro Dober",
    "テイオー": "Tokai Teio",
    "サクラチヨノオー": "Sakura Chiyono O",
    "メジロブライト": "Mejiro Bright",
    "トランセンド": "Transcend",
    "ノースフライト": "North Flight",
    "シングウィズミー": "Sing With Me",
    "エスポワールシチー": "Espoir City",
    "ナカヤマフェスタ": "Nakayama Festa",
    "メジロラモーヌ": "Mejiro Ramonu",
    "サトノクラウン": "Satono Crown",
    "シュヴァルグラン": "Cheval Grand",
    "ラッキーライラック": "Lucky Lilac",
    "グランアレグリア": "Gran Alegria",
    "ラヴズオンリーユー": "Loves Only You",
    "ルーラーシップ": "Rulership",
    "ナリタブライアン": "Narita Brian",
    "ケイエスミラクル": "KS Miracle",
    "スティルインラブ": "Still in Love",
    "デュランダル": "Durandal",
    "フサイチパンドラ": "Fusaichi Pandora",
    "カルストンライトオ": "Karuston Light O",
    "ヒシミラクル": "Hishi Miracle",
    "ハルウララ": "Haru Urara",
}


def _extract_character_name(filename: str, content: str):
    """
    Extract character name from char_XXXX_NNN.md files.
    Returns (japanese_name, english_name) or None.
    """
    import re
    if not filename.startswith("char_"):
        return None
    match = re.search(r"^#\s+(?:Profile of|Character Profile:)\s+(.+)$", content, re.MULTILINE)
    if not match:
        return None
    jp_name = match.group(1).strip()
    en_name = _JP_TO_EN.get(jp_name)
    return jp_name, en_name


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

            char_info = _extract_character_name(md_file.name, content)
            if char_info:
                jp_name, en_name = char_info
                metadata["character_jp"] = jp_name
                metadata["character_en"] = en_name or ""
                metadata["character"] = f"{jp_name} ({en_name})" if en_name else jp_name

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

        metadata = doc["metadata"]
        chunk_index = 0
        for md_chunk in md_chunks:
            text = md_chunk.page_content

            # Assign prefix dynamically: chunk_index 0 gets the rich prefix
            current_prefix = ""
            if "character_en" in metadata and metadata["character_en"]:
                en = metadata["character_en"]
                jp = metadata["character_jp"]
                if chunk_index == 0:
                    current_prefix = f"Nhân vật (Character): {en} ({jp}) - profile, thông tin cá nhân, stats, birthdate, adaptability, cự ly\n\n"
                else:
                    current_prefix = f"Nhân vật (Character): {en} ({jp})\n\n"
            elif "character_jp" in metadata and metadata["character_jp"]:
                jp = metadata["character_jp"]
                if chunk_index == 0:
                    current_prefix = f"Nhân vật (Character): {jp} - profile, thông tin cá nhân, stats, birthdate, adaptability, cự ly\n\n"
                else:
                    current_prefix = f"Nhân vật (Character): {jp}\n\n"

            # Merge heading metadata from splitter with document metadata
            heading_meta = md_chunk.metadata or {}

            if len(text) > CHUNK_SIZE:
                sub_texts = sub_splitter.split_text(text)
            else:
                sub_texts = [text]

            for sub_text in sub_texts:
                # Filter out chunks that are too short to be useful (e.g. empty headers/titles)
                # 60 chars is a safe limit to preserve profiles and event lists but filter noise.
                if len(sub_text.strip()) < 60:
                    continue
                # If chunk was sub-split, ensure sub_text keeps the prefix if it's not already there
                final_text = sub_text
                if current_prefix and not final_text.startswith(current_prefix.strip()):
                    final_text = current_prefix + final_text

                chunks.append({
                    "content": final_text,
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
        
        char_prefix = ""
        metadata = doc["metadata"]
        if "character_en" in metadata and metadata["character_en"]:
            char_prefix = f"Nhân vật (Character): {metadata['character_en']} ({metadata['character_jp']})\n\n"
        elif "character_jp" in metadata and metadata["character_jp"]:
            char_prefix = f"Nhân vật (Character): {metadata['character_jp']}\n\n"

        for i, chunk_text in enumerate(splits):
            if len(chunk_text.strip()) < 60:
                continue
            
            # Assign prefix dynamically: the first split is the profile chunk.
            current_prefix = ""
            if "character_en" in metadata and metadata["character_en"]:
                en = metadata["character_en"]
                jp = metadata["character_jp"]
                if i == 0:
                    current_prefix = f"Nhân vật (Character): {en} ({jp}) - profile, thông tin cá nhân, stats, birthdate, adaptability, cự ly\n\n"
                else:
                    current_prefix = f"Nhân vật (Character): {en} ({jp})\n\n"
            elif "character_jp" in metadata and metadata["character_jp"]:
                jp = metadata["character_jp"]
                if i == 0:
                    current_prefix = f"Nhân vật (Character): {jp} - profile, thông tin cá nhân, stats, birthdate, adaptability, cự ly\n\n"
                else:
                    current_prefix = f"Nhân vật (Character): {jp}\n\n"

            final_text = chunk_text
            if current_prefix:
                final_text = current_prefix + final_text
            chunks.append({
                "content": final_text,
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
