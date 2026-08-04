"""
Uma Musume RAG Chatbot — Giao diện do Nguyễn Minh Phúc thiết kế.
Tích hợp RAG Pipeline (Task 9 + Task 10) vào UI.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Uma Musume Reference Library",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# HELPER — ảnh local fallback sang URL
# =============================================================================

def get_image_src(name: str, fallback_url: str) -> str:
    """Dùng ảnh local nếu có, không thì dùng URL."""
    local_paths = [
        PROJECT_ROOT / "assets" / f"{name}.png",
        PROJECT_ROOT / "assets" / f"{name}.jpg",
        PROJECT_ROOT / "data" / "images" / f"{name}.png",
    ]
    for p in local_paths:
        if p.exists():
            return str(p)
    return fallback_url

# =============================================================================
# CUSTOM CSS — thiết kế của Phúc
# =============================================================================

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #fdf2f8 50%, #f0f9ff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fdf2f8 100%);
        border-right: 2px solid #8cc63f;
    }

    .stButton > button {
        background: #e83984 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(232, 57, 132, 0.3);
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px);
        background: #f64b82 !important;
        box-shadow: 0 8px 12px rgba(232, 57, 132, 0.4);
    }

    @keyframes breathing {
        0%   { transform: scale(1) translateY(0px); }
        50%  { transform: scale(1.03) translateY(-6px); }
        100% { transform: scale(1) translateY(0px); }
    }

    .uma-card {
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(140, 198, 63, 0.15);
        border: 3px solid #8cc63f;
        text-align: center;
        transition: all 0.3s ease;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .uma-card:hover {
        border-color: #00b5e2;
        box-shadow: 0 12px 20px rgba(0, 181, 226, 0.2);
    }
    .uma-card img {
        width: 100%;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        animation: breathing 3.5s infinite ease-in-out;
        margin-bottom: 10px;
    }
    .uma-name {
        color: #e83984;
        font-size: 1.5rem;
        font-weight: 900;
        margin-top: 15px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .uma-desc {
        color: #4b5563;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    [data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 20px;
        padding: 15px;
        border: 2px solid #8cc63f;
        box-shadow: 0 4px 8px rgba(140, 198, 63, 0.1);
        margin-bottom: 15px;
    }

    h1, h2, h3, h4 {
        color: #8cc63f !important;
        font-weight: 900 !important;
        text-shadow: 1px 1px 0px rgba(0,0,0,0.05);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    "<h1 style='text-align:center;margin-bottom:2rem;font-size:3rem;'>🐎 Uma Musume Reference Library</h1>",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/9/91/Uma_Musume_Pretty_Derby_logo.png/220px-Uma_Musume_Pretty_Derby_logo.png",
        use_container_width=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    menu_selection = st.radio(
        "Điều hướng",
        ["🏠 Trang chủ", "🐎 Thư viện Uma", "🤖 Chatbot AI (RAG)", "⚙️ Cài đặt API"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("✨ Phát triển bởi Master Trainer")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào Trainer! Hôm nay chúng ta sẽ lên kế hoạch huấn luyện cho ai đây? 🐎",
        }
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# TRANG CHỦ
# =============================================================================

if menu_selection == "🏠 Trang chủ":
    st.markdown("### 🌟 Giới thiệu Dự án")
    st.write(
        "Dự án bách khoa toàn thư về thế giới game **Uma Musume Pretty Derby**. "
        "Ứng dụng tích hợp Trợ lý ảo AI thông minh, hỗ trợ Trainer tra cứu thông tin "
        "nhân vật, kỹ năng và tối ưu hóa chỉ số huấn luyện một cách dễ dàng và hiệu quả."
    )
    st.divider()
    st.markdown("### 🏆 Uma Musume nổi bật")

    col1, col2, col3 = st.columns(3)

    with col1:
        img = get_image_src(
            "special_week",
            "https://static.miraheze.org/umamusumeptwiki/thumb/e/e0/Special_Week.png/300px-Special_Week.png",
        )
        st.markdown(
            f"""<div class="uma-card">
                <img src="{img}" alt="Special Week"/>
                <div class="uma-name">Special Week</div>
                <div class="uma-desc">Năng động, lạc quan và luôn nỗ lực hết mình để trở thành Uma Musume số 1 Nhật Bản!</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        img = get_image_src(
            "silence_suzuka",
            "https://static.miraheze.org/umamusumeptwiki/thumb/7/7b/Silence_Suzuka.png/300px-Silence_Suzuka.png",
        )
        st.markdown(
            f"""<div class="uma-card">
                <img src="{img}" alt="Silence Suzuka"/>
                <div class="uma-name">Silence Suzuka</div>
                <div class="uma-desc">Thiên tài tốc độ, lạnh lùng ở vẻ ngoài nhưng luôn hướng tới khung cảnh rực rỡ ở phía trước.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        img = get_image_src(
            "tokai_teio",
            "https://static.miraheze.org/umamusumeptwiki/thumb/1/1a/Tokai_Teio.png/300px-Tokai_Teio.png",
        )
        st.markdown(
            f"""<div class="uma-card">
                <img src="{img}" alt="Tokai Teio"/>
                <div class="uma-name">Tokai Teio</div>
                <div class="uma-desc">Tự tin, kiêu hãnh với những bước chạy nhảy nhẹ nhàng như chiếc lò xo mạnh mẽ.</div>
            </div>""",
            unsafe_allow_html=True,
        )

# =============================================================================
# THƯ VIỆN UMA
# =============================================================================

elif menu_selection == "🐎 Thư viện Uma":
    st.markdown("### 🐎 Cơ sở dữ liệu Uma Musume")
    st.info(
        "Khu vực này đang được xây dựng. "
        "Sắp tới sẽ bao gồm danh sách đầy đủ các nhân vật, "
        "thẻ Hỗ trợ (Support Cards) và từ điển Kỹ năng (Skills)."
    )

# =============================================================================
# CHATBOT AI (RAG)
# =============================================================================

elif menu_selection == "🤖 Chatbot AI (RAG)":
    st.markdown("### 🤖 Trợ lý ảo Uma (RAG Chatbot)")
    st.caption(
        "Hãy hỏi tôi về cách huấn luyện, thông tin nhân vật, hoặc cơ chế game Uma Musume! "
        "Tôi sẽ tìm kiếm trong dữ liệu để trả lời bạn."
    )

    # Gợi ý câu hỏi nhanh
    st.markdown("**💡 Gợi ý:**")
    quick_cols = st.columns(3)
    quick_questions = [
        "Công thức tính HP của uma là gì?",
        "Rice Shower train kỹ năng gì trong MANT?",
        "Silence Suzuka nên chạy cự ly nào?",
    ]
    for i, qq in enumerate(quick_questions):
        if quick_cols[i].button(qq, key=f"quick_{i}"):
            st.session_state.pending_query = qq

    st.divider()

    # Lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        score = src.get("score", 0)
                        st.markdown(f"**[{i}] {source_name}** | score: `{score:.4f}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

    # Input
    user_input = st.chat_input("Nhập câu hỏi về Uma Musume (VD: Chỉ số ưu tiên cho cự ly dài là gì?)")
    query = user_input or st.session_state.pending_query

    if query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang lục tìm tài liệu..."):
                try:
                    from src.task10_generation import generate_with_citation
                    response = generate_with_citation(query, top_k=5)
                    answer = response.get("answer", "Chưa thể trả lời.")
                    sources = response.get("sources", [])
                except Exception as e:
                    answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        score = src.get("score", 0)
                        st.markdown(f"**[{i}] {source_name}** | score: `{score:.4f}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

# =============================================================================
# CÀI ĐẶT API
# =============================================================================

elif menu_selection == "⚙️ Cài đặt API":
    st.markdown("### ⚙️ Cài đặt hệ thống")
    st.write("Cấu hình API Key và trạng thái kết nối Cơ sở dữ liệu RAG.")

    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-...")
    if api_key_input:
        st.success("Đã ghi nhận API Key!")

    st.divider()
    st.markdown("#### Trạng thái ChromaDB")

    try:
        import chromadb
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(name=COLLECTION_NAME)
        count = col.count()
        st.success(f"✅ ChromaDB đang hoạt động — **{count:,} chunks** đã được index.")
    except Exception as e:
        st.error(f"❌ Không kết nối được ChromaDB: {e}")

    st.divider()
    st.markdown("#### Kiến trúc hệ thống")
    st.code(
        "User Query\n"
        "  → Semantic Search (ChromaDB + OpenAI Embedding)\n"
        "  + Lexical Search (BM25)\n"
        "  → RRF Fusion + Jina Reranking\n"
        "  → PageIndex Fallback (nếu score < threshold)\n"
        "  → GPT-4o-mini Generation với Citation\n"
        "  → Câu trả lời + Nguồn tham khảo",
        language="text",
    )
