<<<<<<< Updated upstream
"""
RAG Chatbot — University Services (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
=======
import streamlit as st
import time
import os
import base64
>>>>>>> Stashed changes

# [KHU VỰC KẾT NỐI RAG]
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv

# Hàm đọc ảnh local sang base64 để có thể nhúng trực tiếp vào thẻ HTML <img>
def get_image_src(filename, fallback_url):
    # Ưu tiên tìm file ảnh được lưu ở thư mục hiện tại
    for ext in ["png", "jpg", "jpeg", "gif"]:
        path = f"{filename}.{ext}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{b64}"
    # Nếu không tìm thấy file ở máy, dùng URL dự phòng
    return fallback_url

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
<<<<<<< Updated upstream
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về dịch vụ và chính sách đại học (học phí, học bổng, ký túc xá, thư viện)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng Academic Achievement?",
        "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
        "Cách đăng ký học phần qua myRMIT?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học (Học phí, Học bổng, Ký túc xá, Thư viện)")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                    st.text(src.get("content", "")[:300] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

# Xử lý khi bấm nút gợi ý hoặc nhập câu hỏi mới
user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                # TODO (Học viên): Tích hợp hàm sinh câu trả lời từ Task 10
                # Ví dụ:
                # from src.task10_generation import generate_with_citation
                # response = generate_with_citation(query, top_k=top_k)
                # answer = response["answer"]
                # sources = response.get("sources", [])

                # Tạm thời mockup để test UI:
                from src.task10_generation import generate_with_citation
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        score = src.get("score", 0)
                        st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
=======
    page_title="Uma Musume Reference Library",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# [KHU VỰC CSS]
# ==========================================
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#d6ecc5 1.5px, transparent 1.5px);
        background-size: 25px 25px;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 4px solid #8cc63f; 
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebar"] * {
        color: #3e9f21 !important;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 30px !important;
        background: #e83984 !important; 
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(232, 57, 132, 0.3);
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px);
        background: #f64b82 !important;
        box-shadow: 0 8px 12px rgba(232, 57, 132, 0.4);
    }

    @keyframes breathing {
        0% { transform: scale(1) translateY(0px); }
        50% { transform: scale(1.03) translateY(-6px); }
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
        position: relative;
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

# ==========================================
# HEADER BANNER
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem; font-size: 3rem;'>🐎 Uma Musume Reference Library</h1>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/9/91/Uma_Musume_Pretty_Derby_logo.png/220px-Uma_Musume_Pretty_Derby_logo.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu_selection = st.radio(
        "Điều hướng",
        ["🏠 Trang chủ", "🐎 Thư viện Uma", "🤖 Chatbot AI (RAG)", "⚙️ Cài đặt API"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("✨ Phát triển bởi Master Trainer")

# ==========================================
# TABS CONTENT
# ==========================================

if menu_selection == "🏠 Trang chủ":
    st.markdown("### 🌟 Giới thiệu Dự án")
    st.write("Dự án bách khoa toàn thư về thế giới game **Uma Musume Pretty Derby**. Ứng dụng tích hợp Trợ lý ảo AI thông minh, hỗ trợ Trainer tra cứu thông tin nhân vật, kỹ năng và tối ưu hóa chỉ số huấn luyện một cách dễ dàng và hiệu quả.")
    
    st.divider()
    st.markdown("### 🏆 Uma Musume nổi bật (Hoạt họa 2D)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Load ảnh local ưu tiên, nếu không có sẽ dùng URL 3D model chính thức (y hệt ảnh bạn đã upload)
        img_src = get_image_src("special_week", "https://static.miraheze.org/umamusumeptwiki/thumb/e/e0/Special_Week.png/300px-Special_Week.png")
        st.markdown(
            f"""
            <div class="uma-card">
                <img src="{img_src}" alt="Special Week" />
                <div class="uma-name">Special Week</div>
                <div class="uma-desc">Năng động, lạc quan và luôn nỗ lực hết mình để trở thành Uma Musume số 1 Nhật Bản!</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    with col2:
        img_src = get_image_src("silence_suzuka", "https://static.miraheze.org/umamusumeptwiki/thumb/7/7b/Silence_Suzuka.png/300px-Silence_Suzuka.png")
        st.markdown(
            f"""
            <div class="uma-card">
                <img src="{img_src}" alt="Silence Suzuka" />
                <div class="uma-name">Silence Suzuka</div>
                <div class="uma-desc">Thiên tài tốc độ, lạnh lùng ở vẻ ngoài nhưng luôn hướng tới khung cảnh rực rỡ ở phía trước.</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    with col3:
        img_src = get_image_src("tokai_teio", "https://static.miraheze.org/umamusumeptwiki/thumb/1/1a/Tokai_Teio.png/300px-Tokai_Teio.png")
        st.markdown(
            f"""
            <div class="uma-card">
                <img src="{img_src}" alt="Tokai Teio" />
                <div class="uma-name">Tokai Teio</div>
                <div class="uma-desc">Tự tin, kiêu hãnh với những bước chạy nhảy nhẹ nhàng như chiếc lò xo mạnh mẽ.</div>
            </div>
            """, unsafe_allow_html=True
        )

elif menu_selection == "🐎 Thư viện Uma":
    st.markdown("### 🐎 Cơ sở dữ liệu Uma Musume")
    st.info("Khu vực này đang được xây dựng. Sắp tới sẽ bao gồm danh sách đầy đủ các nhân vật, thẻ Hỗ trợ (Support Cards) và từ điển Kỹ năng (Skills).")
    
elif menu_selection == "🤖 Chatbot AI (RAG)":
    st.markdown("### 🤖 Trợ lý ảo Uma (RAG Chatbot)")
    st.caption("Hãy hỏi tôi về cách huấn luyện, thông tin sự kiện, hoặc cốt truyện của Uma Musume! Tôi sẽ tìm kiếm trong dữ liệu để trả lời bạn.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Xin chào Trainer! Hôm nay chúng ta sẽ lên kế hoạch huấn luyện cho ai đây? 🐎"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Nhập câu hỏi của bạn (VD: Chỉ số ưu tiên cho cự ly dài là gì?)")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Đang lục tìm tài liệu..."):
                time.sleep(1.5) 
                mock_response = f"Trainer hỏi về: **{user_input}**. \n\nDựa theo dữ liệu phân tích: Đối với cự ly dài (Long), **Stamina (Thể lực)** là chỉ số sống còn, bên cạnh đó **Speed (Tốc độ)** vẫn là yếu tố quyết định ở đoạn nước rút. Hãy ưu tiên sắp xếp các thẻ Support Card thiên về Stamina nhé! ✨"
                st.markdown(mock_response)
                
            st.session_state.messages.append({"role": "assistant", "content": mock_response})

elif menu_selection == "⚙️ Cài đặt API":
    st.markdown("### ⚙️ Cài đặt hệ thống")
    st.write("Cấu hình API Key và trạng thái kết nối Cơ sở dữ liệu RAG.")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-...")
    if api_key_input:
        st.success("Đã ghi nhận API Key! Hệ thống Chatbot AI (nếu được kết nối) có thể sử dụng key này.")
    st.divider()
    st.markdown("#### Trạng thái ChromaDB")
    st.warning("⚠️ Cơ sở dữ liệu ChromaDB hiện chưa được kết nối trong phiên bản này. Trợ lý AI đang sử dụng chế độ Mock data.")
>>>>>>> Stashed changes
