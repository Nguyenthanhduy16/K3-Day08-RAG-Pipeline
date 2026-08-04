import streamlit as st
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RAG Chatbot Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot Demo Lab")
st.caption("Ứng dụng hỏi đáp AI với dữ liệu nội bộ (có trích dẫn nguồn)")

# ==========================================
# CHROMA DB INIT
# ==========================================
@st.cache_resource
def get_chroma_collection():
    try:
        if os.path.exists('./chroma_db'):
            client = chromadb.PersistentClient(path='./chroma_db')
            collections = client.list_collections()
            if collections:
                # Lấy collection đầu tiên
                col_name = collections[0].name if hasattr(collections[0], 'name') else collections[0]
                return client.get_collection(col_name)
    except Exception as e:
        st.error(f"Lỗi khởi tạo ChromaDB: {e}")
    return None

collection = get_chroma_collection()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Nhập API Key hoặc để trống nếu đã có trong file .env")
    openai_api_key = api_key_input if api_key_input else os.getenv("OPENAI_API_KEY")
    
    st.divider()
    st.subheader("Trạng thái kết nối")
    if collection:
        st.success(f"✅ Đã kết nối ChromaDB")
    else:
        st.warning("⚠️ Không tìm thấy ChromaDB")

# ==========================================
# CHATBOT STATE & UI
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là AI trợ lý. Bạn có câu hỏi gì cần tôi giải đáp dựa trên dữ liệu lab không?"}
    ]

# Render lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Nếu có thông tin nguồn, hiển thị ra
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Nguồn trích dẫn"):
                for cit in msg["citations"]:
                    st.markdown(f"- **{cit['source']}**\\n  - *Trích đoạn*: {cit['content'][:150]}...")

# ==========================================
# XỬ LÝ INPUT NGƯỜI DÙNG
# ==========================================
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # 1. Thêm câu hỏi của user vào giao diện
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 2. Sinh câu trả lời của Assistant
    with st.chat_message("assistant"):
        if not openai_api_key:
            error_msg = "⚠️ Vui lòng nhập OpenAI API Key ở thanh bên để tiếp tục."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Đang tìm kiếm thông tin..."):
                try:
                    context = ""
                    citations = []
                    
                    # Truy xuất thông tin từ ChromaDB
                    if collection:
                        results = collection.query(
                            query_texts=[user_input],
                            n_results=3
                        )
                        
                        if results and 'documents' in results and results['documents']:
                            retrieved_docs = results['documents'][0]
                            retrieved_metadatas = results['metadatas'][0] if 'metadatas' in results else []
                            
                            for idx, doc in enumerate(retrieved_docs):
                                metadata = retrieved_metadatas[idx] if idx < len(retrieved_metadatas) else {}
                                source = metadata.get('source', metadata.get('filename', f'Tài liệu {idx+1}'))
                                
                                context += f"[{source}]\\n{doc}\\n\\n"
                                citations.append({
                                    "source": source,
                                    "content": doc
                                })
                                
                    # Khởi tạo OpenAI Client
                    client = OpenAI(api_key=openai_api_key)
                    
                    prompt = f"""Bạn là trợ lý AI. Dựa vào thông tin BỐI CẢNH dưới đây, hãy trả lời câu hỏi của người dùng.
Trong câu trả lời, hãy trích dẫn ngắn gọn nguồn tài liệu (ví dụ: "[Tên_Tài_liệu]").
Nếu bối cảnh không chứa thông tin, hãy nói rõ là "Tôi không tìm thấy thông tin trong tài liệu".

BỐI CẢNH TỪ TÀI LIỆU:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {user_input}
"""
                    
                    # Gọi API
                    stream = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Bạn là AI trợ lý trả lời câu hỏi dựa trên tài liệu cung cấp. Luôn sử dụng tiếng Việt."},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                    )
                    
                    # Stream kết quả
                    response = st.write_stream(stream)
                    
                    # Hiển thị nguồn trực tiếp dưới câu trả lời
                    if citations:
                        with st.expander("📚 Nguồn trích dẫn"):
                            for cit in citations:
                                st.markdown(f"- **{cit['source']}**\\n  - *Trích đoạn*: {cit['content'][:150]}...")
                                
                    # Lưu vào lịch sử chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "citations": citations
                    })
                    
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Lỗi: {e}"})

