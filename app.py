import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from prompt import handle_user_query  # <-- move agent logic into chatbot.py

st.set_page_config(
    page_title="Xavier AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("⚙️ Settings & Uploads")
st.sidebar.markdown("Upload documents or images to enrich responses.")

uploaded_pdf = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])
uploaded_docx = st.sidebar.file_uploader("📝 Upload DOCX", type=["docx"])
uploaded_excel = st.sidebar.file_uploader("📊 Upload Excel", type=["xlsx", "xls"])
uploaded_img = st.sidebar.file_uploader("🖼️ Upload an Image", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("---")
st.sidebar.caption("Powered by LangChain + Streamlit 🚀")

st.title("🤖 Xavier AI — Chat with Docs & Images")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello 👋 I can answer from your docs, search the web, or analyze images.")
    ]

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(f"<div style='background:#131f2b;padding:10px;border-radius:10px;'>{msg.content}</div>", unsafe_allow_html=True)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("Human", avatar="🧑"):
                st.markdown(f"<div style='background:#131f2b;padding:10px;border-radius:10px;text-align:right;'>{msg.content}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("Human", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = handle_user_query(user_input)  # your chatbot logic
            st.markdown(f"<div style='background:#131f2b;padding:10px;border-radius:10px;'>{response}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append(AIMessage(content=response))
