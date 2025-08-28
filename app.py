import os
import tempfile
import streamlit as st
from embedchain import App

# Embedchain app configured with Groq LLM and FAISS vector DB backend (no ChromaDB, no SQLite dependency)
@st.cache_resource
def get_embedchain_app():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": "mixtral-8x7b-32768",
                    "api_key": api_key,
                    "stream": True,
                },
            },
            "vectordb": {
                "provider": "faiss",  # Use FAISS, no ChromaDB
                "config": {},
            },
        }
        return App.from_config(config=config)
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in the Streamlit Cloud settings.")
        st.stop()

st.title("📄 Document LLM Bot")
st.caption("Upload documents and chat with their content! Powered by Embedchain and Groq.")

with st.sidebar:
    st.markdown("[Get a Groq API Key](https://console.groq.com/keys)")
    st.markdown("[View Source Code](https://github.com/embedchain/embedchain)")
    st.markdown("**Note:** Set `GROQ_API_KEY` in Streamlit Cloud secrets.")

app = get_embedchain_app()

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv", "json", "md", "mdx"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Ingest Documents"):
    with st.spinner("Ingesting documents..."):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                app.add(tmp_path)
                st.success(f"Successfully ingested {uploaded_file.name}")

                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error ingesting {uploaded_file.name}: {str(e)}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload documents and ask me questions about them."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = app.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
