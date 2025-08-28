import os
import tempfile
import streamlit as st
from embedchain import App

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
                "provider": "faiss",  # Use FAISS backend to avoid SQLite dependency issues
                "config": {},
            },
        }
        return App.from_config(config=config)
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please set it in the Streamlit Cloud dashboard.")
        st.stop()

st.title("📄 Document Reader with Embedchain")
st.caption("Upload your documents and ask questions about their content.")

with st.sidebar:
    st.markdown("[Get a Groq API Key](https://console.groq.com/keys)")
    st.markdown("Set `GROQ_API_KEY` in Streamlit secrets to use this app.")

app = get_embedchain_app()

uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt", "csv", "json", "md", "mdx"], accept_multiple_files=True)

if uploaded_files and st.button("Ingest Documents"):
    with st.spinner("Ingesting..."):
        for f in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                app.add(tmp_path)
                st.success(f"Ingested {f.name}")
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Failed to ingest {f.name}: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload documents and ask me anything about them."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = app.query(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")
