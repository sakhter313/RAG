import sys

# Patch for ChromaDB's SQLite requirement on Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import streamlit as st
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores.faiss import FAISS
    logger.info("Successfully imported langchain_community.vectorstores.faiss")
except ImportError as e:
    logger.error(f"Failed to import langchain_community.vectorstores.faiss: {e}")
    st.error(f"Import error: Failed to load langchain_community.vectorstores.faiss. Check logs for details. Error: {e}")
    st.stop()

from embedchain import App

# Ensure temporary directory for FAISS index
os.environ["EMBEDCHAIN_DB_DIR"] = "/tmp/embedchain_db"
os.environ["EMBEDCHAIN_VECTOR_DB"] = "faiss"  # Force FAISS

@st.cache_resource
def get_embedchain_app():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        # Flatten the config into keyword arguments
        config = {
            "llm_provider": "groq",
            "llm_config": {
                "model": "mixtral-8x7b-32768",
                "api_key": api_key,
                "stream": True,
            },
            "vectordb_provider": "faiss",
            "vectordb_config": {
                "index_dir": "/tmp/embedchain_db",
                "dimension": 1536,
            },
            "embedder_provider": "openai",
            "embedder_config": {
                "model": "text-embedding-ada-002",
            },
        }
        logger.info("Initializing Embedchain app with FAISS backend")
        app = App(**config)  # Pass as keyword arguments
        if not isinstance(app.vectordb, FAISS):
            raise RuntimeError("FAISS vector database not initialized correctly. Check configuration.")
        return app
    except KeyError as e:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please set it in the Streamlit Cloud dashboard.")
        logger.error(f"KeyError: {e}")
        st.stop()
    except TypeError as e:
        st.error(f"Failed to initialize app: {e}. Check config parameters.")
        logger.error(f"Initialization error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize app: {e}")
        logger.error(f"Initialization error: {e}")
        st.stop()

st.title("📄 Document Reader with Embedchain")
st.caption("Upload your documents and ask questions about their content.")

with st.sidebar:
    st.markdown("[Get a Groq API Key](https://console.groq.com/keys)")
    st.markdown("Set `GROQ_API_KEY` in Streamlit secrets to use this app.")

app = get_embedchain_app()

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv", "json", "md", "mdx"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Ingest Documents"):
    with st.spinner("Ingesting..."):
        for f in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(
                    dir="/tmp", delete=False, suffix=f".{f.name.split('.')[-1]}"
                ) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                logger.info(f"Ingesting file: {f.name}")
                app.add(tmp_path)
                st.success(f"Ingested {f.name}")
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Failed to ingest {f.name}: {e}")
                logger.error(f"Ingestion error for {f.name}: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload documents and ask me anything about them."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                logger.info(f"Processing query: {prompt}")
                answer = app.query(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error processing query: {e}")
                logger.error(f"Query error: {e}")