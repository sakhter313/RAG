import os
import tempfile
import streamlit as st
from embedchain import App

# Function to get Embedchain app with Groq config and FAISS backend (no SQLite errors)
@st.cache_resource
def get_embedchain_app():
    try:
        api_key = st.secrets["GROQ_API_KEY"]  # Access Groq API key from Streamlit secrets
        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": "mixtral-8x7b-32768",  # Default Groq model; can change to 'llama3-70b-8192', etc.
                    "api_key": api_key,
                    "stream": True  # Enable streaming for responses
                }
            },
            "vectordb": {
                "provider": "faiss",  # Use FAISS instead of ChromaDB
                "config": {}          # Default in-memory, or can pass path if needed
            }
        }
        return App.from_config(config=config)
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please configure it in the Streamlit Cloud dashboard.")
        st.stop()

# Streamlit app layout
st.title("📄 Document LLM Bot")
st.caption("Upload documents and chat with their content! Powered by Embedchain and Groq.")

# Sidebar with info
with st.sidebar:
    st.markdown("[Get a Groq API Key](https://console.groq.com/keys)")
    st.markdown("[View Source Code](https://github.com/embedchain/embedchain) (inspired by Embedchain examples)")
    st.markdown("**Note**: Ensure your Groq API key is set in Streamlit Cloud secrets under `GROQ_API_KEY`.")

# Initialize app
app = get_embedchain_app()

# File uploader for documents
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv", "json", "md", "mdx"],
    accept_multiple_files=True
)

# Button to ingest files
if uploaded_files and st.button("Ingest Documents"):
    with st.spinner("Ingesting documents..."):
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Add to Embedchain (auto-detects data_type based on file extension/path)
                app.add(tmp_path)
                st.success(f"Successfully ingested {uploaded_file.name}")

                # Clean up temp file
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error ingesting {uploaded_file.name}: {str(e)}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload documents and ask me questions about them."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = app.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
