
import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Inject custom CSS for background and styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to bottom right, #e0f7fa, #ffffff);
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("📖 About the App")
    st.markdown("""
    This is an advanced Document Reader using RAG (Retrieval-Augmented Generation).
    
    - Upload PDF or TXT files.
    - The app builds a FAISS vector index.
    - Ask questions and get answers powered by Groq AI.
    
    **Supported Models:** Choose from the latest Groq models for optimal performance.
    """)
    # Model selection
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
    selected_model = st.selectbox("Select Groq Model", models, index=1)  # Default to llama-3.3-70b-versatile

# Function to process uploaded files and build vector store
@st.cache_resource
def build_vector_store(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue
            docs.extend(loader.load())
        finally:
            os.unlink(tmp_path)
    
    if not docs:
        raise ValueError("No valid documents uploaded.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Main Streamlit app
st.title("📚 Document Reader RAG Model")
st.markdown("""
Upload your documents (PDF or TXT) and ask questions about their content.  
The system uses FAISS for vector storage and Groq for intelligent generation.
""")

# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# API Key from secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your app settings.")
else:
    groq_api_key = st.secrets["GROQ_API_KEY"]

    if uploaded_files:
        try:
            with st.spinner("Processing documents and building index... This may take a moment."):
                vectorstore = build_vector_store(tuple(uploaded_files))  # Tuple for cache hash
            
            st.success("Index built successfully! Now you can ask questions.")
            
            # Set up LLM with selected model
            llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)
            
            # Prompt template
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            
            # Chains
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 5}), question_answer_chain)
            
            # Query input - displayed after index is built
            query = st.text_input("Ask a question about the documents:", placeholder="Type your question here...")
            
            if query:
                with st.spinner("Generating answer..."):
                    response = rag_chain.invoke({"input": query})
                
                # Display answer in a nice box
                st.markdown("### 💡 Answer")
                st.info(response["answer"])
                
                # Expander for retrieved context
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(response["context"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content)
                        st.write(f"*Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')}*")
                        st.divider()
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    else:
        st.info("Please upload at least one document to get started.")