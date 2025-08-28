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
st.title("Document Reader RAG Model")

st.markdown("""
Upload your documents (PDF or TXT) and ask questions about their content.
The system uses FAISS for vector storage and Groq for generation.
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
            with st.spinner("Processing documents and building index..."):
                vectorstore = build_vector_store(tuple(uploaded_files))  # Tuple for cache hash
            
            # Set up LLM
            llm = ChatGroq(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")
            
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
            
            # Query input
            query = st.text_input("Ask a question about the documents:")
            
            if query:
                with st.spinner("Generating answer..."):
                    response = rag_chain.invoke({"input": query})
                    st.write("**Answer:**")
                    st.write(response["answer"])
                    st.write("**Retrieved Context:**")
                    for doc in response["context"]:
                        st.write(f"- {doc.page_content[:200]}... (from {doc.metadata.get('source', 'unknown')})")
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    else:
        st.info("Please upload at least one document to get started.")