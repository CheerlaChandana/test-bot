import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
import os
import tempfile
import shutil
import hashlib
from pathlib import Path
from utils import clean_text
import re
import json

@st.cache_resource(show_spinner=False)
def load_and_process_documents(uploaded_files, groq_api_key, dirs):
    with st.spinner("Processing documents..."):
        all_chunks = []
        tmp_dir = tempfile.mkdtemp(dir=dirs["temp_dir"])
        
        try:
            # Initialize session state if not present
            if "processed_docs" not in st.session_state:
                st.session_state.processed_docs = {}
            if "last_file_hashes" not in st.session_state:
                st.session_state.last_file_hashes = []

            file_hashes = [hashlib.md5(file.getbuffer()).hexdigest() for file in uploaded_files]
            file_names = [file.name for file in uploaded_files]

            for file in uploaded_files:
                filepath = os.path.join(dirs["uploads_dir"], file.name)
                file_hash = hashlib.md5(file.getbuffer()).hexdigest()
                if file_hash not in st.session_state.processed_docs:
                    with open(filepath, "wb") as f:
                        f.write(file.getbuffer())
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(filepath)
                    elif file.name.endswith(".txt"):
                        loader = TextLoader(filepath)
                    elif file.name.endswith(".csv"):
                        loader = CSVLoader(filepath)
                    else:
                        continue
                    docs = loader.load()
                    for doc in docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata['source'] = file.name  # Ensure source is in metadata
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    st.session_state.processed_docs[file_hash] = {
                        "chunks": chunks,
                        "file_name": file.name
                    }
                all_chunks.extend(st.session_state.processed_docs[file_hash]["chunks"])

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            faiss_index_path = dirs["cache_dir"] / "faiss_index"
            cache_valid = os.path.exists(faiss_index_path) and st.session_state.get("last_file_hashes") == file_hashes
            
            if cache_valid:
                faiss_store = FAISS.load_local(str(faiss_index_path), embeddings, allow_dangerous_deserialization=True)
            else:
                faiss_store = FAISS.from_documents(all_chunks, embeddings)
                faiss_store.save_local(str(faiss_index_path))
                st.session_state.last_file_hashes = file_hashes

            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = 3

            hybrid_retriever = EnsembleRetriever(
                retrievers=[faiss_store.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
                weights=[0.5, 0.5]
            )

            llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=hybrid_retriever,
                memory=memory
            )

            return llm, qa_chain, hybrid_retriever, file_names, all_chunks  # Return chunks for reference

        finally:
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temp dir: {e}")

def refine_question(base_question, llm, selected_files=None, all_file_names=None):
    """
    Refines the user query based on selected files or all files.
    """
    if not all_file_names:
        all_file_names = st.session_state.file_names if hasattr(st.session_state, "file_names") else []

    file_context = ", ".join(selected_files) if selected_files else "all uploaded files: " + ", ".join(all_file_names)
    context_str = f" considering the following files: {file_context}" if file_context else ""

    prompt = f"""
Refine this question into a single, clear, and concise query to be answered in one cohesive response, correcting any spelling mistakes or short forms and ensuring it relates to the documents{context_str}. If no relevant content exists in the documents, indicate that and provide a general answer if possible.

User's question: {base_question}

Return the final refined question or response:
"""

    refined_response = llm.invoke(prompt.strip())
    return refined_response.content.strip()
