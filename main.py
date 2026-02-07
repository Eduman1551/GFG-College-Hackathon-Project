# =====================================================
# Intelligent Academic Second Brain
# =====================================================

#added new commit

import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import streamlit as st

st.title("My App")
# =====================================================
# PLACEHOLDER 1: Ollama model name
# Change ONLY if you pulled a different model
# =====================================================
OLLAMA_MODEL = "mistral"   # <-- PLACEHOLDER

# =====================================================
# Folder paths
# =====================================================
BASE_DATA_PATH = "data"        # Preloaded PDFs
UPLOAD_PATH = "uploads"        # User uploaded PDFs
VECTORSTORE_PATH = "vectorstore"

os.makedirs(BASE_DATA_PATH, exist_ok=True)  
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# =====================================================
# Initialize LLM, embeddings, splitter
# =====================================================
llm = Ollama(model=OLLAMA_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# =====================================================
# Load or create vector store
# =====================================================
def load_vectorstore():
    # Check if the specific index file exists, not just the folder
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    # If the file is missing, start with an empty store
    print("Creating new empty vectorstore...")
    # FAISS requires at least one text to initialize, so we give it a dummy one
    # We will overwrite or add to this later
    return FAISS.from_texts(["Initial empty index"], embeddings)

vectorstore = load_vectorstore()

# =====================================================
# PDF ingestion logic
# =====================================================
def ingest_pdfs_from_folder(folder_path: str):
    global vectorstore
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file

            documents.extend(docs)

    if documents:
        chunks = text_splitter.split_documents(documents)
        vectorstore.add_documents(chunks)
        vectorstore.save_local(VECTORSTORE_PATH)

# =====================================================
# Initial ingestion (base data)
# =====================================================
ingest_pdfs_from_folder(BASE_DATA_PATH)

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="Academic Second Brain API")

# =====================================================
# Request schema for asking questions
# =====================================================
class QuestionRequest(BaseModel):
    question: str

# =====================================================
# Upload PDF endpoint
# =====================================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_PATH, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_pdfs_from_folder(UPLOAD_PATH)

    return {
        "status": "success",
        "message": f"{file.filename} uploaded and indexed"
    }

# =====================================================
# Question answering endpoint
# =====================================================
@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    question = payload.question

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using ONLY the context below.
    Cite sources at the end.

    Context:
    {context}

    Question:
    {question}
    """

    answer = llm.invoke(prompt)

    sources = list(set(
        d.metadata.get("source", "Unknown") for d in docs
    ))

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
