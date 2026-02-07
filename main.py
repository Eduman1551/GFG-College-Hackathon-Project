# =====================================================
# Intelligent Academic Second Brain - FULL VERSION
# Includes: Auth (Login/Register), SQLite DB, & RAG
# =====================================================

import os
import shutil
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from PIL import Image
import pytesseract
from langchain_core.documents import Document

# --- Database & Auth Imports ---
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- AI & LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# =====================================================
# CONFIGURATION
# =====================================================


SECRET_KEY = "HACKATHON_SECRET_KEY_CHANGE_ME"  # <--- Change this for production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

OLLAMA_MODEL = "mistral"
BASE_DATA_PATH = "data"
UPLOAD_PATH = "uploads"
VECTORSTORE_PATH = "vectorstore"

os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)
# =====================================================
# OCR CONFIGURATION
# =====================================================
# Update this path if you installed Tesseract somewhere else
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================
# DATABASE SETUP (SQLite)
# =====================================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- User Table Model ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Create Tables
Base.metadata.create_all(bind=engine)

# =====================================================
# SECURITY & AUTH UTILS
# =====================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get Current User (Protects Endpoints)
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# =====================================================
# RAG / AI SETUP (Original Logic)
# =====================================================
llm = Ollama(model=OLLAMA_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def load_vectorstore():
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return FAISS.from_texts(["Initial empty index"], embeddings)

vectorstore = load_vectorstore()

def extract_text_from_image(image_path: str):
    """
    Uses Tesseract OCR to extract text from an image file.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""
    
def ingest_files_from_folder(folder_path: str):
    global vectorstore
    documents = []
    
    if not os.path.exists(folder_path):
        return

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # 1. Handle PDFs
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            try:
                docs = loader.load()
                # Check if PDF text is empty (scanned PDF)
                if not docs or len(docs[0].page_content) < 10:
                    print(f"PDF {file} seems empty/scanned. Skipping OCR for now (requires extra libraries).")
                    # Note: Full PDF OCR requires 'pdf2image' + 'poppler', which is complex to install.
                    # For a hackathon, often easier to convert PDF to JPG manually or use a paid API.
                else:
                    for d in docs:
                        d.metadata["source"] = file
                    documents.extend(docs)
            except Exception as e:
                print(f"Error loading PDF {file}: {e}")

        # 2. Handle Images (JPG, PNG) - NEW FEATURE
        elif file.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processing image: {file}")
            extracted_text = extract_text_from_image(file_path)
            
            if extracted_text.strip():
                # Create a LangChain Document manually
                doc = Document(
                    page_content=extracted_text,
                    metadata={"source": file}
                )
                documents.append(doc)

    if documents:
        chunks = text_splitter.split_documents(documents)
        vectorstore.add_documents(chunks)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"Ingested {len(documents)} new documents.")

# Initial Load
ingest_files_from_folder(BASE_DATA_PATH)

# =====================================================
# PYDANTIC MODELS (Schemas)
# =====================================================
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class QuestionRequest(BaseModel):
    question: str

class MindMapRequest(BaseModel):
    topic: str

# =====================================================
# API ENDPOINTS
# =====================================================
app = FastAPI(title="Academic Second Brain API")

# --- 1. Register User ---
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"User {user.username} created successfully"}

# --- 2. Login (Generate Token) ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- 3. Upload File (PDF or Image) ---
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    # Save the file
    save_path = os.path.join(UPLOAD_PATH, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger ingestion (now supports images!)
    ingest_files_from_folder(UPLOAD_PATH)

    return {
        "status": "success",
        "message": f"File {file.filename} uploaded and processed by {current_user.username}",
        "filename": file.filename
    }

# --- 4. Ask Question (PROTECTED) ---
@app.post("/ask")
async def ask_question(
    payload: QuestionRequest,
    current_user: User = Depends(get_current_user) # <--- Forces Login
):
    question = payload.question

    # Retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    # Generate Prompt
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
        "sources": sources,
        "user": current_user.username
    }

# --- 5. Generate Mind Map (PROTECTED) ---
@app.post("/generate_mindmap")
async def generate_mindmap(
    payload: MindMapRequest,
    current_user: User = Depends(get_current_user)
):
    topic = payload.topic

    # 1. Retrieve relevant content
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(topic)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. The Strict Prompt for Mermaid.js
    prompt = f"""
    You are an expert at creating mind maps.
    Based ONLY on the context below, generate a Mermaid.js mindmap code for the topic: '{topic}'.
    
    Rules:
    1. Start with 'graph TD' (Top-Down).
    2. Use square brackets for nodes: A[Main Topic] --> B[Subtopic]
    3. Do NOT use special characters or quotes inside the brackets that might break syntax.
    4. Return ONLY the code. No markdown backticks (```), no explanations.
    
    Context:
    {context}
    """

    # 3. Get the code from LLM
    mindmap_code = llm.invoke(prompt)
    
    # Clean up common LLM mistakes (optional but recommended)
    mindmap_code = mindmap_code.replace("```mermaid", "").replace("```", "").strip()

    return {
        "topic": topic,
        "mermaid_code": mindmap_code
    }