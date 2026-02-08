# =====================================================
# Intelligent Academic Second Brain - FINAL VERSION
# Includes: Auth, RAG, Mind Maps (Safe Mode), & Graph View
# =====================================================

import os
import re
import shutil
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

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

SECRET_KEY = "HACKATHON_SECRET_KEY_CHANGE_ME"  # Change for production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

OLLAMA_MODEL = "mistral"  # Ensure you have run: ollama pull mistral
BASE_DATA_PATH = "data"
UPLOAD_PATH = "uploads"
VECTORSTORE_PATH = "vectorstore"

# Create directories if they don't exist
os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# =====================================================
# OCR CONFIGURATION
# =====================================================
# Update this path if Tesseract is installed elsewhere
# On Mac/Linux, you usually don't need this line if it's in PATH
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
# RAG / AI SETUP
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
                    print(f"PDF {file} seems empty/scanned. Skipping OCR for now.")
                else:
                    for d in docs:
                        d.metadata["source"] = file
                    documents.extend(docs)
            except Exception as e:
                print(f"Error loading PDF {file}: {e}")

        # 2. Handle Images (JPG, PNG)
        elif file.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processing image: {file}")
            extracted_text = extract_text_from_image(file_path)
            
            if extracted_text.strip():
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL origins (perfect for hackathons)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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

    # Trigger ingestion
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
    current_user: User = Depends(get_current_user)
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

# --- 5. Generate Mind Map (CONNECTED FLOW FIX) ---
@app.post("/generate_mindmap")
async def generate_mindmap(
    payload: MindMapRequest,
    current_user: User = Depends(get_current_user)
):
    topic = payload.topic

    # 1. Retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(topic)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. ASK FOR A LIST
    prompt = f"""
    Based on the context below, break down the topic '{topic}' into a flowchart.
    
    Output a LIST of connections using '->'.
    Format: Parent Concept -> Child Concept
    
    Example:
    Sun -> Plants
    Plants -> Animals
    Animals -> Decomposers
    
    RULES:
    1. One connection per line.
    2. Do NOT number the lines.
    3. Keep text short (max 4 words).
    4. Reuse the exact same terms to connect steps.
    
    Context:
    {context}
    """

    # 3. Get raw text response
    raw_response = llm.invoke(prompt)
    
    # 4. PYTHON BUILDS THE CODE
    mermaid_lines = ["graph TD"]
    node_registry = {} 
    node_counter = 1
    
    # Process line by line
    for line in raw_response.split('\n'):
        # 4a. STRIP NUMBERING (Crucial Fix)
        # Removes "1. ", "2. ", "- " from start of lines
        clean_line = re.sub(r'^[\d-]+\.\s*', '', line.strip())
        
        if "->" in clean_line:
            parts = clean_line.split("->")
            
            for i in range(len(parts) - 1):
                src_text = parts[i].strip().replace('"', '').replace("'", "")
                tgt_text = parts[i+1].strip().replace('"', '').replace("'", "")
                
                if not src_text or not tgt_text:
                    continue

                # Assign ID to Source
                if src_text not in node_registry:
                    node_registry[src_text] = f"node{node_counter}"
                    node_counter += 1
                
                # Assign ID to Target
                if tgt_text not in node_registry:
                    node_registry[tgt_text] = f"node{node_counter}"
                    node_counter += 1
                
                src_id = node_registry[src_text]
                tgt_id = node_registry[tgt_text]
                
                mermaid_lines.append(f'    {src_id}["{src_text}"] --> {tgt_id}["{tgt_text}"]')

    final_code = "\n".join(mermaid_lines)
    
    if len(mermaid_lines) == 1:
        final_code = f'graph TD\n    node1["Topic: {topic}"] --> node2["No clear flow found"]'

    return {
        "topic": topic,
        "mermaid_code": final_code
    }

# --- 6. List Files Endpoint ---
@app.get("/files")
async def list_files(current_user: User = Depends(get_current_user)):
    """Returns a list of all uploaded files."""
    if not os.path.exists(UPLOAD_PATH):
        return {"files": []}
    
    files = os.listdir(UPLOAD_PATH)
    return {"files": files}

# --- 7. Graph Data Endpoint ---
@app.get("/graph_data")
async def get_graph_data(current_user: User = Depends(get_current_user)):
    """Generates nodes and links for the Force Graph."""
    
    nodes = []
    links = []
    
    # 1. Add the "Central User" node
    nodes.append({"id": "User", "group": 1})
    
    # 2. Add File Nodes
    if os.path.exists(UPLOAD_PATH):
        files = os.listdir(UPLOAD_PATH)
        for filename in files:
            # Create a node for the file
            # Group 2 = PDFs, Group 3 = Images (Just for visual variety)
            group = 2 if filename.endswith(".pdf") else 3
            nodes.append({"id": filename, "group": group})
            
            # Create a link from User to File (Star Topology)
            links.append({"source": "User", "target": filename})
            
    return {"nodes": nodes, "links": links}

# Run with: uvicorn main:app --reload