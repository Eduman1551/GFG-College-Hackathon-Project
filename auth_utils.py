# This handles the "Free Database" and "Security".

# auth_utils.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy .ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

# --- 1. DATABASE SETUP (SQLite) ---
DATABASE_URL = "sqlite:///./users.db"  # This creates a file named users.db
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the User Table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

# --- 2. SECURITY SETUP ---
SECRET_KEY = "HACKATHON_SECRET_KEY"  # Change this to a random string
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60) # Token valid for 1 hour
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()