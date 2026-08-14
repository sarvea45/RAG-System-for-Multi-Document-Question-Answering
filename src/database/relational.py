import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from src.core.config import settings

Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)

class SessionModel(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("MessageModel", back_populates="session", order_by="MessageModel.created_at")

class MessageModel(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("SessionModel", back_populates="messages")

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
