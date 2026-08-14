from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import logging

from src.database.relational import get_db, SessionModel, MessageModel
from src.services.embedding import get_embeddings
from src.database.vector_store import retrieve_context
from src.services.llm import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    session_id: str = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # 1. Fetch conversational history (last 5 messages)
        chat_history = []
        db_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if db_session:
            history_objs = db.query(MessageModel).filter(
                MessageModel.session_id == session_id
            ).order_by(MessageModel.created_at.desc()).limit(5).all()
            
            # Reverse to chronological
            history_objs = reversed(history_objs)
            chat_history = [{"role": msg.role, "content": msg.content} for msg in history_objs]
        else:
            # Create new session if it doesn't exist
            new_session = SessionModel(id=session_id)
            db.add(new_session)
            db.commit()

        # 2. Embed user query
        query_embedding = get_embeddings([request.query])[0]
        
        # 3. Retrieve Context (with thresholding)
        retrieved_chunks = retrieve_context(query_embedding, top_k=15)
        
        # 4. Graceful Failure Check
        if not retrieved_chunks:
            answer_payload = {
                "answer": "I could not find an answer in the provided documents.",
                "citations": []
            }
        else:
            # 5. Generate Answer via LLM
            answer_payload = generate_answer(request.query, retrieved_chunks, chat_history)
            
        # 6. Save Interaction to DB
        user_msg = MessageModel(session_id=session_id, role="user", content=request.query)
        asst_msg = MessageModel(session_id=session_id, role="assistant", content=answer_payload.get("answer", ""))
        db.add(user_msg)
        db.add(asst_msg)
        db.commit()
        
        # 7. Return Final Payload
        answer_payload["session_id"] = session_id
        return answer_payload
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
