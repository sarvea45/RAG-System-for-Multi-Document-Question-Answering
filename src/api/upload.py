from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import uuid
import logging

from src.database.relational import get_db, DocumentModel
from src.services.document import parse_document
from src.services.chunking import chunk_document
from src.services.embedding import get_embeddings
from src.database.vector_store import upsert_vectors

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    Ingestion Pipeline: Extracts text, chunks it, generates embeddings, 
    and upserts to Pinecone for multiple documents simultaneously.
    """
    results = []
    
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            results.append({"filename": file.filename, "error": "Only .pdf and .docx files are supported."})
            continue

        file_bytes = await file.read()
        
        try:
            # 1. Register Document in DB
            doc_id = str(uuid.uuid4())
            new_doc = DocumentModel(id=doc_id, filename=file.filename)
            db.add(new_doc)
            db.commit()
            
            # 2. Extract Text & Metadata
            pages_data = parse_document(file_bytes, file.filename)
            if not pages_data:
                results.append({"filename": file.filename, "error": "Could not extract text from document."})
                continue
                
            # 3. Text Chunking
            chunks = chunk_document(pages_data, doc_id, file.filename)
            
            # 4. Generate Embeddings (batch)
            texts_to_embed = [c["text"] for c in chunks]
            
            # Process embeddings in batches to avoid rate limits
            batch_size = 50
            all_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i+batch_size]
                embeddings = get_embeddings(batch_texts)
                all_embeddings.extend(embeddings)
                
            # 5. Upsert to Vector Database
            upsert_vectors(chunks, all_embeddings)
            
            results.append({
                "filename": file.filename, 
                "status": "Ingested successfully", 
                "document_id": doc_id, 
                "chunks_processed": len(chunks)
            })
            
        except Exception as e:
            logger.error(f"Error ingesting document {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})
            
    return {"results": results}
