import uuid
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Ensure index exists (this might take time on first run, ideally done in a setup script)
def ensure_index():
    if settings.pinecone_index_name not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=384, # all-MiniLM-L6-v2 dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=settings.pinecone_environment
            )
        )
    return pc.Index(settings.pinecone_index_name)

try:
    index = ensure_index()
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    index = None

def upsert_vectors(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """Upserts chunks and their embeddings to Pinecone."""
    if not index:
        raise ValueError("Pinecone index not initialized.")
        
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = str(uuid.uuid4())
        # Attach raw text to metadata
        metadata = chunk["metadata"].copy()
        metadata["text"] = chunk["text"]
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

def retrieve_context(query_embedding: List[float], top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Queries Pinecone and filters out results below the threshold.
    Returns the metadata of the passing chunks.
    """
    if not index:
        raise ValueError("Pinecone index not initialized.")
        
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    valid_results = [match.metadata for match in response.matches]
    return valid_results
