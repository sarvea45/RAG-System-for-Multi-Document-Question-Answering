from typing import List
from sentence_transformers import SentenceTransformer

# Load the local embedding model (downloads automatically on first run)
# 'all-MiniLM-L6-v2' is highly efficient and maps to 384 dimensions.
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts using a local open-source model.
    """
    if not texts:
        return []
        
    embeddings = model.encode(texts)
    return embeddings.tolist()
