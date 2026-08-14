import pytest
from fastapi.testclient import TestClient
from src.main import app
import os
import io

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_hallucination_graceful_failure():
    """
    Simulates a query for which there are no uploaded documents,
    ensuring the API returns the fallback response.
    """
    # NOTE: This requires OPENAI_API_KEY and PINECONE_API_KEY to be set 
    # in the test environment to actually run against the embeddings API.
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Skipping test because OPENAI_API_KEY is not set.")
        
    payload = {
        "query": "What is the capital of France?"
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["answer"] == "I could not find an answer in the provided documents."
    assert len(data["citations"]) == 0
