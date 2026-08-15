# Enterprise RAG System for Multi-Document Q&A

This project provides a robust, context-aware Retrieval-Augmented Generation (RAG) API built with Python, FastAPI, Pinecone, and SQLite. 

It is engineered with a **100% Free, Local-First AI Stack**: swapping expensive OpenAI APIs for lightweight local embeddings (`BAAI/bge-small-en-v1.5`) and blazing fast LLM inference (`llama-3.1-8b` via Groq).

It enables multi-turn, conversational Q&A against proprietary document uploads (PDF/DOCX) while strictly avoiding hallucinations by citing exact document names and page numbers.

---

## Architectural Overview
The system strictly decouples the data pipelines:

1. **Ingestion Pipeline (`/api/upload`)**: 
   - Accepts multiple PDF/DOCX files simultaneously.
   - Extracts text and metadata (page numbers) using PyMuPDF.
   - Chunks the text using a sliding window strategy (1000 chars, 200 overlap).
   - Vectorizes text locally on the CPU using `BAAI/bge-small-en-v1.5`.
   - Upserts vectors and structural metadata into Pinecone Serverless.

2. **Query Pipeline (`/api/chat`)**: 
   - Embeds the user's query locally.
   - Retrieves the Top-15 relevant context chunks from Pinecone.
   - Evaluates similarity thresholds to filter out noisy vectors.
   - Fetches conversational history from the SQLite relational DB for pronoun resolution.
   - Generates a structured, verified JSON response containing the answer and citations via Llama 3.1.
   - Features a strict **Anti-Hallucination Guardrail**: If context is missing, the system deterministically replies: *"I could not find an answer in the provided documents."*

---

## Setup Instructions

### 1. Configuration
Create a `.env` file based on the provided template:
```bash
cp .env.example .env
```
Ensure you fill in your `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX_NAME`.

### 2. Single-Command Setup (Docker)
Start the application and its dependencies using Docker Compose. The `Dockerfile` handles everything, including downloading the PyTorch CPU wheels and the BGE embedding model weights.
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`. 
You can interactively test the endpoints at the Swagger UI: `http://localhost:8000/docs`.

---

## Usage Example (cURL)

### Triggering Ingestion (Multi-File)
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/handbook.pdf" \
  -F "files=@/path/to/syllabus.pdf"
```

### Triggering the Query Endpoint (With Memory)
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What are the core units covered in the syllabus?", 
        "session_id": "user-session-123"
      }'
```
