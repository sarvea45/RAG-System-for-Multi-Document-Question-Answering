# Enterprise RAG System for Multi-Document Q&A

This project provides a robust, context-aware Retrieval-Augmented Generation (RAG) API built with Python, FastAPI, Pinecone, and SQLite. It enables multi-turn, conversational Q&A against proprietary document uploads (PDF/DOCX) while strictly avoiding hallucinations by citing exact document names and page numbers.

## Architectural Overview
The system strictly decouples the data pipelines:
1. **Ingestion Pipeline (`/api/upload`)**: Extracts text and metadata (page numbers) from uploaded files, chunks the text using a sliding window, vectorizes it using OpenAI embeddings, and upserts it to Pinecone.
2. **Query Pipeline (`/api/chat`)**: Embeds the user's query, retrieves the Top-K relevant context chunks from Pinecone, fetches conversational history from SQLite, and generates a structured, verified response via an LLM.

## Chunking Strategy
The ingestion pipeline utilizes a sliding-window text chunking strategy. Text is split into chunks of approximately 500-1000 characters with an overlap of 100-200 characters. This ensures that semantic meaning is not lost across chunk boundaries, preventing context fragmentation while respecting LLM token limits.

## Setup Instructions

### 1. Configuration
Create a `.env` file based on the provided template:
```bash
cp .env.example .env
```
Ensure you fill in your `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX_NAME`.

### 2. Single-Command Setup (Docker)
Start the application and its dependencies using Docker Compose:
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`. You can explore the interactive documentation at `http://localhost:8000/docs`.

## Usage
### Triggering Ingestion
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### Triggering the Query Endpoint
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the PTO policy?", "session_id": "optional-session-id"}'
```
