from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.upload import router as upload_router
from src.api.chat import router as chat_router
from src.database.relational import init_db

# Initialize relational DB tables
init_db()

app = FastAPI(
    title="Enterprise RAG System API",
    description="Multi-Document Q&A with Vector Database and graceful failure guardrails."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api", tags=["Ingestion"])
app.include_router(chat_router, prefix="/api", tags=["Query"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
