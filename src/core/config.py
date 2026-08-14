from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    
    groq_api_key: str
    
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1" # Default or fallback
    pinecone_index_name: str
    
    database_url: str = "sqlite:///./rag_history.db"

    class Config:
        env_file = ".env"

settings = Settings()
