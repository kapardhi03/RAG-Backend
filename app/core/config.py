import os
import secrets
from typing import List, Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    PROJECT_NAME: str = "RAG-Microservices"
    API_V1_STR: str = "/api"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]  # For production, specify actual origins
    
    # Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 4320
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Default Model Configuration
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo-preview")
    FALLBACK_EMBEDDING_MODEL: str = os.getenv("FALLBACK_EMBEDDING_MODEL", "text-embedding-ada-002")
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo")
    
    # Available Models Configuration
    AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "description": "Latest and most capable embedding model"
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "description": "Efficient and cost-effective embedding model"
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "description": "Legacy embedding model for backward compatibility"
        }
    }
    
    AVAILABLE_LLM_MODELS: Dict[str, Dict[str, Any]] = {
        "gpt-4-turbo-preview": {
            "max_tokens": 128000,
            "context_window": 128000,
            "description": "Most capable GPT-4 model with large context window"
        },
        "gpt-4": {
            "max_tokens": 8192,
            "context_window": 8192,
            "description": "Standard GPT-4 model"
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "context_window": 16385,
            "description": "Fast and cost-effective model"
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "context_window": 16385,
            "description": "Extended context version of GPT-3.5"
        }
    }
    
    # Cohere Configuration for Re-ranking
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "50"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "10"))
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "cloneifyai")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Redis Configuration (for task queue)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Database Configuration
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_microservices")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    # Enhanced Chunking Configuration
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1024"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2048"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "256"))
    
    # Semantic Chunking Configuration
    ENABLE_SEMANTIC_CHUNKING: bool = os.getenv("ENABLE_SEMANTIC_CHUNKING", "true").lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.7"))
    
    # Hybrid Search Configuration
    ENABLE_HYBRID_SEARCH: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    VECTOR_SEARCH_WEIGHT: float = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))
    KEYWORD_SEARCH_WEIGHT: float = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
    
    # Query Expansion Configuration
    ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    MAX_EXPANDED_TERMS: int = int(os.getenv("MAX_EXPANDED_TERMS", "5"))
    
    # Advanced Web Scraping Configuration
    ENABLE_ADVANCED_SCRAPING: bool = os.getenv("ENABLE_ADVANCED_SCRAPING", "true").lower() == "true"
    SCRAPING_TIMEOUT: int = int(os.getenv("SCRAPING_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    USE_PROXY: bool = os.getenv("USE_PROXY", "false").lower() == "true"
    PROXY_LIST: List[str] = os.getenv("PROXY_LIST", "").split(",") if os.getenv("PROXY_LIST") else []
    
    # Worker Configuration
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Performance Configuration
    RETRIEVAL_TIMEOUT: int = int(os.getenv("RETRIEVAL_TIMEOUT", "30"))
    GENERATION_TIMEOUT: int = int(os.getenv("GENERATION_TIMEOUT", "60"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            if field_name == "JWT_SECRET_KEY" and raw_val == "your_secret_key_here":
                return secrets.token_hex(32)
            return raw_val

settings = Settings()

# Validate required settings
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")

if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials must be set in environment variables or .env file")