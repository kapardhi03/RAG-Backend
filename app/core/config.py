import os
import secrets
from typing import List, Dict, Any

class Settings:
    """Simple settings class that avoids Pydantic parsing issues"""
    
    def __init__(self):
        # API Configuration
        self.PROJECT_NAME = "RAG-Microservices"
        self.API_V1_STR = "/api"
        
        # Authentication
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
        self.JWT_ALGORITHM = "HS256"
        self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "4320"))
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        
        # Default Model Configuration
        self.DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo-preview")
        self.FALLBACK_EMBEDDING_MODEL = os.getenv("FALLBACK_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo")
        
        # Cohere Configuration for Re-ranking
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
        self.ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
        self.FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
        
        # AWS S3 Configuration
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "cloneifyai")
        self.AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
        
        # ChromaDB Configuration
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        # Redis Configuration (for task queue)
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Database Configuration
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_microservices")
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
        
        # Enhanced Chunking Configuration
        self.DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1024"))
        self.DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
        self.MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2048"))
        self.MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "256"))
        
        # Semantic Chunking Configuration
        self.ENABLE_SEMANTIC_CHUNKING = os.getenv("ENABLE_SEMANTIC_CHUNKING", "true").lower() == "true"
        self.SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.7"))
        
        # Hybrid Search Configuration
        self.ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
        self.VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))
        self.KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
        
        # Query Expansion Configuration
        self.ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        self.MAX_EXPANDED_TERMS = int(os.getenv("MAX_EXPANDED_TERMS", "5"))
        
        # Advanced Web Scraping Configuration
        self.ENABLE_ADVANCED_SCRAPING = os.getenv("ENABLE_ADVANCED_SCRAPING", "true").lower() == "true"
        self.SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "30"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
        
        # Worker Configuration
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        
        # Performance Configuration
        self.RETRIEVAL_TIMEOUT = int(os.getenv("RETRIEVAL_TIMEOUT", "30"))
        self.GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "60"))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
        
        # Validate critical settings
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate critical settings"""
        if self.ENVIRONMENT == "production":
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set for production")
            if not self.AWS_ACCESS_KEY_ID or not self.AWS_SECRET_ACCESS_KEY:
                raise ValueError("AWS credentials must be set for production")
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Get CORS origins"""
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        if cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    
    @property
    def PROXY_LIST(self) -> List[str]:
        """Get proxy list"""
        proxy_list_str = os.getenv("PROXY_LIST", "")
        if not proxy_list_str or proxy_list_str.strip() == "":
            return []
        return [proxy.strip() for proxy in proxy_list_str.split(",") if proxy.strip()]
    
    @property
    def AVAILABLE_EMBEDDING_MODELS(self) -> Dict[str, Dict[str, Any]]:
        """Get available embedding models configuration"""
        return {
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
    
    @property
    def AVAILABLE_LLM_MODELS(self) -> Dict[str, Dict[str, Any]]:
        """Get available LLM models configuration"""
        return {
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

# Create settings instance with proper error handling
try:
    settings = Settings()
    print("Settings loaded successfully")
    
    # Warning messages for missing critical settings
    if not settings.OPENAI_API_KEY:
        print("⚠️  WARNING: OPENAI_API_KEY is not set. AI features will not work.")
    if not settings.AWS_ACCESS_KEY_ID:
        print("⚠️  WARNING: AWS_ACCESS_KEY_ID is not set. S3 storage will not work.")
    if not settings.AWS_SECRET_ACCESS_KEY:
        print("⚠️  WARNING: AWS_SECRET_ACCESS_KEY is not set. S3 storage will not work.")
    
    # Info messages about configuration
    print(f"🔧 Environment: {settings.ENVIRONMENT}")
    print(f"🤖 Default embedding model: {settings.DEFAULT_EMBEDDING_MODEL}")
    print(f"🧠 Default LLM model: {settings.DEFAULT_LLM_MODEL}")
    print(f"🔍 Semantic chunking: {'✅' if settings.ENABLE_SEMANTIC_CHUNKING else ''}")
    print(f"🔄 Hybrid search: {'✅' if settings.ENABLE_HYBRID_SEARCH else ''}")
    print(f"📈 Query expansion: {'✅' if settings.ENABLE_QUERY_EXPANSION else ''}")
    print(f"🎯 Reranking: {'✅' if settings.ENABLE_RERANKING else ''}")
    
except Exception as e:
    print(f" Critical error loading settings: {e}")
    print("Please check your .env file and fix the configuration.")
    raise