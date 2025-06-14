# app/api/deps.py - Updated to use custom JWT module
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import logging
import sys
from typing import Optional

from app.db.session import get_db
from app.db.models import Tenant
from app.core.config import settings

# Use custom JWT module instead of direct import
from app.auth.jwt import decode_token, verify_tenant_access

# Import new services
from app.services.llamaindex.engine import LlamaIndexRAGEngine
from app.services.document_processor.processor import DocumentProcessor
from app.services.web_scraper.scraper import AdvancedWebScraper
from app.services.query.engine import AdvancedQueryEngine
from app.services.storage.s3 import S3Storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger specifically for authentication
auth_logger = logging.getLogger("auth")
auth_logger.setLevel(logging.INFO)

# Security scheme for JWT
security = HTTPBearer()

# Global service instances (initialized once)
_rag_engine: Optional[LlamaIndexRAGEngine] = None
_document_processor: Optional[DocumentProcessor] = None
_query_engine: Optional[AdvancedQueryEngine] = None
_web_scraper: Optional[AdvancedWebScraper] = None
_s3_storage: Optional[S3Storage] = None

async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """
    Extract and validate tenant from JWT token using custom JWT module
    Returns the Tenant object from the database
    """
    auth_logger.info("=== Authentication attempt started ===")
    
    try:
        # Get token from header
        token = credentials.credentials
        auth_logger.info(f"Token received (first 10 chars): {token[:10]}...")
        
        # Decode token using custom JWT module
        auth_logger.info("Attempting to decode JWT token using custom JWT module...")
        payload = decode_token(token)
        auth_logger.info(f"Token decoded successfully. Claims: {payload}")
        
        # Extract tenant_id from token
        tenant_id_raw = payload.get("tenant_id")
        auth_logger.info(f"Extracted tenant_id from token: {tenant_id_raw}")

        if not tenant_id_raw:
            auth_logger.error("Token missing tenant_id claim")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials: missing tenant_id claim"
            )
        
        # Convert string to UUID
        try:
            if isinstance(tenant_id_raw, str):
                tenant_id = uuid.UUID(tenant_id_raw)
                auth_logger.info(f"Converted string tenant_id to UUID: {tenant_id}")
            else:
                tenant_id = tenant_id_raw
                auth_logger.info(f"Using tenant_id directly: {tenant_id}")
        except ValueError as e:
            auth_logger.error(f"Invalid tenant_id format: {tenant_id_raw}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid tenant ID format: {str(e)}"
            )
        
        # Verify tenant access using custom JWT module
        has_access = verify_tenant_access(payload, tenant_id)
        if not has_access:
            auth_logger.error(f"Tenant access verification failed: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this tenant"
            )
        
        # Get tenant from database
        auth_logger.info(f"Looking up tenant in database with ID: {tenant_id}")
        tenant = await db.get(Tenant, tenant_id)
        
        if not tenant:
            auth_logger.error(f"Tenant not found in database: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant not found in database"
            )
        
        auth_logger.info(f"Tenant found: {tenant.email} (ID: {tenant.id})")
        
        if not tenant.is_active:
            auth_logger.error(f"Tenant is inactive: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant is inactive"
            )
        
        auth_logger.info("=== Authentication successful ===")
        return tenant
        
    except HTTPException:
        # Re-raise HTTPException as-is (they're already handled by custom JWT module)
        raise
    except Exception as e:
        auth_logger.error(f"Authentication error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )

def get_rag_engine(
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None
) -> LlamaIndexRAGEngine:
    """Get or create RAG engine instance with configurable models"""
    global _rag_engine
    
    # Check if we need to recreate the engine with different models
    if (_rag_engine is None or 
        (embedding_model and embedding_model != _rag_engine.embedding_model) or
        (llm_model and llm_model != _rag_engine.llm_model)):
        
        logger = logging.getLogger(__name__)
        logger.info(f"Creating RAG engine with embedding_model={embedding_model}, llm_model={llm_model}")
        
        _rag_engine = LlamaIndexRAGEngine(
            embedding_model=embedding_model,
            llm_model=llm_model
        )
    
    return _rag_engine

def get_document_processor() -> DocumentProcessor:
    """Get or create enhanced document processor instance"""
    global _document_processor, _rag_engine
    
    if _document_processor is None:
        # Get RAG engine if not already created
        if _rag_engine is None:
            _rag_engine = get_rag_engine()
        
        _document_processor = DocumentProcessor(rag_engine=_rag_engine)
    
    return _document_processor

def get_query_engine() -> AdvancedQueryEngine:
    """Get or create advanced query engine instance"""
    global _query_engine, _rag_engine
    
    if _query_engine is None:
        # Get RAG engine if not already created
        if _rag_engine is None:
            _rag_engine = get_rag_engine()
        
        _query_engine = AdvancedQueryEngine(rag_engine=_rag_engine)
    
    return _query_engine

def get_web_scraper() -> AdvancedWebScraper:
    """Get or create advanced web scraper instance"""
    global _web_scraper
    
    if _web_scraper is None:
        _web_scraper = AdvancedWebScraper()
    
    return _web_scraper

def get_s3_storage() -> S3Storage:
    """Get or create S3 storage instance"""
    global _s3_storage
    
    if _s3_storage is None:
        _s3_storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY_ID,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            bucket_name=settings.S3_BUCKET_NAME,
            region=settings.AWS_REGION
        )
    
    return _s3_storage

# Legacy compatibility functions
def get_embedding_service():
    """Legacy compatibility - get embedding service from RAG engine"""
    rag_engine = get_rag_engine()
    return rag_engine.embed_model

def get_vector_store():
    """Legacy compatibility - get vector store from RAG engine"""
    rag_engine = get_rag_engine()
    return rag_engine

def get_url_processor():
    """Legacy compatibility - return web scraper"""
    return get_web_scraper()

# Configuration dependencies for model selection
def get_available_models() -> dict:
    """Get available model configurations"""
    return {
        "embedding_models": settings.AVAILABLE_EMBEDDING_MODELS,
        "llm_models": settings.AVAILABLE_LLM_MODELS
    }

def validate_model_selection(
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None
) -> dict:
    """Validate and return model selection"""
    
    # Validate embedding model
    if embedding_model and embedding_model not in settings.AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid embedding model: {embedding_model}. Available models: {list(settings.AVAILABLE_EMBEDDING_MODELS.keys())}"
        )
    
    # Validate LLM model
    if llm_model and llm_model not in settings.AVAILABLE_LLM_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid LLM model: {llm_model}. Available models: {list(settings.AVAILABLE_LLM_MODELS.keys())}"
        )
    
    return {
        "embedding_model": embedding_model or settings.DEFAULT_EMBEDDING_MODEL,
        "llm_model": llm_model or settings.DEFAULT_LLM_MODEL
    }

# Service factory functions with model configuration
def create_configured_rag_engine(
    embedding_model: str,
    llm_model: str
) -> LlamaIndexRAGEngine:
    """Create a new RAG engine with specific model configuration"""
    return LlamaIndexRAGEngine(
        embedding_model=embedding_model,
        llm_model=llm_model
    )

def create_configured_query_engine(
    rag_engine: LlamaIndexRAGEngine
) -> AdvancedQueryEngine:
    """Create a new query engine with specific RAG engine"""
    return AdvancedQueryEngine(rag_engine=rag_engine)

# Database session dependency
def get_db_dependency():
    """Dependency for database session"""
    return get_db

# Cleanup function for graceful shutdown
async def cleanup_services():
    """Cleanup global service instances"""
    global _rag_engine, _document_processor, _query_engine, _web_scraper, _s3_storage
    
    # Clean up web scraper resources
    if _web_scraper:
        try:
            _web_scraper.__del__()
        except:
            pass
    
    # Reset global instances
    _rag_engine = None
    _document_processor = None
    _query_engine = None
    _web_scraper = None
    _s3_storage = None
    
    logging.getLogger(__name__).info("Services cleaned up successfully")