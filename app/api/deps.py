# app/api/deps.py - Updated to use the fixed OpenAI service
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
import uuid
import logging
import sys

from app.db.session import get_db
from app.db.models import Tenant
from app.core.config import settings
from app.services.document_processor.processor import DocumentProcessor
from app.services.url_processor.processor import URLProcessor
from app.services.embedding.openai import OpenAIEmbedding  
from app.services.vector_store.chroma import ChromaVectorStore
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

async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """
    Extract and validate tenant from JWT token
    Returns the Tenant object from the database
    """
    auth_logger.info("=== Authentication attempt started ===")
    
    try:
        # Get token from header
        token = credentials.credentials
        auth_logger.info(f"Token received (first 10 chars): {token[:10]}...")
        
        # Decode token
        auth_logger.info("Attempting to decode JWT token...")
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        auth_logger.info(f"Token decoded successfully. Claims: {payload}")
        
        # Extract tenant_id from token
        tenant_id_raw = payload.get("tenant_id")  # Only use tenant_id, not sub
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
        
    except jwt.ExpiredSignatureError:
        auth_logger.error("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        auth_logger.error(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        auth_logger.error(f"Authentication error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )

# Database session dependency
def get_db_dependency():
    """Dependency for database session"""
    return get_db

# Service dependencies
def get_document_processor() -> DocumentProcessor:
    """Dependency for document processor service"""
    return DocumentProcessor()

def get_url_processor() -> URLProcessor:
    """Dependency for URL processor service"""
    return URLProcessor()

def get_embedding_service() -> OpenAIEmbedding:
    """Dependency for embedding service"""
    return OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)

def get_vector_store() -> ChromaVectorStore:
    """Dependency for vector store service"""
    return ChromaVectorStore(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)

def get_s3_storage() -> S3Storage:
    """Dependency for S3 storage service"""
    return S3Storage(
        aws_access_key=settings.AWS_ACCESS_KEY_ID,
        aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
        bucket_name=settings.S3_BUCKET_NAME,
        region=settings.AWS_REGION
    )