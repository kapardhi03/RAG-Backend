from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import jwt
from uuid import UUID, uuid4
from fastapi import HTTPException, status
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

def create_access_token(
    tenant_id: Union[UUID, str], 
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token for a tenant
    
    Args:
        tenant_id: The tenant's UUID
        email: The tenant's email
        expires_delta: Optional custom expiration time
        
    Returns:
        JWT token string
    """
    # Convert tenant_id to string if it's a UUID
    str_tenant_id = str(tenant_id)
    
    expire = datetime.utcnow() + (
        expires_delta if expires_delta 
        else timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    to_encode = {
        "sub": email,
        "tenant_id": str_tenant_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid4())  # JWT ID - unique identifier for this token
    }
    
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating JWT token: {str(e)}")
        raise

def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Attempt to use expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected error decoding JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_tenant_access(token_payload: Dict[str, Any], requested_tenant_id: UUID) -> bool:
    """
    Verify if the token has access to the requested tenant
    
    Args:
        token_payload: Decoded token payload
        requested_tenant_id: The tenant ID being accessed
        
    Returns:
        True if access is allowed, False otherwise
    """
    token_tenant_id = token_payload.get("tenant_id")
    
    if not token_tenant_id:
        logger.warning("Token missing tenant_id claim")
        return False
    
    # Check if the token's tenant_id matches the requested tenant_id
    has_access = str(token_tenant_id) == str(requested_tenant_id)
    
    if not has_access:
        logger.warning(
            f"Tenant access violation - Token tenant: {token_tenant_id}, "
            f"Requested tenant: {requested_tenant_id}"
        )
    
    return has_access