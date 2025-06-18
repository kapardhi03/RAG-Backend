from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import uuid
from uuid import UUID
from fastapi import HTTPException, status
import logging

# Explicit PyJWT import with error handling
try:
    import jwt as pyjwt
    # Verify this is PyJWT and not python-jwt
    if not hasattr(pyjwt, 'encode') or not hasattr(pyjwt, 'decode'):
        raise ImportError("Wrong JWT library detected")
    print(f"Using PyJWT version: {pyjwt.__version__}")
except ImportError as e:
    print(f"JWT Import Error: {e}")
    print("Please run: pip uninstall python-jwt jwt PyJWT -y && pip install PyJWT==2.10.1")
    raise

from app.core.config import settings

logger = logging.getLogger(__name__)

# JWT Configuration constants
ALGORITHM = settings.JWT_ALGORITHM
SECRET_KEY = settings.JWT_SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(
    tenant_id: Union[UUID, str], 
    email: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT access token for a tenant
    
    Args:
        tenant_id: The tenant's UUID
        email: The tenant's email
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional claims to include in token
        
    Returns:
        JWT token string
    """
    try:
        # Convert tenant_id to string if it's a UUID
        str_tenant_id = str(tenant_id)
        
        expire = datetime.utcnow() + (
            expires_delta if expires_delta 
            else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        to_encode = {
            "sub": email,  # Subject (standard JWT claim)
            "tenant_id": str_tenant_id,
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # JWT ID - unique identifier for this token
            "type": "access_token"
        }
        
        # Add any additional claims
        if additional_claims:
            to_encode.update(additional_claims)
        
        # Create JWT token using PyJWT
        encoded_jwt = pyjwt.encode(
            to_encode, 
            SECRET_KEY, 
            algorithm=ALGORITHM
        )
        
        logger.info(f"JWT token created successfully for tenant {str_tenant_id}")
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Error creating JWT token: {str(e)}")
        logger.error(f"PyJWT available methods: {[method for method in dir(pyjwt) if not method.startswith('_')]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authentication token"
        )

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
        payload = pyjwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        # Validate token type
        if payload.get("type") != "access_token":
            logger.warning(f"Invalid token type: {payload.get('type')}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(f"Token decoded successfully for tenant: {payload.get('tenant_id')}")
        return payload
        
    except pyjwt.ExpiredSignatureError:
        logger.warning("Attempt to use expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except pyjwt.InvalidTokenError as e:
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

def extract_tenant_id_from_token(token: str) -> UUID:
    """
    Extract tenant ID from JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Tenant UUID
        
    Raises:
        HTTPException: If token is invalid or missing tenant_id
    """
    payload = decode_token(token)
    tenant_id_str = payload.get("tenant_id")
    
    if not tenant_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tenant_id claim"
        )
    
    try:
        return UUID(tenant_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid tenant_id format in token"
        )

def create_refresh_token(
    tenant_id: Union[UUID, str],
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a refresh token for extended authentication
    
    Args:
        tenant_id: The tenant's UUID
        email: The tenant's email
        expires_delta: Optional custom expiration time (default: 7 days)
        
    Returns:
        JWT refresh token string
    """
    str_tenant_id = str(tenant_id)
    
    expire = datetime.utcnow() + (
        expires_delta if expires_delta 
        else timedelta(days=7)  # Refresh tokens last longer
    )
    
    to_encode = {
        "sub": email,
        "tenant_id": str_tenant_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),
        "type": "refresh_token"
    }
    
    try:
        encoded_jwt = pyjwt.encode(
            to_encode, 
            SECRET_KEY, 
            algorithm=ALGORITHM
        )
        
        logger.info(f"Refresh token created for tenant {str_tenant_id}")
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Error creating refresh token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create refresh token"
        )

def validate_refresh_token(token: str) -> Dict[str, Any]:
    """
    Validate and decode a refresh token
    
    Args:
        token: JWT refresh token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or not a refresh token
    """
    payload = decode_token(token)  # This will handle basic validation
    
    # Additional validation for refresh token type
    if payload.get("type") != "refresh_token":
        logger.warning(f"Expected refresh token, got: {payload.get('type')}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type - expected refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload

def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired without raising an exception
    
    Args:
        token: JWT token string
        
    Returns:
        True if token is expired, False otherwise
    """
    try:
        pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return False
    except pyjwt.ExpiredSignatureError:
        return True
    except pyjwt.InvalidTokenError:
        return True  # Consider invalid tokens as expired

def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get the expiry time of a token
    
    Args:
        token: JWT token string
        
    Returns:
        Expiry datetime or None if token is invalid
    """
    try:
        payload = pyjwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            options={"verify_exp": False}  # Don't verify expiry for this check
        )
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp)
        return None
    except pyjwt.InvalidTokenError:
        return None

def create_token_pair(
    tenant_id: Union[UUID, str],
    email: str,
    access_expires_delta: Optional[timedelta] = None,
    refresh_expires_delta: Optional[timedelta] = None
) -> Dict[str, str]:
    """
    Create both access and refresh tokens
    
    Args:
        tenant_id: The tenant's UUID
        email: The tenant's email
        access_expires_delta: Optional custom expiration for access token
        refresh_expires_delta: Optional custom expiration for refresh token
        
    Returns:
        Dictionary with both tokens
    """
    access_token = create_access_token(tenant_id, email, access_expires_delta)
    refresh_token = create_refresh_token(tenant_id, email, refresh_expires_delta)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

def refresh_access_token(refresh_token: str) -> str:
    """
    Create a new access token using a valid refresh token
    
    Args:
        refresh_token: Valid refresh token
        
    Returns:
        New access token
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    payload = validate_refresh_token(refresh_token)
    
    # Extract tenant info from refresh token
    tenant_id = payload.get("tenant_id")
    email = payload.get("email")
    
    if not tenant_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token claims"
        )
    
    # Create new access token
    return create_access_token(tenant_id, email)

# Utility functions for common JWT operations
class JWTHelper:
    """Helper class with common JWT operations"""
    
    @staticmethod
    def create_login_response(tenant_id: Union[UUID, str], email: str, name: str) -> Dict[str, str]:
        """Create a standard login response with tokens"""
        tokens = create_token_pair(tenant_id, email)
        
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": tokens["token_type"],
            "tenant_id": str(tenant_id),
            "email": email,
            "name": name
        }
    
    @staticmethod
    def validate_and_extract_tenant(token: str) -> Dict[str, Any]:
        """Validate token and extract tenant information"""
        payload = decode_token(token)
        
        return {
            "tenant_id": payload.get("tenant_id"),
            "email": payload.get("email"),
            "jti": payload.get("jti"),
            "exp": payload.get("exp"),
            "iat": payload.get("iat")
        }