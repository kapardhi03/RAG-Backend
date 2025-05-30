from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional
from datetime import timedelta, datetime
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.config import settings
from app.db.session import get_db
from app.db.models import Tenant

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    tenant_id: str
    email: str
    name: str

class UserCreate(BaseModel):
    """Schema for creating a new user"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    name: str = Field(..., description="User name")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)  # Get database session from dependency
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    logger.info(f"Login attempt for: {form_data.username}")
    
    try:
        logger.info(f"Looking for user in database: {form_data.username}")
        tenant = await Tenant.get_by_email(form_data.username, db)
        logger.info(f"Database query result: {tenant}")
        
        if tenant and tenant.verify_password(form_data.password):
            logger.info(f"Password verification successful for: {form_data.username}")
            access_token = create_access_token(
                data={"sub": tenant.email, "tenant_id": str(tenant.id)}
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "tenant_id": str(tenant.id),
                "email": tenant.email,
                "name": tenant.name
            }
    except Exception as e:
        logger.error(f"Database auth error: {str(e)}")
    
    # If we get here, authentication failed
    logger.warning(f"Authentication failed for: {form_data.username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    try:
        logger.info(f"Attempting to register user: {user_data.email}")
            
        # Check if email already exists in database
        existing_tenant = await Tenant.get_by_email(user_data.email, db)
        if existing_tenant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new tenant
        tenant = Tenant(
            email=user_data.email,
            hashed_password=Tenant.get_password_hash(user_data.password),
            name=user_data.name
        )
        
        logger.info("Created tenant object successfully")
        
        # Add to database
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)
        
        logger.info(f"User {tenant.email} registered successfully with ID: {tenant.id}")
        
        # Create token
        access_token = create_access_token(
            data={"sub": tenant.email, "tenant_id": str(tenant.id)}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "tenant_id": str(tenant.id),
            "email": tenant.email,
            "name": tenant.name
        }
    except Exception as e:
        logger.error(f"Registration error details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Registration error: {str(e)}"
        )