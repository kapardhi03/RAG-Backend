from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from uuid import  UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import KnowledgeBase, Document
from app.api.deps import get_current_tenant, get_web_scraper
from app.db.session import get_db
from app.utils.url_utils import extract_all_urls  # Fixed import to use utils directory
import logging

from app.schemas.knowledge_base import (
    KnowledgeBaseCreate, 
    KnowledgeBaseRead, 
    DocumentRead,
    URLSitemapResponse,  # Add this schema
    URLSubmit  # Add this schema
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/create", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    kb_create: KnowledgeBaseCreate,
    tenant = Depends(get_current_tenant), 
    db: AsyncSession = Depends(get_db)
):
    """Create a new knowledge base for a tenant"""
    try:
        # Create a new knowledge base
        kb = KnowledgeBase(
            tenant_id=tenant.id,  
            name=kb_create.name,
            description=kb_create.description
        )
        
        # Add to database
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        
        logger.info(f"Created knowledge base {kb.id} for tenant {tenant.id}")
        
        return {
            "kb_id": str(kb.id),
            "tenant_id": str(kb.tenant_id),
            "name": kb.name,
            "description": kb.description,
            "created_at": kb.created_at,
            "document_count": 0
        }
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create knowledge base: {str(e)}"
        )

@router.post("/extract-sitemap", response_model=URLSitemapResponse)
async def extract_sitemap_urls(
    url_data: URLSubmit,
    tenant = Depends(get_current_tenant)
):
    """
    Extract all unique URLs from a website using advanced scraping.
    
    This endpoint combines multiple URL discovery methods:
    - Sitemap.xml parsing
    - robots.txt analysis  
    - Intelligent crawling with link discovery
    """
    try:
        url = str(url_data.url)
        logger.info(f"Starting URL extraction for: {url}")
        
        # Use the extract_all_urls function from utils
        urls = await extract_all_urls(
            base_url=url,
            max_urls=500,
            include_crawling=True,
            same_domain_only=True
        )
        
        logger.info(f"URL extraction completed. Found {len(urls)} unique URLs")
        
        return URLSitemapResponse(urls=urls)
        
    except Exception as e:
        logger.error(f"Failed to extract URLs from {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract URLs: {str(e)}"
        )

@router.get("/{kb_id}", response_model=List[DocumentRead])
async def get_all_files_in_kb(
    kb_id: str,
    tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get all files in a knowledge base"""
    try:
        # Check if knowledge base exists and belongs to tenant
        kb = await KnowledgeBase.get_by_id_and_tenant(UUID(kb_id), tenant.id, db)
        if not kb:
            logger.warning(f"Knowledge base {kb_id} not found for tenant {tenant.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found for this tenant"
            )
        
        # Get all documents for the knowledge base
        documents = await Document.get_by_kb_and_tenant(UUID(kb_id), tenant.id, db)
        
        # Convert to response format
        result = []
        for doc in documents:
            result.append({
                "document_id": str(doc.id),
                "kb_id": str(doc.kb_id),
                "tenant_id": str(doc.tenant_id),
                "name": doc.name,
                "type": doc.type,
                "status": doc.status,
                "created_at": doc.created_at,
                "source_url": doc.source_url
            })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get files in KB {kb_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve files: {str(e)}"
        )

@router.get("/", response_model=List[KnowledgeBaseRead])
async def get_knowledge_bases(
    tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get all knowledge bases for a tenant"""
    try:
        # Get all knowledge bases for the tenant from database
        knowledge_bases = await KnowledgeBase.get_by_tenant(tenant.id, db)
        
        # Convert to response format
        result = []
        for kb in knowledge_bases:
            # Count documents for this knowledge base
            from sqlalchemy import func, select
            count_query = select(func.count()).filter(
                Document.kb_id == kb.id,
                Document.tenant_id == tenant.id,
            )
            count_result = await db.execute(count_query)
            document_count = count_result.scalar()
            
            result.append({
                "kb_id": str(kb.id),
                "tenant_id": str(kb.tenant_id),
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at,
                "document_count": document_count or 0
            })
        
        return result
    except Exception as e:
        logger.error(f"Failed to get knowledge bases for tenant {tenant.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge bases: {str(e)}"
        )