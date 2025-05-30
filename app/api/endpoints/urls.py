from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
import uuid
import logging
from app.db.session import get_db
from app.db.models import Document, Chunk, KnowledgeBase
from app.schemas.document import (
    URLCreate, DocumentResponse, DocumentStatus,
    ChunkResponse, ReprocessRequest, URLListResponse
)
import aiohttp
from app.api.deps import (
    get_current_tenant, get_url_processor,
    get_embedding_service, get_vector_store, get_s3_storage
)
from app.services.url_processor.processor import URLProcessor
from app.services.embedding.openai import OpenAIEmbedding
from app.services.vector_store.chroma import ChromaVectorStore
from app.services.storage.s3 import S3Storage

router = APIRouter()

DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_COMPLETED = "completed"
DOCUMENT_STATUS_ERROR = "error"
DOCUMENT_STATUS_PROCESSED = "processed"

logger = logging.getLogger("s3_storage")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


@router.post("/{kb_id}/add-url", response_model=DocumentResponse)
async def add_url(
    kb_id: str,
    url_data: URLCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    url_processor: URLProcessor = Depends(get_url_processor),
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    s3_storage: S3Storage = Depends(get_s3_storage)
):
    """Add a URL to a knowledge base"""
    try:
        # Verify knowledge base exists and belongs to tenant
        kb_query = select(KnowledgeBase).where(
            KnowledgeBase.id == uuid.UUID(kb_id),
            KnowledgeBase.tenant_id == current_tenant.id
        )
        result = await db.execute(kb_query)
        kb = result.scalar_one_or_none()
        
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Validate URL format
        url = url_data.url
        if not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL must start with http:// or https://"
            )
        
        # Check if URL already exists in this KB
        existing_query = select(Document).where(
            Document.kb_id == uuid.UUID(kb_id),
            Document.source_url == url,
            Document.tenant_id == current_tenant.id
        )
        result = await db.execute(existing_query)
        existing = result.scalar_one_or_none()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"URL already exists in this knowledge base"
            )
        
        # Create document record for URL
        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            kb_id=uuid.UUID(kb_id),
            tenant_id=current_tenant.id,
            name=url,
            type="url",
            source_url=url,
            status=DOCUMENT_STATUS_PROCESSING
        )
        
        # Save document to database
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Process URL in background
        background_tasks.add_task(
            process_url_background,
            document_id=str(document_id),
            kb_id=kb_id,
            tenant_id=str(current_tenant.id),
            url=url,
            url_processor=url_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
            s3_storage=s3_storage,
            db=db
        )
        
        return {
            "id": str(document.id),
            "kb_id": kb_id,
            "name": document.name,
            "type": document.type,
            "status": document.status,
            "created_at": document.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add URL: {str(e)}"
        )

async def process_url_background(
    document_id: str,
    kb_id: str,
    tenant_id: str,
    url: str,
    url_processor: URLProcessor,
    embedding_service: OpenAIEmbedding,
    vector_store: ChromaVectorStore,
    s3_storage: S3Storage,
    db: AsyncSession
):
    """Background task to process a URL"""
    try:
        # Get document from database
        document = await db.get(Document, uuid.UUID(document_id))
        if not document:
            logger.error(f"Document {document_id} not found")
            return
        
        # Update status to processing
        document.status = DOCUMENT_STATUS_PROCESSING
        await db.commit()
        
        # Fetch URL content
        try:
            # First, fetch the raw HTML content
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
                }) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch URL: {url}, status code: {response.status}")
                    
                    html_content = await response.text()
            
            # Store the raw HTML content in S3
            file_path = f"{tenant_id}/{kb_id}/{document_id}/url_content.html"
            upload_result = await s3_storage.upload_file(
                file_content=html_content.encode('utf-8'),
                file_path=file_path,
                content_type="text/html"
            )
            
            # Update document with file path to HTML content
            document.file_path = file_path
            s3_url = upload_result.get("file_url", "")
            # Keep the original URL in source_url
            await db.commit()
            
            # Now process the text content for embedding
            text = await url_processor.fetch_url(url)
        except Exception as e:
            logger.error(f"Error fetching URL: {str(e)}")
            document.status = DOCUMENT_STATUS_ERROR
            await db.commit()
            return
        
        # Process URL
        result = await url_processor.process_and_embed(
            url=url,
            document_id=document_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Create chunk records in database
        if result["success"]:
            # Extract chunks from result
            chunks = await url_processor.chunk_text(text)
            
            # Create chunk records
            for i, chunk in enumerate(chunks):
                db_chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    kb_id=uuid.UUID(kb_id),
                    tenant_id=uuid.UUID(tenant_id),
                    text=chunk.text,
                    position=i,
                    vector_id=f"{document_id}_{i}"
                )
                db.add(db_chunk)
            
            # Update document status
            document.status = DOCUMENT_STATUS_PROCESSED
        else:
            document.status = DOCUMENT_STATUS_ERROR
        
        await db.commit()
        
    except Exception as e:
        logger.error(f"Error processing URL {document_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Update document status to error
        try:
            document = await db.get(Document, uuid.UUID(document_id))
            if document:
                document.status = DOCUMENT_STATUS_ERROR
                await db.commit()
        except Exception as inner_e:
            logger.error(f"Error updating document status: {str(inner_e)}")

async def process_url_background(
    document_id: str,
    kb_id: str,
    tenant_id: str,
    url: str,
    url_processor: URLProcessor,
    embedding_service: OpenAIEmbedding,
    vector_store: ChromaVectorStore,
    s3_storage: S3Storage,
    db: AsyncSession
):
    """Background task to process a URL"""
    try:
        # Get document from database
        document = await db.get(Document, uuid.UUID(document_id))
        if not document:
            print(f"Document {document_id} not found")
            return
        
        # Update status to processing
        document.status = DOCUMENT_STATUS_PROCESSING
        await db.commit()
        
        # Fetch URL content
        try:
            text = await url_processor.fetch_url(url)
            
            # Store the raw HTML content in S3
            file_path = f"{tenant_id}/{kb_id}/{document_id}/url_content.txt"
            await s3_storage.upload_file(text.encode('utf-8'), file_path, "text/plain")
            
            # Update document with file path
            document.file_path = file_path
            await db.commit()
        except Exception as e:
            print(f"Error fetching URL: {str(e)}")
            document.status = DOCUMENT_STATUS_ERROR
            await db.commit()
            return
        
        # Process URL
        result = await url_processor.process_and_embed(
            url=url,
            document_id=document_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Create chunk records in database
        if result["success"]:
            # Extract chunks from result
            chunks = await url_processor.chunk_text(text)
            
            # Create chunk records
            for i, chunk in enumerate(chunks):
                db_chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    kb_id=uuid.UUID(kb_id),
                    tenant_id=uuid.UUID(tenant_id),
                    text=chunk.text,
                    position=i,
                    vector_id=f"{document_id}_{i}"
                )
                db.add(db_chunk)
            
            # Update document status
            document.status = DOCUMENT_STATUS_PROCESSED
        else:
            document.status = DOCUMENT_STATUS_ERROR
        
        await db.commit()
        
    except Exception as e:
        print(f"Error processing URL {document_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Update document status to error
        try:
            document = await db.get(Document, uuid.UUID(document_id))
            if document:
                document.status = DOCUMENT_STATUS_ERROR
                await db.commit()
        except Exception as inner_e:
            print(f"Error updating document status: {str(inner_e)}")

@router.get("/{kb_id}/urls", response_model=List[URLListResponse])
async def list_urls(
    kb_id: str,
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant)
):
    """List all URLs in a knowledge base"""
    try:
        # Verify knowledge base exists and belongs to tenant
        kb_query = select(KnowledgeBase).where(
            KnowledgeBase.id == uuid.UUID(kb_id),
            KnowledgeBase.tenant_id == current_tenant.id
        )
        result = await db.execute(kb_query)
        kb = result.scalar_one_or_none()
        
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Query URLs
        url_query = select(Document).where(
            Document.kb_id == uuid.UUID(kb_id),
            Document.tenant_id == current_tenant.id,
            Document.type == "url"
        ).order_by(Document.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(url_query)
        urls = result.scalars().all()
        
        # Get chunk counts for each URL
        url_list = []
        for url in urls:
            # Count chunks
            chunk_query = select(Chunk).where(
                Chunk.document_id == url.id
            )
            chunk_result = await db.execute(chunk_query)
            chunks = chunk_result.scalars().all()
            
            url_list.append({
                "id": str(url.id),
                "kb_id": kb_id,
                "url": url.source_url,
                "status": url.status,
                "chunk_count": len(chunks),
                "created_at": url.created_at
            })
        
        return url_list
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list URLs: {str(e)}"
        )

@router.get("/{kb_id}/urls/{url_id}/status", response_model=DocumentStatus)
async def get_url_status(
    kb_id: str,
    url_id: str,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant)
):
    """Get the processing status of a URL"""
    try:
        # Query document
        document = await db.get(Document, uuid.UUID(url_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"URL with ID {url_id} not found"
            )
        
        # Check ownership
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this URL"
            )
        
        # Count chunks
        chunks_query = select(Chunk).where(
            Chunk.document_id == uuid.UUID(url_id)
        )
        result = await db.execute(chunks_query)
        chunks = result.scalars().all()
        
        return {
            "id": url_id,
            "status": document.status,
            "chunks": len(chunks),
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get URL status: {str(e)}"
        )

@router.get("/{kb_id}/urls/{url_id}/chunks", response_model=List[ChunkResponse])
async def get_url_chunks(
    kb_id: str,
    url_id: str,
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant)
):
    """Get chunks for a URL"""
    try:
        # Verify document exists and belongs to tenant
        document = await db.get(Document, uuid.UUID(url_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"URL with ID {url_id} not found"
            )
        
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this URL"
            )
        
        # Query chunks with pagination
        chunks_query = select(Chunk).where(
            Chunk.document_id == uuid.UUID(url_id)
        ).order_by(Chunk.position).offset(skip).limit(limit)
        
        result = await db.execute(chunks_query)
        chunks = result.scalars().all()
        
        return [
            {
                "id": str(chunk.id),
                "document_id": url_id,
                "position": chunk.position,
                "text": chunk.text,
                "vector_id": chunk.vector_id,
                "created_at": chunk.created_at
            }
            for chunk in chunks
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get URL chunks: {str(e)}"
        )

@router.post("/{kb_id}/urls/{url_id}/reprocess", response_model=DocumentStatus)
async def reprocess_url(
    kb_id: str,
    url_id: str,
    request: ReprocessRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    url_processor: URLProcessor = Depends(get_url_processor),
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    s3_storage: S3Storage = Depends(get_s3_storage)
):
    """Reprocess a URL"""
    try:
        # Verify document exists and belongs to tenant
        document = await db.get(Document, uuid.UUID(url_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"URL with ID {url_id} not found"
            )
        
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this URL"
            )
        
        # Check if document has a source URL
        if not document.source_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no associated URL"
            )
        
        # Update status to pending
        document.status = DOCUMENT_STATUS_PENDING
        await db.commit()
        
        # Delete existing chunks
        await db.execute(
            "DELETE FROM chunks WHERE document_id = :doc_id",
            {"doc_id": uuid.UUID(url_id)}
        )
        await db.commit()
        
        # Delete existing vectors from ChromaDB
        try:
            collection = vector_store.client.get_collection(name=kb_id)
            collection.delete(
                where={"doc_id": url_id}
            )
        except Exception as e:
            print(f"Error deleting vectors from ChromaDB: {str(e)}")
        
        # Process URL in background
        background_tasks.add_task(
            process_url_background,
            document_id=url_id,
            kb_id=kb_id,
            tenant_id=str(current_tenant.id),
            url=document.source_url,
            url_processor=url_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
            s3_storage=s3_storage,
            db=db
        )
        
        return {
            "id": url_id,
            "status": DOCUMENT_STATUS_PENDING,
            "chunks": 0,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reprocess URL: {str(e)}"
        )