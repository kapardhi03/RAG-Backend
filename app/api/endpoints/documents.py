from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, BackgroundTasks, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
import uuid
import logging
from app.db.session import get_db
from app.db.models import Document, Chunk, KnowledgeBase
from app.schemas.document import (
    DocumentResponse, DocumentStatus,
    ChunkResponse, ReprocessRequest
)
from app.api.deps import (
    get_current_tenant, get_document_processor,
    get_embedding_service, get_vector_store, get_s3_storage
)
from app.services.document_processor.processor import DocumentProcessor
from app.services.embedding.openai import OpenAIEmbedding
from app.services.vector_store.chroma import ChromaVectorStore
from app.services.storage.s3 import S3Storage
from app.utils.file_utils import get_file_extension, is_supported_file_type

router = APIRouter()

DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_COMPLETED = "completed"  # This exists in your enum
DOCUMENT_STATUS_ERROR = "error"  # This exists in your enum
DOCUMENT_STATUS_PROCESSED = "processed"  # This exists in your enum

logger = logging.getLogger("s3_storage")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


@router.post("/{kb_id}/upload", response_model=DocumentResponse)
async def upload_document(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    s3_storage: S3Storage = Depends(get_s3_storage)
):
    """Upload a document to a knowledge base"""
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
        
        # Check if file type is supported
        if not is_supported_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported types: PDF, DOCX, DOC, TXT, MD"
            )
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Extract file extension
        filename = file.filename
        extension = get_file_extension(filename)
        
        # Create document record in database
        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            kb_id=uuid.UUID(kb_id),
            tenant_id=current_tenant.id,
            name=filename,
            type="file",
            content_type=file.content_type,
            file_size=file_size,
            status=DOCUMENT_STATUS_PENDING
        )
        
        # Save document to database
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Save file to S3
        file_path = f"{current_tenant.id}/{kb_id}/{document_id}/original.{extension}"
        
        try:
            # Upload the raw file content to S3
            upload_result = await s3_storage.upload_file(
                file_content=file_content,
                file_path=file_path,
                content_type=file.content_type
            )
            
            # Get the S3 URL
            s3_url = upload_result.get("file_url", "")
            
            # Update document with file path and S3 URL
            document.file_path = file_path
            document.source_url = s3_url  # Store the S3 URL in the database
            await db.commit()
            
        except Exception as s3_error:
            logger.error(f"Error uploading to S3: {str(s3_error)}")
            document.status = DOCUMENT_STATUS_ERROR
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload document to storage: {str(s3_error)}"
            )
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            document_id=str(document_id),
            kb_id=kb_id,
            tenant_id=str(current_tenant.id),
            file_content=file_content,
            filename=filename,
            document_processor=document_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
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
        logger.error(f"Failed to upload document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )
    
async def process_document_background(
    document_id: str,
    kb_id: str,
    tenant_id: str,
    file_content: bytes,
    filename: str,
    document_processor: DocumentProcessor,
    embedding_service: OpenAIEmbedding,
    vector_store: ChromaVectorStore,
    db: AsyncSession
):
    """Background task to process a document"""
    try:
        # Get document from database
        document = await db.get(Document, uuid.UUID(document_id))
        if not document:
            print(f"Document {document_id} not found")
            return
        
        # Update status to processing
        document.status = DOCUMENT_STATUS_PROCESSING
        await db.commit()
        
        # Process document
        result = await document_processor.process_and_embed(
            file_content=file_content,
            filename=filename,
            document_id=document_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Create chunk records in database
        if result["success"]:
            # Extract chunks from result
            text = await document_processor.process_file(file_content, filename)
            chunks = await document_processor.chunk_text(text)
            
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
            
            # Update document status to completed
            document.status = DOCUMENT_STATUS_COMPLETED
        else:
            # Update document status to error
            document.status = DOCUMENT_STATUS_ERROR
        
        await db.commit()
        
    except Exception as e:
        print(f"Error processing document {document_id}: {str(e)}")
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

@router.get("/{kb_id}/documents/{doc_id}/status", response_model=DocumentStatus)
async def get_document_status(
    kb_id: str,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant)
):
    """Get the processing status of a document"""
    try:
        # Query document
        document = await db.get(Document, uuid.UUID(doc_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found"
            )
        
        # Check ownership
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this document"
            )
        
        # Count chunks
        chunks_query = select(Chunk).where(
            Chunk.document_id == uuid.UUID(doc_id)
        )
        result = await db.execute(chunks_query)
        chunks = result.scalars().all()
        
        return {
            "id": doc_id,
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
            detail=f"Failed to get document status: {str(e)}"
        )

@router.get("/{kb_id}/documents/{doc_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    kb_id: str,
    doc_id: str,
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant)
):
    """Get chunks for a document"""
    try:
        # Verify document exists and belongs to tenant
        document = await db.get(Document, uuid.UUID(doc_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found"
            )
        
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this document"
            )
        
        # Query chunks with pagination
        chunks_query = select(Chunk).where(
            Chunk.document_id == uuid.UUID(doc_id)
        ).order_by(Chunk.position).offset(skip).limit(limit)
        
        result = await db.execute(chunks_query)
        chunks = result.scalars().all()
        
        return [
            {
                "id": str(chunk.id),
                "document_id": doc_id,
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
            detail=f"Failed to get document chunks: {str(e)}"
        )

@router.post("/{kb_id}/documents/{doc_id}/reprocess", response_model=DocumentStatus)
async def reprocess_document(
    kb_id: str,
    doc_id: str,
    request: ReprocessRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    s3_storage: S3Storage = Depends(get_s3_storage)
):
    """Reprocess a document"""
    try:
        # Verify document exists and belongs to tenant
        document = await db.get(Document, uuid.UUID(doc_id))
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found"
            )
        
        if document.tenant_id != current_tenant.id or str(document.kb_id) != kb_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this document"
            )
        
        # Check if document has a file path
        if not document.file_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no associated file"
            )
        
        # Update status to pending
        document.status = DOCUMENT_STATUS_PENDING
        await db.commit()
        
        # Delete existing chunks
        await db.execute(
            "DELETE FROM chunks WHERE document_id = :doc_id",
            {"doc_id": uuid.UUID(doc_id)}
        )
        await db.commit()
        
        # Delete existing vectors from ChromaDB
        try:
            collection = vector_store.client.get_collection(name=kb_id)
            collection.delete(
                where={"doc_id": doc_id}
            )
        except Exception as e:
            print(f"Error deleting vectors from ChromaDB: {str(e)}")
        
        # Download file from S3
        try:
            file_content = await s3_storage.download_file(document.file_path)
        except Exception as e:
            document.status = DOCUMENT_STATUS_ERROR
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download document from storage: {str(e)}"
            )
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            document_id=doc_id,
            kb_id=kb_id,
            tenant_id=str(current_tenant.id),
            file_content=file_content,
            filename=document.name,
            document_processor=document_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
            db=db
        )
        
        return {
            "id": doc_id,
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
            detail=f"Failed to reprocess document: {str(e)}"
        )