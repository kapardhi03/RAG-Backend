# app/api/endpoints/vectors.py

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
import uuid

from app.db.session import get_db
from app.db.models import Document, Chunk, KnowledgeBase
from app.schemas.document import VectorStats, ReindexResponse
from app.api.deps import (
    get_current_tenant, get_document_processor, get_url_processor,
    get_embedding_service, get_vector_store, get_s3_storage
)
from app.services.document_processor.processor import DocumentProcessor
from app.services.url_processor.processor import URLProcessor
from app.services.embedding.openai import OpenAIEmbedding
from app.services.vector_store.chroma import ChromaVectorStore
from app.services.storage.s3 import S3Storage

# Import background processing functions
from app.api.endpoints.documents import process_document_background
from app.api.endpoints.urls import process_url_background

router = APIRouter()

@router.get("/{kb_id}/vectors", response_model=VectorStats)
async def get_kb_vector_stats(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Get statistics about vectors in a knowledge base"""
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
        
        # Check if collection exists in ChromaDB
        collection_exists = await vector_store.collection_exists(kb_id)
        
        if not collection_exists:
            return {
                "kb_id": kb_id,
                "total_vectors": 0,
                "total_documents": 0,
                "file_documents": 0,
                "url_documents": 0,
                "avg_chunks_per_document": 0
            }
        
        # Get vector stats from ChromaDB
        try:
            vector_stats = await vector_store.get_vector_stats(kb_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get vector stats: {str(e)}"
            )
        
        # Get document counts from database
        doc_query = select(Document).where(
            Document.kb_id == uuid.UUID(kb_id),
            Document.tenant_id == current_tenant.id
        )
        doc_result = await db.execute(doc_query)
        documents = doc_result.scalars().all()
        
        total_documents = len(documents)
        file_documents = sum(1 for doc in documents if doc.type == "file")
        url_documents = sum(1 for doc in documents if doc.type == "url")
        
        # Calculate average chunks per document
        avg_chunks = vector_stats["total_vectors"] / total_documents if total_documents > 0 else 0
        
        return {
            "kb_id": kb_id,
            "total_vectors": vector_stats["total_vectors"],
            "total_documents": total_documents,
            "file_documents": file_documents,
            "url_documents": url_documents,
            "avg_chunks_per_document": round(avg_chunks, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector stats: {str(e)}"
        )

@router.post("/{kb_id}/reindex", response_model=ReindexResponse)
async def reindex_kb(
    kb_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_tenant = Depends(get_current_tenant),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    url_processor: URLProcessor = Depends(get_url_processor),
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    s3_storage: S3Storage = Depends(get_s3_storage)
):
    """Regenerate all vectors in a knowledge base"""
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
        
        # Query all documents in the KB
        doc_query = select(Document).where(
            Document.kb_id == uuid.UUID(kb_id),
            Document.tenant_id == current_tenant.id
        )
        doc_result = await db.execute(doc_query)
        documents = doc_result.scalars().all()
        
        # Delete all chunks from the database
        await db.execute(
            "DELETE FROM chunks WHERE kb_id = :kb_id",
            {"kb_id": uuid.UUID(kb_id)}
        )
        await db.commit()
        
        # Delete collection from ChromaDB if it exists
        try:
            if await vector_store.collection_exists(kb_id):
                await vector_store.delete_collection(kb_id)
        except Exception as e:
            print(f"Error deleting ChromaDB collection: {str(e)}")
        
        # Create new collection
        await vector_store.create_collection(kb_id, kb.description or "")
        
        # Set all documents to pending
        for doc in documents:
            doc.status = "pending"
        await db.commit()
        
        # Schedule reprocessing for all documents
        reindexed_count = 0
        file_count = 0
        url_count = 0
        
        for doc in documents:
            doc_id = str(doc.id)
            tenant_id = str(current_tenant.id)
            
            if doc.type == "file" and doc.file_path:
                # Reprocess file
                file_count += 1
                
                try:
                    # Download file from S3
                    file_content = await s3_storage.download_file(doc.file_path)
                    
                    # Schedule processing
                    background_tasks.add_task(
                        process_document_background,
                        document_id=doc_id,
                        kb_id=kb_id,
                        tenant_id=tenant_id,
                        file_content=file_content,
                        filename=doc.name,
                        document_processor=document_processor,
                        embedding_service=embedding_service,
                        vector_store=vector_store,
                        db=db
                    )
                    
                    reindexed_count += 1
                except Exception as e:
                    print(f"Error scheduling reindexing for file {doc_id}: {str(e)}")
                    doc.status = "error"
                    
            elif doc.type == "url" and doc.source_url:
                # Reprocess URL
                url_count += 1
                
                try:
                    # Schedule processing
                    background_tasks.add_task(
                        process_url_background,
                        document_id=doc_id,
                        kb_id=kb_id,
                        tenant_id=tenant_id,
                        url=doc.source_url,
                        url_processor=url_processor,
                        embedding_service=embedding_service,
                        vector_store=vector_store,
                        s3_storage=s3_storage,
                        db=db
                    )
                    
                    reindexed_count += 1
                except Exception as e:
                    print(f"Error scheduling reindexing for URL {doc_id}: {str(e)}")
                    doc.status = "error"
        
        await db.commit()
        
        return {
            "kb_id": kb_id,
            "documents_scheduled": reindexed_count,
            "file_documents": file_count,
            "url_documents": url_count,
            "status": "processing"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex knowledge base: {str(e)}"
        )