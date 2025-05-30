from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentBase(BaseModel):
    """Base model for document"""
    name: str = Field(..., description="Document name")
    type: str = Field(..., description="Document type (file or url)")

class DocumentCreate(BaseModel):
    """Schema for creating a document"""
    name: str = Field(..., description="Document name")

class DocumentUpload(BaseModel):
    """Schema for uploading a document"""
    kb_id: str = Field(..., description="Knowledge base ID")

class URLCreate(BaseModel):
    """Schema for creating a URL document"""
    url: str = Field(..., description="URL to process")

class DocumentInDB(DocumentBase):
    """Internal database model for document"""
    id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    tenant_id: str = Field(..., description="Tenant ID")
    source_url: Optional[str] = Field(None, description="Source URL (for URL documents)")
    file_path: Optional[str] = Field(None, description="File path (for file documents)")
    content_type: Optional[str] = Field(None, description="Content type")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    status: str = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    name: str = Field(..., description="Document name")
    type: str = Field(..., description="Document type (file or url)")
    status: str = Field(..., description="Document processing status")
    created_at: datetime = Field(..., description="Creation timestamp")

class DocumentDetail(DocumentResponse):
    """Schema for detailed document information"""
    source_url: Optional[str] = Field(None, description="Source URL (for URL documents)")
    file_path: Optional[str] = Field(None, description="File path (for file documents)")
    content_type: Optional[str] = Field(None, description="Content type")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    status: str = Field(..., description="Processing status")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    chunk_count: int = Field(0, description="Number of chunks")

class URLListResponse(BaseModel):
    """Schema for URL list response"""
    id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    url: str = Field(..., description="URL")
    status: str = Field(..., description="Processing status")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")

class DocumentStatus(BaseModel):
    """Schema for document status"""
    id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    chunks: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class ChunkBase(BaseModel):
    """Base model for document chunk"""
    text: str = Field(..., description="Chunk text")
    position: int = Field(..., description="Position in document")

class ChunkCreate(ChunkBase):
    """Schema for creating a chunk"""
    document_id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    tenant_id: str = Field(..., description="Tenant ID")

class ChunkInDB(ChunkBase):
    """Internal database model for chunk"""
    id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    tenant_id: str = Field(..., description="Tenant ID")
    vector_id: Optional[str] = Field(None, description="Vector ID in ChromaDB")
    created_at: datetime = Field(..., description="Creation timestamp")

class ChunkResponse(BaseModel):
    """Schema for document chunk"""
    id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    position: int = Field(..., description="Position in document")
    text: str = Field(..., description="Chunk text")
    vector_id: Optional[str] = Field(None, description="Vector ID in ChromaDB")
    created_at: datetime = Field(..., description="Creation timestamp")

class ReprocessRequest(BaseModel):
    """Schema for document reprocessing request"""
    force: bool = Field(False, description="Force reprocessing even if already processed")

class DocumentVectorStats(BaseModel):
    """Schema for document vector statistics"""
    document_id: str = Field(..., description="Document ID")
    total_chunks: int = Field(..., description="Total number of chunks")
    has_vectors: bool = Field(..., description="Whether document has vectors")
    vector_ids: List[str] = Field(..., description="Vector IDs")

class VectorStats(BaseModel):
    """Schema for knowledge base vector statistics"""
    kb_id: str = Field(..., description="Knowledge base ID")
    total_vectors: int = Field(..., description="Total number of vectors")
    total_documents: int = Field(..., description="Total number of documents")
    file_documents: int = Field(..., description="Number of file documents")
    url_documents: int = Field(..., description="Number of URL documents")
    avg_chunks_per_document: float = Field(..., description="Average chunks per document")

class ReindexResponse(BaseModel):
    """Schema for knowledge base reindexing response"""
    kb_id: str = Field(..., description="Knowledge base ID")
    documents_scheduled: int = Field(..., description="Number of documents scheduled for reindexing")
    file_documents: int = Field(..., description="Number of file documents")
    url_documents: int = Field(..., description="Number of URL documents")
    status: str = Field(..., description="Reindexing status")