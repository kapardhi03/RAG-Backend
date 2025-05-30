#schemas.knowledge_base.py
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import datetime
from uuid import UUID

class KnowledgeBaseBase(BaseModel):
    """Base schema for knowledge base data"""
    name: str = Field(..., description="Name of the knowledge base")
    description: Optional[str] = Field(None, description="Description of the knowledge base")

class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for creating a knowledge base"""
    pass

class KnowledgeBaseRead(KnowledgeBaseBase):
    """Schema for reading a knowledge base"""
    kb_id: str = Field(..., description="Unique identifier for the knowledge base")
    document_count: int = Field(0, description="Number of documents in the knowledge base")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True

class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a knowledge base"""
    name: Optional[str] = Field(None, description="Name of the knowledge base")
    description: Optional[str] = Field(None, description="Description of the knowledge base")

class KnowledgeBaseList(BaseModel):
    """Schema for list of knowledge bases"""
    items: List[KnowledgeBaseRead]
    total: int

class DocumentBase(BaseModel):
    """Base schema for document data"""
    name: str = Field(..., description="Name of the document or URL")
    type: str = Field(..., description="Type of document (file or url)")

class DocumentCreate(DocumentBase):
    """Schema for creating a document"""
    source_url: Optional[str] = Field(None, description="Source URL for URL documents")
    file_path: Optional[str] = Field(None, description="S3 path for file documents")

class DocumentRead(DocumentBase):
    """Schema for reading a document"""
    document_id: str = Field(..., description="Unique identifier for the document")
    status: str = Field(..., description="Status of document processing")
    created_at: datetime = Field(..., description="Creation timestamp")
    source_url: Optional[str] = Field(None, description="Source URL for URL documents")

    class Config:
        from_attributes = True

class URLSubmit(BaseModel):
    """Schema for submitting a URL for sitemap extraction"""
    url: str = Field(..., description="URL to extract sitemap from", example="https://example.com")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com"
                }
            ]
        }
    }

class URLSitemapResponse(BaseModel):
    """Schema for sitemap extraction response"""
    urls: List[str] = Field(..., description="List of URLs extracted from sitemap")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "urls": [
                        "https://example.com/about",
                        "https://example.com/services"
                    ]
                }
            ]
        }
    }

class URLAddRequest(BaseModel):
    """Schema for adding URLs to a knowledge base"""
    urls: List[str] = Field(..., description="List of URLs to add to knowledge base", 
                          example=["https://example.com/about", "https://example.com/services"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "urls": [
                        "https://example.com/about",
                        "https://example.com/services"
                    ]
                }
            ]
        }
    }

class URLAddResponse(BaseModel):
    """Schema for URL addition response"""
    added_documents: List[dict] = Field(..., description="List of added documents")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "added_documents": [
                        {
                            "document_id": "123e4567-e89b-12d3-a456-426614174000",
                            "url": "https://example.com/about",
                            "status": "pending"
                        }
                    ]
                }
            ]
        }
    }

class SearchResponse(BaseModel):
    """Schema for search response"""
    results: List[dict] = Field(..., description="Search results")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "chunk": "Example text from a document",
                            "document_name": "example.pdf",
                            "score": 0.92
                        }
                    ]
                }
            ]
        }
    }

class ChunkRead(BaseModel):
    """Schema for reading a chunk"""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Chunk text content")
    position: int = Field(..., description="Position in document")
    
    class Config:
        from_attributes = True