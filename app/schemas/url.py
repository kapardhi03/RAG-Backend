from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    """Enum for URL processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

class URLBase(BaseModel):
    """Base schema for URL data"""
    url: HttpUrl = Field(..., description="URL to process")
    kb_id: str = Field(..., description="ID of the knowledge base this URL belongs to")

class URLCreate(BaseModel):
    """Schema for submitting a URL for processing"""
    url: HttpUrl = Field(..., description="URL to process")

class URLRead(BaseModel):
    """Schema for reading a processed URL"""
    id: str = Field(..., description="Unique identifier for the URL")
    url: str = Field(..., description="The URL that was processed")
    kb_id: str = Field(..., description="ID of the knowledge base this URL belongs to")
    status: ProcessingStatus = Field(..., description="Processing status of the URL")
    chunk_count: int = Field(0, description="Number of text chunks created from the URL")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    title: Optional[str] = Field(None, description="Title extracted from the URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the URL")
    created_at: Optional[datetime] = Field(None, description="Submission timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")

    class Config:
        orm_mode = True

class URLList(BaseModel):
    """Schema for list of URLs"""
    items: List[URLRead]
    total: int

class URLChunk(BaseModel):
    """Schema for a URL content chunk"""
    text: str = Field(..., description="Text content of the chunk")
    url_id: str = Field(..., description="ID of the parent URL")
    chunk_index: int = Field(..., description="Index of this chunk within the URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the chunk")