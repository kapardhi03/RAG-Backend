from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ProcessingStatusResponse(BaseModel):
    """Schema for document processing status response"""
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    job_status: Optional[str] = Field(None, description="Status of background job")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    chunk_count: int = Field(0, description="Number of text chunks")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    filename: str = Field(..., description="Original filename")
    file_type: Optional[str] = Field(None, description="File MIME type")
    file_size: Optional[int] = Field(None, description="File size in bytes")

class QueueStatusResponse(BaseModel):
    """Schema for queue status"""
    queued: int = Field(0, description="Number of jobs in queue")
    active: int = Field(0, description="Number of active jobs")
    failed: int = Field(0, description="Number of failed jobs")

class DocStatusCounts(BaseModel):
    """Schema for document status counts"""
    pending: int = Field(0, description="Number of pending documents") 
    processing: int = Field(0, description="Number of processing documents")
    processed: int = Field(0, description="Number of processed documents")
    failed: int = Field(0, description="Number of failed documents")

class SystemStatusResponse(BaseModel):
    """Schema for system status response"""
    document_queue: QueueStatusResponse = Field(..., description="Document processing queue status")
    url_queue: QueueStatusResponse = Field(..., description="URL processing queue status")
    document_counts: DocStatusCounts = Field(..., description="Document counts by status")
    system_status: str = Field("operational", description="Overall system status")
    api_version: str = Field("1.0", description="API version")