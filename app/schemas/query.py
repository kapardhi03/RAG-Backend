from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Schema for query request"""
    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria for documents")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the main feature of the product?",
                "top_k": 3,
                "filter": {"source_type": "documentation"}
            }
        }

class QueryResponseItem(BaseModel):
    """Schema for an individual query result"""
    text: str = Field(..., description="Text content of the result")
    metadata: Dict[str, Any] = Field(..., description="Metadata of the result")
    score: float = Field(..., description="Relevance score")

class QueryResponse(BaseModel):
    """Schema for query response"""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")

class RAGResponse(BaseModel):
    """Schema for RAG query response"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used for generation")