from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional

# Legacy schemas for backward compatibility
class QueryRequest(BaseModel):
    """Schema for basic query request"""
    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria for documents")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the main feature of the product?",
                "top_k": 3,
                "filter": {"source_type": "documentation"}
            }
        }
    )

class QueryResponseItem(BaseModel):
    """Schema for an individual query result"""
    text: str = Field(..., description="Text content of the result")
    metadata: Dict[str, Any] = Field(..., description="Metadata of the result")
    score: float = Field(..., description="Relevance score")

class QueryResponse(BaseModel):
    """Schema for basic query response"""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")

class RAGResponse(BaseModel):
    """Schema for RAG query response"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used for generation")

# Advanced schemas for production features
class AdvancedQueryRequest(BaseModel):
    """Schema for advanced query request with configurable options"""
    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")
    llm_model: Optional[str] = Field(None, description="LLM model to use for generation")
    enable_reranking: Optional[bool] = Field(None, description="Enable/disable reranking")
    enable_query_expansion: Optional[bool] = Field(None, description="Enable/disable query expansion")
    similarity_threshold: Optional[float] = Field(0.3, description="Minimum similarity threshold")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How does the system handle authentication?",
                "top_k": 10,
                "embedding_model": "text-embedding-3-large",
                "llm_model": "gpt-4-turbo-preview",
                "enable_reranking": True,
                "filters": {"document_type": "technical"}
            }
        }
    )

class QueryAnalysis(BaseModel):
    """Schema for query analysis results"""
    intent: str = Field(..., description="Detected query intent")
    keywords: List[str] = Field(..., description="Extracted keywords")
    entities: List[str] = Field(..., description="Extracted entities")
    expansion_terms: List[str] = Field(..., description="Query expansion terms")
    filters_applied: Dict[str, Any] = Field(..., description="Applied filters")

class SourceInfo(BaseModel):
    """Schema for enhanced source information"""
    index: int = Field(..., description="Source index")
    content: str = Field(..., description="Source content excerpt")
    metadata: Dict[str, Any] = Field(..., description="Source metadata")
    score: float = Field(..., description="Relevance score")
    node_id: str = Field(..., description="Unique node identifier")
    source_ref: str = Field(..., description="Human-readable source reference")

class AdvancedQueryResponse(BaseModel):
    """Schema for advanced query response"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(..., description="Enhanced source information")
    confidence_score: float = Field(..., description="Overall confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    query_analysis: QueryAnalysis = Field(..., description="Query analysis results")
    model_config_used: Dict[str, str] = Field(..., description="Models used for processing")
    metadata: Dict[str, Any] = Field(..., description="Additional processing metadata")

class BatchQueryRequest(BaseModel):
    """Schema for batch query request"""
    queries: List[str] = Field(..., description="List of queries to process")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")
    llm_model: Optional[str] = Field(None, description="LLM model to use")
    max_concurrent: Optional[int] = Field(3, description="Maximum concurrent queries")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queries": [
                    "What are the system requirements?",
                    "How do I configure authentication?",
                    "What are the supported file formats?"
                ],
                "embedding_model": "text-embedding-3-large",
                "max_concurrent": 3
            }
        }
    )

class BatchQueryResult(BaseModel):
    """Schema for individual batch query result"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source information")
    confidence_score: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time")

class BatchQueryResponse(BaseModel):
    """Schema for batch query response"""
    results: List[BatchQueryResult] = Field(..., description="Batch query results")
    total_queries: int = Field(..., description="Total number of queries")
    successful_queries: int = Field(..., description="Number of successful queries")
    model_config_used: Dict[str, str] = Field(..., description="Models used for processing")

class QueryExplanationResponse(BaseModel):
    """Schema for query explanation response"""
    query: str = Field(..., description="Original query")
    explanation: Dict[str, Any] = Field(..., description="Detailed processing explanation")
    kb_id: str = Field(..., description="Knowledge base ID")

class HybridSearchRequest(BaseModel):
    """Schema for hybrid search request"""
    query: str = Field(..., description="Search query")
    vector_weight: Optional[float] = Field(0.7, description="Vector search weight")
    keyword_weight: Optional[float] = Field(0.3, description="Keyword search weight")
    top_k: Optional[int] = Field(10, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class SemanticSearchRequest(BaseModel):
    """Schema for semantic search request"""
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(10, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(0.3, description="Minimum similarity score")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in results")

class ChunkOptimizationRequest(BaseModel):
    """Schema for chunk optimization request"""
    sample_texts: List[str] = Field(..., description="Sample texts for analysis")
    target_chunk_count: Optional[int] = Field(None, description="Target number of chunks")
    document_type: Optional[str] = Field(None, description="Type of documents")

class ChunkOptimizationResponse(BaseModel):
    """Schema for chunk optimization response"""
    recommended_chunk_size: int = Field(..., description="Recommended chunk size")
    recommended_overlap: int = Field(..., description="Recommended chunk overlap")
    chunking_strategy: str = Field(..., description="Recommended chunking strategy")
    reasoning: str = Field(..., description="Explanation of recommendations")
    estimated_chunks: int = Field(..., description="Estimated number of chunks")

class ModelPerformanceMetrics(BaseModel):
    """Schema for model performance metrics"""
    model_name: str = Field(..., description="Model name")
    average_response_time: float = Field(..., description="Average response time in seconds")
    average_confidence: float = Field(..., description="Average confidence score")
    total_queries: int = Field(..., description="Total queries processed")
    success_rate: float = Field(..., description="Success rate percentage")

class KnowledgeBaseStats(BaseModel):
    """Schema for knowledge base statistics"""
    kb_id: str = Field(..., description="Knowledge base ID")
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    average_chunk_size: float = Field(..., description="Average chunk size")
    embedding_model: str = Field(..., description="Current embedding model")
    last_updated: str = Field(..., description="Last update timestamp")
    query_count: int = Field(..., description="Total queries processed")
    average_query_time: float = Field(..., description="Average query processing time")

class SystemHealthResponse(BaseModel):
    """Schema for system health response"""
    status: str = Field(..., description="Overall system status")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    available_models: Dict[str, List[str]] = Field(..., description="Available models")
    configuration: Dict[str, Any] = Field(..., description="Current configuration")

class ProcessingPipelineStatus(BaseModel):
    """Schema for processing pipeline status"""
    document_id: str = Field(..., description="Document ID")
    pipeline_stage: str = Field(..., description="Current pipeline stage")
    progress_percentage: float = Field(..., description="Progress percentage")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")