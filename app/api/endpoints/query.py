# app/api/endpoints/query.py - Updated with advanced query capabilities
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
import logging

from app.schemas.query import (
    QueryRequest, 
    QueryResponse, 
    RAGResponse,
    AdvancedQueryRequest,
    AdvancedQueryResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    QueryExplanationResponse
)
from app.services.query.engine import AdvancedQueryEngine
from app.services.llamaindex.engine import LlamaIndexRAGEngine
from app.api.deps import (
    get_query_engine, 
    get_rag_engine, 
    get_current_tenant,
    validate_model_selection,
    create_configured_rag_engine,
    create_configured_query_engine
)
from app.core.logging import log_rag_query

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/{kb_id}/advanced", response_model=AdvancedQueryResponse)
async def advanced_query(
    kb_id: str,
    query_request: AdvancedQueryRequest,
    current_tenant = Depends(get_current_tenant)
):
    """Execute advanced query with configurable models and enhanced features"""
    
    start_time = time.time()
    
    try:
        # Validate model selection if provided
        model_config = validate_model_selection(
            query_request.embedding_model,
            query_request.llm_model
        )
        
        # Get or create RAG engine with specified models
        if (query_request.embedding_model or query_request.llm_model):
            rag_engine = create_configured_rag_engine(
                model_config["embedding_model"],
                model_config["llm_model"]
            )
            query_engine = create_configured_query_engine(rag_engine)
        else:
            query_engine = get_query_engine()
        
        # Execute advanced query
        result = await query_engine.execute_advanced_query(
            query=query_request.query,
            kb_id=kb_id,
            top_k=query_request.top_k,
            filters=query_request.filters,
            enable_reranking=query_request.enable_reranking
        )
        
        # Log query
        log_rag_query(
            query_text=query_request.query,
            kb_id=kb_id,
            results_count=len(result.sources),
            execution_time_ms=int(result.processing_time * 1000)
        )
        
        # Format response
        return AdvancedQueryResponse(
            query=result.query_context.query,
            answer=result.answer,
            sources=result.sources,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            query_analysis={
                "intent": result.query_context.intent,
                "keywords": result.query_context.keywords,
                "entities": result.query_context.entities,
                "expansion_terms": result.query_context.expansion_terms,
                "filters_applied": result.query_context.filters
            },
            model_config=model_config,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Advanced query failed: {e}", exc_info=True)
        
        log_rag_query(
            query_text=query_request.query,
            kb_id=kb_id,
            results_count=0,
            execution_time_ms=int(execution_time * 1000)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced query failed: {str(e)}"
        )

@router.post("/{kb_id}/batch", response_model=BatchQueryResponse)
async def batch_query(
    kb_id: str,
    batch_request: BatchQueryRequest,
    current_tenant = Depends(get_current_tenant)
):
    """Execute multiple queries concurrently"""
    
    try:
        # Validate model selection
        model_config = validate_model_selection(
            batch_request.embedding_model,
            batch_request.llm_model
        )
        
        # Get query engine
        if (batch_request.embedding_model or batch_request.llm_model):
            rag_engine = create_configured_rag_engine(
                model_config["embedding_model"],
                model_config["llm_model"]
            )
            query_engine = create_configured_query_engine(rag_engine)
        else:
            query_engine = get_query_engine()
        
        # Execute batch queries
        results = await query_engine.batch_query(
            queries=batch_request.queries,
            kb_id=kb_id,
            max_concurrent=batch_request.max_concurrent
        )
        
        # Format results
        query_results = []
        for result in results:
            query_results.append({
                "query": result.query_context.query,
                "answer": result.answer,
                "sources": result.sources[:5],  # Limit sources for batch response
                "confidence_score": result.confidence_score,
                "processing_time": result.processing_time
            })
        
        # Log batch operation
        for result in results:
            log_rag_query(
                query_text=result.query_context.query,
                kb_id=kb_id,
                results_count=len(result.sources),
                execution_time_ms=int(result.processing_time * 1000)
            )
        
        return BatchQueryResponse(
            results=query_results,
            total_queries=len(batch_request.queries),
            successful_queries=len([r for r in results if r.confidence_score > 0.3]),
            model_config=model_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query failed: {str(e)}"
        )

@router.post("/{kb_id}/explain", response_model=QueryExplanationResponse)
async def explain_query(
    kb_id: str,
    query: str = Query(..., description="Query to explain"),
    current_tenant = Depends(get_current_tenant)
):
    """Explain how a query would be processed"""
    
    try:
        query_engine = get_query_engine()
        
        explanation = await query_engine.explain_query_processing(query, kb_id)
        
        return QueryExplanationResponse(
            query=query,
            explanation=explanation,
            kb_id=kb_id
        )
        
    except Exception as e:
        logger.error(f"Query explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query explanation failed: {str(e)}"
        )

@router.get("/{kb_id}/semantic-search", response_model=QueryResponse)
async def semantic_search(
    kb_id: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return"),
    similarity_threshold: float = Query(0.3, description="Minimum similarity score"),
    embedding_model: Optional[str] = Query(None, description="Embedding model to use"),
    current_tenant = Depends(get_current_tenant)
):
    """Perform semantic search without LLM generation"""
    
    try:
        # Validate and get RAG engine
        if embedding_model:
            model_config = validate_model_selection(embedding_model=embedding_model)
            rag_engine = create_configured_rag_engine(
                model_config["embedding_model"],
                model_config["llm_model"]
            )
        else:
            rag_engine = get_rag_engine()
        
        # Perform semantic search
        results = await rag_engine.semantic_search(
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Log search
        log_rag_query(
            query_text=query,
            kb_id=kb_id,
            results_count=len(results),
            execution_time_ms=0  # Semantic search timing would need to be added
        )
        
        return QueryResponse(
            query=query,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )

@router.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """Get available embedding and LLM models"""
    
    from app.api.deps import get_available_models
    return get_available_models()

@router.get("/{kb_id}/stats", response_model=Dict[str, Any])
async def get_query_stats(
    kb_id: str,
    current_tenant = Depends(get_current_tenant)
):
    """Get query engine statistics for a knowledge base"""
    
    try:
        query_engine = get_query_engine()
        rag_engine = get_rag_engine()
        
        # Get engine stats
        engine_stats = query_engine.get_engine_stats()
        
        # Get knowledge base specific stats
        kb_stats = await rag_engine.get_index_stats(kb_id)
        
        return {
            "kb_id": kb_id,
            "engine_config": engine_stats,
            "kb_stats": kb_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get query stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get query stats: {str(e)}"
        )

# Legacy endpoints for backward compatibility
@router.post("/{kb_id}", response_model=QueryResponse)
async def query_knowledge_base(
    kb_id: str,
    query_request: QueryRequest,
    current_tenant = Depends(get_current_tenant)
):
    """Legacy semantic search endpoint"""
    
    try:
        rag_engine = get_rag_engine()
        
        # Generate query embedding
        query_embedding = await rag_engine.embed_model.aget_query_embedding(query_request.query)
        
        # Perform similarity search
        results = await rag_engine.semantic_search(
            kb_id=kb_id,
            query=query_request.query,
            top_k=query_request.top_k or 5
        )
        
        return QueryResponse(
            query=query_request.query,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Legacy query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.post("/{kb_id}/rag", response_model=RAGResponse)
async def rag_query(
    kb_id: str,
    query_request: QueryRequest,
    current_tenant = Depends(get_current_tenant)
):
    """Legacy RAG endpoint - redirects to advanced query"""
    
    # Convert to advanced query request
    advanced_request = AdvancedQueryRequest(
        query=query_request.query,
        top_k=query_request.top_k,
        filters=query_request.filter
    )
    
    # Use advanced query
    result = await advanced_query(kb_id, advanced_request, current_tenant)
    
    # Convert to legacy format
    return RAGResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources
    )

@router.post("/{kb_id}/chat", response_model=RAGResponse)
async def chat_with_kb(
    kb_id: str,
    query_request: QueryRequest,
    current_tenant = Depends(get_current_tenant)
):
    """Legacy chat endpoint - redirects to advanced query"""
    return await rag_query(kb_id, query_request, current_tenant)