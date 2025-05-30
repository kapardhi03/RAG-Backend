# app/api/endpoints/query.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
import logging
import re
import time

from app.schemas.query import QueryRequest, QueryResponse, RAGResponse
from app.services.embedding.openai import OpenAIEmbedding
from app.services.vector_store.chroma import ChromaVectorStore
from app.services.query.meta_handler import MetaQueryHandler
from app.api.deps import get_embedding_service, get_vector_store, get_current_tenant
from app.core.logging import log_rag_query

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/{kb_id}", response_model=QueryResponse)
async def query_knowledge_base(
    kb_id: str,
    query_request: QueryRequest,
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    current_tenant = Depends(get_current_tenant)
):
    """Query a knowledge base using semantic search"""
    try:
        # Check if knowledge base exists
        kb_exists = await vector_store.collection_exists(kb_id)
        if not kb_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Generate query embedding
        query_embedding = await embedding_service.embed_query(query_request.query)
        
        # Perform similarity search
        results = await vector_store.similarity_search(
            kb_id=kb_id,
            query_embedding=query_embedding,
            k=query_request.top_k if query_request.top_k else 5
        )
        
        return {
            "query": query_request.query,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query_knowledge_base: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge base: {str(e)}"
        )

def calculate_relevance_score(query: str, chunks: List[Dict], similarity_threshold: float = 0.3) -> Dict:
    """
    Calculate relevance scores and determine if the chunks are actually relevant to the query.
    Returns info about relevance and filtered chunks.
    """
    if not chunks:
        return {
            "has_relevant_content": False,
            "relevant_chunks": [],
            "avg_similarity": 0.0,
            "confidence": "none"
        }
    
    # Get similarity scores (ChromaDB returns distances, lower = more similar)
    # Convert distances to similarity scores (1 - distance)
    similarities = []
    for chunk in chunks:
        distance = chunk.get("score", 1.0)  # Default to max distance if not provided
        similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities)
    
    # Filter chunks by relevance threshold
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        if similarities[i] >= similarity_threshold:
            chunk["similarity"] = similarities[i]
            relevant_chunks.append(chunk)
    
    # Determine confidence level
    if avg_similarity >= 0.8:
        confidence = "high"
    elif avg_similarity >= 0.6:
        confidence = "medium"
    elif avg_similarity >= 0.4:
        confidence = "low"
    else:
        confidence = "very_low"
    
    has_relevant_content = len(relevant_chunks) > 0 and avg_similarity >= similarity_threshold
    
    return {
        "has_relevant_content": has_relevant_content,
        "relevant_chunks": relevant_chunks,
        "avg_similarity": avg_similarity,
        "confidence": confidence,
        "total_chunks": len(chunks),
        "relevant_chunk_count": len(relevant_chunks)
    }

def build_enhanced_context(query: str, chunks: List[Dict], max_context_length: int = 8000) -> str:
    """
    Build an enhanced context string with better organization and metadata.
    """
    if not chunks:
        return ""
    
    # Group chunks by document/source
    docs_by_source = {}
    for chunk in chunks:
        source = chunk["metadata"].get("source", "Unknown Document")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(chunk)
    
    context_parts = []
    current_length = 0
    
    # Add query context
    context_parts.append(f"User Query: {query}\n")
    context_parts.append("=" * 50)
    context_parts.append("")
    
    # Process each document
    for doc_idx, (source, doc_chunks) in enumerate(docs_by_source.items(), 1):
        # Clean source name for display
        if source.startswith(("http://", "https://")):
            display_source = f"Web Page: {source[:60]}..." if len(source) > 60 else f"Web Page: {source}"
        else:
            # Extract filename from path
            filename = source.split("/")[-1] if "/" in source else source
            display_source = f"Document: {filename}"
        
        doc_header = f"[SOURCE {doc_idx}] {display_source}"
        context_parts.append(doc_header)
        context_parts.append("-" * len(doc_header))
        
        # Sort chunks by similarity score (if available)
        doc_chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Add chunks with metadata
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk_text = chunk["text"].strip()
            
            # Add similarity info if available
            similarity_info = ""
            if "similarity" in chunk:
                similarity_info = f" (Relevance: {chunk['similarity']:.2f})"
            
            chunk_header = f"Section {chunk_idx + 1}{similarity_info}:"
            chunk_content = f"{chunk_header}\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed length limit
            if current_length + len(chunk_content) > max_context_length:
                context_parts.append("[... content truncated to fit context limit ...]")
                break
            
            context_parts.append(chunk_content)
            current_length += len(chunk_content)
        
        context_parts.append("")  # Add spacing between documents
        
        # Stop if we're approaching context limit
        if current_length > max_context_length * 0.9:
            break
    
    return "\n".join(context_parts)

def detect_query_intent(query: str) -> Dict[str, Any]:
    """
    Analyze the query to understand user intent and expectations.
    """
    query_lower = query.lower()
    
    intent_info = {
        "is_list_request": False,
        "is_summary_request": False,
        "is_specific_info": False,
        "is_comparison": False,
        "is_explanation": False,
        "is_file_query": False,
        "is_meta_query": False,  # Asking about the documents themselves
        "keywords": [],
        "question_type": "general"
    }
    
    # Detect list requests
    list_indicators = ["list", "what are", "show me", "give me all", "enumerate", "items", "things"]
    if any(indicator in query_lower for indicator in list_indicators):
        intent_info["is_list_request"] = True
        intent_info["question_type"] = "list"
    
    # Detect summary requests
    summary_indicators = ["summarize", "summary", "overview", "what is", "explain", "describe", "tell me about"]
    if any(indicator in query_lower for indicator in summary_indicators):
        intent_info["is_summary_request"] = True
        intent_info["question_type"] = "summary"
    
    # Detect meta queries (about the documents/context itself)
    meta_indicators = ["what documents", "what files", "what is in", "what content", "what information", "documents do you have", "what do you know"]
    if any(indicator in query_lower for indicator in meta_indicators):
        intent_info["is_meta_query"] = True
        intent_info["question_type"] = "meta"
    
    # Detect comparison requests
    comparison_indicators = ["compare", "difference", "vs", "versus", "better", "worse", "similarities", "contrast"]
    if any(indicator in query_lower for indicator in comparison_indicators):
        intent_info["is_comparison"] = True
        intent_info["question_type"] = "comparison"
    
    # Detect file-specific queries
    file_pattern = r'\b[\w.-]+\.[a-zA-Z0-9]+\b'
    if re.search(file_pattern, query):
        intent_info["is_file_query"] = True
        intent_info["question_type"] = "file_specific"
    
    # Extract key terms
    # Remove common stop words and focus on meaningful terms
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "how", "when", "where", "why"}
    words = re.findall(r'\b\w+\b', query_lower)
    intent_info["keywords"] = [word for word in words if word not in stop_words and len(word) > 2]
    
    return intent_info

@router.post("/{kb_id}/rag", response_model=RAGResponse)
async def rag_query(
    kb_id: str,
    query_request: QueryRequest,
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    current_tenant = Depends(get_current_tenant)
):
    """
    Perform enhanced RAG query with better context understanding and response quality.
    """
    start_time = time.time()
    
    try:
        # Check if knowledge base exists
        kb_exists = await vector_store.collection_exists(kb_id)
        if not kb_exists:
            log_rag_query(
                query_text=query_request.query,
                kb_id=kb_id,
                results_count=0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Analyze query intent
        intent_info = detect_query_intent(query_request.query)
        logger.info(f"Query intent analysis: {intent_info}")
        
        # Handle meta queries specially
        if intent_info["is_meta_query"]:
            meta_handler = MetaQueryHandler(vector_store)
            meta_info = await meta_handler.handle_meta_query(kb_id, query_request.query)
            
            if meta_info.get("has_content"):
                answer = meta_handler.generate_meta_response(meta_info, query_request.query)
                
                # Create sources from sample documents
                sources = []
                for sample in meta_info.get("samples", [])[:5]:
                    sources.append({
                        "source": sample.get("source", "Unknown"),
                        "text": sample.get("text", ""),
                        "score": 1.0,  # Meta queries always have high relevance
                        "similarity": 1.0
                    })
                
                # Log and return meta response
                total_time_ms = int((time.time() - start_time) * 1000)
                log_rag_query(
                    query_text=query_request.query,
                    kb_id=kb_id,
                    results_count=len(sources),
                    execution_time_ms=total_time_ms
                )
                
                return {
                    "query": query_request.query,
                    "answer": answer,
                    "sources": sources
                }
            else:
                return {
                    "query": query_request.query,
                    "answer": "This knowledge base appears to be empty. No documents or content have been added yet.",
                    "sources": []
                }
        
        # For non-meta queries, proceed with standard RAG
        # Generate query embedding
        embed_start_time = time.time()
        query_embedding = await embedding_service.embed_query(query_request.query)
        embed_time = time.time() - embed_start_time
        
        # Extract query terms for enhanced ranking
        query_terms = intent_info.get("keywords", [])
        
        # Determine search parameters based on intent
        search_k = query_request.top_k or 5
        if intent_info["is_list_request"]:
            search_k = min(search_k * 2, 15)  # Get more results for comprehensive answers
        elif intent_info["is_summary_request"]:
            search_k = min(search_k * 3, 20)  # Get even more for summaries
        
        # Perform enhanced similarity search
        search_start_time = time.time()
        results = await vector_store.similarity_search(
            kb_id=kb_id,
            query_embedding=query_embedding,
            k=search_k,
            query_terms=query_terms
        )
        search_time = time.time() - search_start_time
        
        logger.info(f"Retrieved {len(results)} chunks in {search_time:.2f}s")
        
        # Calculate relevance and filter results
        relevance_info = calculate_relevance_score(query_request.query, results)
        logger.info(f"Relevance analysis: {relevance_info}")
        
        # Use relevant chunks for context
        relevant_chunks = relevance_info["relevant_chunks"]
        if not relevant_chunks:
            # If no relevant chunks, check if we should return "null" or try with lower threshold
            if relevance_info["total_chunks"] == 0:
                return {
                    "query": query_request.query,
                    "answer": "I don't have any relevant information about that topic in the knowledge base.",
                    "sources": []
                }
            else:
                # Use top results but with low confidence
                relevant_chunks = results[:min(3, len(results))]
                relevance_info["has_relevant_content"] = False
        
        # Build enhanced context
        context = build_enhanced_context(query_request.query, relevant_chunks)
        
        # Generate response using enhanced prompting
        llm_start_time = time.time()
        answer = await embedding_service.generate_advanced_answer(
            query=query_request.query,
            context=context,
            intent_info=intent_info,
            relevance_info=relevance_info,
            kb_id=kb_id
        )
        llm_time = time.time() - llm_start_time
        
        # Check if the answer is "null" and handle appropriately
        if answer.strip().lower() == "null":
            if intent_info["is_file_query"]:
                answer = f"I don't have the specific file mentioned in your query in this knowledge base."
            else:
                answer = "I don't have specific information about that topic in the available documents."
        
        # Format sources for response
        sources = []
        for chunk in relevant_chunks[:query_request.top_k or 5]:
            source = chunk["metadata"].get("source", "Unknown source")
            text_preview = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            sources.append({
                "source": source,
                "text": text_preview,
                "score": chunk.get("score", 0.0),
                "similarity": chunk.get("similarity", 0.0)
            })
        
        # Calculate total execution time and log
        total_time = time.time() - start_time
        total_time_ms = int(total_time * 1000)
        
        log_rag_query(
            query_text=query_request.query,
            kb_id=kb_id,
            results_count=len(results),
            execution_time_ms=total_time_ms
        )
        
        logger.info(f"RAG query completed in {total_time:.2f}s (embedding: {embed_time:.2f}s, search: {search_time:.2f}s, LLM: {llm_time:.2f}s)")
        logger.info(f"Confidence: {relevance_info['confidence']}, Relevant chunks: {relevance_info['relevant_chunk_count']}/{relevance_info['total_chunks']}")
        
        return {
            "query": query_request.query,
            "answer": answer,
            "sources": sources
        }
        
    except HTTPException:
        execution_time = time.time() - start_time
        log_rag_query(
            query_text=query_request.query,
            kb_id=kb_id,
            results_count=0,
            execution_time_ms=int(execution_time * 1000)
        )
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in rag_query: {str(e)}", exc_info=True)
        
        log_rag_query(
            query_text=query_request.query,
            kb_id=kb_id,
            results_count=0,
            execution_time_ms=int(execution_time * 1000)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform RAG query: {str(e)}"
        )

@router.post("/{kb_id}/chat", response_model=RAGResponse)
async def chat_with_kb(
    kb_id: str,
    query_request: QueryRequest,
    embedding_service: OpenAIEmbedding = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    current_tenant = Depends(get_current_tenant)
):
    """
    Chat with a knowledge base using enhanced RAG.
    This is an alias for the RAG endpoint with a more intuitive name.
    """
    return await rag_query(kb_id, query_request, embedding_service, vector_store, current_tenant)