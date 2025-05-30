# app/services/vector_store/chroma.py
import os
import uuid
import chromadb
import logging
import traceback
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger("vector_store")

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        logger.info(f"ChromaDB Vector Store initialized with directory: {persist_directory}")
    
    async def collection_exists(self, kb_id: str) -> bool:
        """Check if a collection exists for the given knowledge base ID"""
        try:
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            exists = kb_id in collection_names
            logger.info(f"Checking if collection '{kb_id}' exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            raise
    
    async def create_collection(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new collection (knowledge base)"""
        try:
            logger.info(f"Creating new collection '{name}'")
            
            metadata = {
                "description": description,
                "created": str(datetime.now())
            }
            
            self.client.create_collection(
                name=name,
                metadata=metadata
            )
            
            logger.info(f"Successfully created collection '{name}'")
            
            return {
                "id": name,
                "name": name,
                "description": description,
                "document_count": 0
            }
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {str(e)}")
            raise
    
    async def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        try:
            logger.info(f"Getting collection '{collection_name}'")
            
            if not await self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' not found")
                return None
            
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            
            logger.info(f"Found collection '{collection_name}' with {count} vectors")
            
            return {
                "id": collection_name,
                "name": collection_name,
                "description": collection.metadata.get("description", "") if collection.metadata else "",
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection '{collection_name}': {str(e)}")
            raise
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        try:
            logger.info("Listing all collections")
            
            collections = self.client.list_collections()
            result = []
            
            for collection in collections:
                chroma_collection = self.client.get_collection(name=collection.name)
                result.append({
                    "id": collection.name,
                    "name": collection.name,
                    "description": collection.metadata.get("description", "") if collection.metadata else "",
                    "document_count": chroma_collection.count()
                })
            
            logger.info(f"Found {len(result)} collections")
            return result
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        try:
            logger.info(f"Deleting collection '{collection_name}'")
            
            if await self.collection_exists(collection_name):
                self.client.delete_collection(name=collection_name)
                logger.info(f"Successfully deleted collection '{collection_name}'")
            else:
                logger.warning(f"Collection '{collection_name}' not found, nothing to delete")
                
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            raise
    
    def _calculate_similarity_score(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score.
        ChromaDB uses cosine distance (0 = identical, 2 = opposite)
        Convert to similarity score (1 = identical, 0 = completely different)
        """
        # For cosine distance: similarity = 1 - (distance / 2)
        # Clamp between 0 and 1
        similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        return similarity
    
    def _rank_results_by_relevance(self, results: List[Dict[str, Any]], query_terms: List[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced ranking of results considering both semantic similarity and keyword matching.
        """
        if not results:
            return results
        
        # Add keyword matching scores if query terms provided
        if query_terms:
            for result in results:
                text = result.get("text", "").lower()
                keyword_score = 0
                
                for term in query_terms:
                    if term.lower() in text:
                        # Boost score based on term frequency and position
                        term_count = text.count(term.lower())
                        keyword_score += term_count * 0.1
                        
                        # Boost if term appears early in text
                        position = text.find(term.lower())
                        if position < 100:  # First 100 characters
                            keyword_score += 0.05
                
                result["keyword_score"] = min(keyword_score, 0.5)  # Cap at 0.5
        
        # Combine similarity and keyword scores
        for result in results:
            similarity = result.get("similarity", 0)
            keyword_score = result.get("keyword_score", 0)
            combined_score = similarity + keyword_score
            result["combined_score"] = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.get("combined_score", x.get("similarity", 0)), reverse=True)
        
        return results
    
    async def similarity_search(
        self, 
        kb_id: str, 
        query_embedding: List[float],
        k: int = 5,
        query_terms: List[str] = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced similarity search with better scoring and ranking.
        """
        try:
            logger.info(f"Performing similarity search in collection '{kb_id}' with k={k}")
            
            if not await self.collection_exists(kb_id):
                logger.warning(f"Collection '{kb_id}' not found")
                raise ValueError(f"Collection {kb_id} does not exist")
            
            collection = self.client.get_collection(name=kb_id)
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(k * 2, 50),  # Get more results initially for better filtering
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add metadata filtering if provided
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            # Perform the search
            results = collection.query(**query_params)
            
            if not results["documents"] or not results["documents"][0]:
                logger.info("No results found for similarity search")
                return []
            
            # Format and enhance results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if "distances" in results else 0.0
                similarity = self._calculate_similarity_score(distance)
                
                result = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "distance": distance,
                    "similarity": similarity,
                    "score": similarity  # For backward compatibility
                }
                
                formatted_results.append(result)
            
            # Enhanced ranking
            formatted_results = self._rank_results_by_relevance(formatted_results, query_terms)
            
            # Filter by minimum similarity threshold
            min_similarity = 0.1  # Very low threshold to avoid filtering too aggressively
            filtered_results = [r for r in formatted_results if r["similarity"] >= min_similarity]
            
            # Return top k results
            final_results = filtered_results[:k]
            
            logger.info(f"Similarity search completed: {len(final_results)} results returned")
            logger.info(f"Score range: {final_results[0]['similarity']:.3f} - {final_results[-1]['similarity']:.3f}" if final_results else "No results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def get_vector_stats(self, kb_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics about vectors in a knowledge base"""
        try:
            logger.info(f"Getting vector stats for collection '{kb_id}'")
            
            if not await self.collection_exists(kb_id):
                logger.warning(f"Collection '{kb_id}' not found")
                raise ValueError(f"Collection {kb_id} does not exist")
            
            collection = self.client.get_collection(name=kb_id)
            
            # Get all items to analyze
            result = collection.get(include=["documents", "metadatas", "embeddings"])
            
            if not result["ids"]:
                return {
                    "kb_id": kb_id,
                    "total_vectors": 0,
                    "unique_documents": 0,
                    "file_vectors": 0,
                    "url_vectors": 0,
                    "embedding_dimension": 0,
                    "avg_document_length": 0,
                    "sources": []
                }
            
            # Count total vectors
            total_vectors = len(result["ids"])
            
            # Analyze documents and sources
            doc_ids = set()
            sources = set()
            file_count = 0
            url_count = 0
            document_lengths = []
            
            for i, metadata in enumerate(result["metadatas"]):
                if metadata:
                    # Track unique documents
                    if "doc_id" in metadata:
                        doc_ids.add(metadata["doc_id"])
                    
                    # Track sources
                    if "source" in metadata:
                        source = metadata["source"]
                        sources.add(source)
                        
                        # Classify as file or URL
                        if source.startswith(("http://", "https://")):
                            url_count += 1
                        else:
                            file_count += 1
                
                # Track document lengths
                if result["documents"] and i < len(result["documents"]):
                    document_lengths.append(len(result["documents"][i]))
            
            # Calculate statistics
            avg_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0
            embedding_dim = len(result["embeddings"][0]) if result["embeddings"] and result["embeddings"][0] else 0
            
            stats = {
                "kb_id": kb_id,
                "total_vectors": total_vectors,
                "unique_documents": len(doc_ids),
                "file_vectors": file_count,
                "url_vectors": url_count,
                "embedding_dimension": embedding_dim,
                "avg_document_length": round(avg_length, 2),
                "sources": list(sources)[:10]  # Limit to first 10 sources
            }
            
            logger.info(f"Vector stats calculated: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector stats: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def get_sample_documents(self, kb_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample documents from the knowledge base for context understanding"""
        try:
            if not await self.collection_exists(kb_id):
                return []
            
            collection = self.client.get_collection(name=kb_id)
            result = collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            samples = []
            for i in range(len(result["documents"])):
                samples.append({
                    "text": result["documents"][i][:200] + "..." if len(result["documents"][i]) > 200 else result["documents"][i],
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                    "source": result["metadatas"][i].get("source", "Unknown") if result["metadatas"] else "Unknown"
                })
            
            return samples
            
        except Exception as e:
            logger.error(f"Error getting sample documents: {str(e)}")
            return []