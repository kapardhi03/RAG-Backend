# app/services/query/meta_handler.py - New file for handling meta queries
from typing import Dict, List, Any
import logging
from app.services.vector_store.chroma import ChromaVectorStore

logger = logging.getLogger("meta_query_handler")

class MetaQueryHandler:
    """
    Handles meta queries about the knowledge base itself
    (e.g., "What documents do you have?", "What can you tell me about?")
    """
    
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
    
    async def handle_meta_query(self, kb_id: str, query: str) -> Dict[str, Any]:
        """
        Handle meta queries about the knowledge base contents.
        Returns structured information about available documents and content.
        """
        try:
            logger.info(f"Handling meta query for KB {kb_id}: {query}")
            
            # Get comprehensive stats about the knowledge base
            try:
                stats = await self.vector_store.get_vector_stats(kb_id)
                logger.info(f"Got stats: {stats}")
            except Exception as e:
                logger.error(f"Error getting vector stats: {str(e)}")
                stats = {"total_vectors": 0, "unique_documents": 0}
            
            # Get sample documents for context
            try:
                samples = await self.vector_store.get_sample_documents(kb_id, limit=5)
                logger.info(f"Got {len(samples)} samples")
            except Exception as e:
                logger.error(f"Error getting samples: {str(e)}")
                samples = []
            
            # Analyze content types and topics
            try:
                content_analysis = await self._analyze_content_types(kb_id)
                logger.info(f"Content analysis complete: {len(content_analysis)} items")
            except Exception as e:
                logger.error(f"Error in content analysis: {str(e)}")
                content_analysis = {}
            
            return {
                "has_content": stats.get("total_vectors", 0) > 0,
                "stats": stats,
                "samples": samples,
                "content_analysis": content_analysis,
                "response_type": "meta_information"
            }
            
        except Exception as e:
            logger.error(f"Error handling meta query: {str(e)}")
            return {
                "has_content": False,
                "stats": {"total_vectors": 0, "unique_documents": 0},
                "samples": [],
                "content_analysis": {},
                "response_type": "error",
                "error": str(e)
            }
    
    async def _analyze_content_types(self, kb_id: str) -> Dict[str, Any]:
        """Analyze the types of content in the knowledge base"""
        try:
            if not await self.vector_store.collection_exists(kb_id):
                return {}
            
            collection = self.vector_store.client.get_collection(name=kb_id)
            
            # Get all metadata to analyze content types
            result = collection.get(include=["metadatas"])
            
            if not result["metadatas"]:
                return {}
            
            # Analyze sources and content types
            sources_by_type = {"files": [], "urls": [], "unknown": []}
            file_extensions = {}
            domains = set()
            
            for metadata in result["metadatas"]:
                if not metadata or "source" not in metadata:
                    continue
                
                source = metadata["source"]
                
                if source.startswith(("http://", "https://")):
                    sources_by_type["urls"].append(source)
                    # Extract domain
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source).netloc
                        domains.add(domain)
                    except:
                        pass
                elif "." in source:
                    sources_by_type["files"].append(source)
                    # Extract file extension
                    ext = source.split(".")[-1].lower()
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                else:
                    sources_by_type["unknown"].append(source)
            
            # Remove duplicates and ensure we have lists
            for key in sources_by_type:
                if sources_by_type[key]:  # Only process if not empty
                    sources_by_type[key] = list(set(sources_by_type[key]))
            
            return {
                "sources_by_type": sources_by_type,
                "file_extensions": file_extensions,
                "domains": list(domains),
                "total_files": len(sources_by_type["files"]),
                "total_urls": len(sources_by_type["urls"]),
                "total_unknown": len(sources_by_type["unknown"])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content types: {str(e)}")
            return {}
    
    def generate_meta_response(self, meta_info: Dict[str, Any], query: str) -> str:
        """
        Generate a human-readable response about the knowledge base contents.
        """
        if not meta_info.get("has_content"):
            return "This knowledge base appears to be empty. No documents or content have been added yet."
        
        stats = meta_info.get("stats", {})
        content_analysis = meta_info.get("content_analysis", {})
        samples = meta_info.get("samples", [])
        
        response_parts = []
        
        # Overview
        total_docs = stats.get("unique_documents", 0)
        total_chunks = stats.get("total_vectors", 0)
        
        response_parts.append(f"## Knowledge Base Overview")
        response_parts.append(f"This knowledge base contains **{total_docs} documents** broken down into **{total_chunks} searchable sections**.")
        
        # Content breakdown
        file_count = content_analysis.get("total_files", 0)
        url_count = content_analysis.get("total_urls", 0)
        
        if file_count > 0 or url_count > 0:
            response_parts.append(f"\n## Content Sources")
            if file_count > 0:
                response_parts.append(f"- **{file_count} uploaded files**")
                
                # Show file types
                file_exts = content_analysis.get("file_extensions", {})
                if file_exts:
                    ext_summary = ", ".join([f"{ext.upper()} ({count})" for ext, count in file_exts.items()])
                    response_parts.append(f"  - File types: {ext_summary}")
            
            if url_count > 0:
                response_parts.append(f"- **{url_count} web pages/URLs**")
                
                # Show domains
                domains = content_analysis.get("domains", [])
                if domains:
                    domain_summary = ", ".join(domains[:5])
                    if len(domains) > 5:
                        domain_summary += f" (and {len(domains) - 5} more)"
                    response_parts.append(f"  - From domains: {domain_summary}")
        
        # Sample content
        if samples:
            response_parts.append(f"\n## Available Content Samples")
            for i, sample in enumerate(samples[:3], 1):
                source = sample.get("source", "Unknown")
                text_preview = sample.get("text", "")[:150]
                
                # Clean up source name for display
                if source.startswith(("http://", "https://")):
                    display_source = f"Web: {source[:50]}..." if len(source) > 50 else f"Web: {source}"
                else:
                    filename = source.split("/")[-1] if "/" in source else source
                    display_source = f"File: {filename}"
                
                response_parts.append(f"**{i}. {display_source}**")
                response_parts.append(f"   {text_preview}...")
        
        # What can be asked
        response_parts.append(f"\n## What You Can Ask")
        response_parts.append("You can ask me questions about:")
        
        if file_count > 0:
            response_parts.append("- Content from the uploaded documents")
            response_parts.append("- Specific information within files")
        
        if url_count > 0:
            response_parts.append("- Information from the web pages")
            response_parts.append("- Details from online content")
        
        response_parts.append("- Summaries of any topic covered in the documents")
        response_parts.append("- Comparisons between different sources")
        response_parts.append("- Lists of specific items or concepts")
        
        # Query suggestions based on content
        if "requirements" in query.lower() or any("requirements.txt" in str(s.get("source", "")) for s in samples):
            response_parts.append("\n*For example: 'What are the main dependencies?' or 'List all the Python packages required'*")
        elif any("http" in str(s.get("source", "")) for s in samples):
            response_parts.append("\n*For example: 'Summarize the main points' or 'What are the key features mentioned?'*")
        else:
            response_parts.append("\n*For example: 'Summarize the main topics' or 'What are the key points covered?'*")
        
        return "\n".join(response_parts)