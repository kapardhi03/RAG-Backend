import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tiktoken

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    ServiceContext,
    Document as LlamaDocument,
    Settings
)
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter
)
from llama_index.core.text_splitter import SentenceSplitter as CoreSentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

class LlamaIndexRAGEngine:
    """Production-grade RAG engine using LlamaIndex with advanced features"""
    
    def __init__(
        self,
        embedding_model: str = None,
        llm_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic_chunking: bool = None,
        enable_hybrid_search: bool = None,
        enable_reranking: bool = None
    ):
        """Initialize the RAG engine with configurable parameters"""
        
        # Set default values from config
        self.embedding_model = embedding_model or settings.DEFAULT_EMBEDDING_MODEL
        self.llm_model = llm_model or settings.DEFAULT_LLM_MODEL
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        self.enable_semantic_chunking = enable_semantic_chunking if enable_semantic_chunking is not None else settings.ENABLE_SEMANTIC_CHUNKING
        self.enable_hybrid_search = enable_hybrid_search if enable_hybrid_search is not None else settings.ENABLE_HYBRID_SEARCH
        self.enable_reranking = enable_reranking if enable_reranking is not None else settings.ENABLE_RERANKING
        
        # Initialize tokenizer for token-based operations
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.llm_model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_store()
        self._initialize_node_parsers()
        self._initialize_postprocessors()
        
        # Storage for indices by knowledge base
        self.indices: Dict[str, VectorStoreIndex] = {}
        self.query_engines: Dict[str, RetrieverQueryEngine] = {}
        
        logger.info(f"LlamaIndex RAG Engine initialized with embedding_model={self.embedding_model}, llm_model={self.llm_model}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model with fallback support"""
        try:
            # Validate model availability
            if self.embedding_model not in settings.AVAILABLE_EMBEDDING_MODELS:
                logger.warning(f"Embedding model {self.embedding_model} not in available models, using default")
                self.embedding_model = settings.DEFAULT_EMBEDDING_MODEL
            
            model_config = settings.AVAILABLE_EMBEDDING_MODELS[self.embedding_model]
            
            self.embed_model = OpenAIEmbedding(
                model=self.embedding_model,
                dimensions=model_config.get("dimensions"),
                api_key=settings.OPENAI_API_KEY
            )
            
            # Set global embedding model
            Settings.embed_model = self.embed_model
            
            logger.info(f"Embedding model initialized: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.embedding_model}: {e}")
            # Fallback to default model
            self.embedding_model = settings.FALLBACK_EMBEDDING_MODEL
            self.embed_model = OpenAIEmbedding(
                model=self.embedding_model,
                api_key=settings.OPENAI_API_KEY
            )
            Settings.embed_model = self.embed_model
            logger.info(f"Using fallback embedding model: {self.embedding_model}")
    
    def _initialize_llm(self):
        """Initialize LLM with configuration validation"""
        try:
            # Validate model availability
            if self.llm_model not in settings.AVAILABLE_LLM_MODELS:
                logger.warning(f"LLM model {self.llm_model} not in available models, using default")
                self.llm_model = settings.DEFAULT_LLM_MODEL
            
            model_config = settings.AVAILABLE_LLM_MODELS[self.llm_model]
            
            self.llm = OpenAI(
                model=self.llm_model,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=model_config.get("max_tokens", 4096),
                temperature=0.1
            )
            
            # Set global LLM
            Settings.llm = self.llm
            
            logger.info(f"LLM initialized: {self.llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM {self.llm_model}: {e}")
            # Fallback to default model
            self.llm_model = settings.FALLBACK_LLM_MODEL
            self.llm = OpenAI(
                model=self.llm_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1
            )
            Settings.llm = self.llm
            logger.info(f"Using fallback LLM: {self.llm_model}")
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            logger.info(f"ChromaDB initialized at {settings.CHROMA_PERSIST_DIRECTORY}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_node_parsers(self):
        """Initialize advanced node parsers for different chunking strategies"""
        
        if self.enable_semantic_chunking:
            # Semantic chunking using embeddings
            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model
            )
            logger.info("Semantic splitter initialized")
        
        # Token-based splitter for precise token management
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
            backup_separators=["\n", "\t"]
        )
        
        # Sentence-based splitter for maintaining sentence integrity
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
        )
        
        # Title and keyword extractors for metadata enrichment
        self.title_extractor = TitleExtractor(
            nodes=5,
            llm=self.llm
        )
        
        self.keyword_extractor = KeywordExtractor(
            keywords=10,
            llm=self.llm
        )
        
        logger.info("Node parsers initialized")
    
    def _initialize_postprocessors(self):
        """Initialize postprocessors for retrieval refinement"""
        self.postprocessors = []
        
        # Similarity threshold postprocessor
        self.similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.3
        )
        self.postprocessors.append(self.similarity_postprocessor)
        
        # Keyword-based postprocessor for hybrid search
        if self.enable_hybrid_search:
            self.keyword_postprocessor = KeywordNodePostprocessor(
                required_keywords=[],  # Will be set dynamically per query
                exclude_keywords=[]
            )
        
        # Cohere reranker for advanced relevance scoring
        if self.enable_reranking and settings.COHERE_API_KEY:
            try:
                self.reranker = CohereRerank(
                    api_key=settings.COHERE_API_KEY,
                    top_n=settings.FINAL_TOP_K,
                    model="rerank-english-v2.0"
                )
                self.postprocessors.append(self.reranker)
                logger.info("Cohere reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere reranker: {e}")
                self.enable_reranking = False
        
        logger.info(f"Postprocessors initialized: {len(self.postprocessors)} processors")
    
    def _select_optimal_chunking_strategy(self, documents: List[LlamaDocument]) -> Any:
        """Select the optimal chunking strategy based on document characteristics"""
        
        # Analyze document characteristics
        total_length = sum(len(doc.text) for doc in documents)
        avg_length = total_length / len(documents) if documents else 0
        
        # For long documents, use semantic chunking if enabled
        if self.enable_semantic_chunking and avg_length > 5000:
            logger.info("Using semantic chunking for long documents")
            return self.semantic_splitter
        
        # For medium documents, use sentence-based chunking
        elif avg_length > 1000:
            logger.info("Using sentence-based chunking for medium documents")
            return self.sentence_splitter
        
        # For short documents or high precision needs, use token-based chunking
        else:
            logger.info("Using token-based chunking for short documents")
            return self.token_splitter
    
    async def create_index(self, kb_id: str, documents: List[LlamaDocument]) -> VectorStoreIndex:
        """Create a new index for a knowledge base with advanced processing"""
        
        try:
            logger.info(f"Creating index for KB {kb_id} with {len(documents)} documents")
            
            # Get or create ChromaDB collection
            try:
                collection = self.chroma_client.get_collection(name=kb_id)
                logger.info(f"Using existing collection {kb_id}")
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=kb_id,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection {kb_id}")
            
            # Initialize vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Select optimal chunking strategy
            node_parser = self._select_optimal_chunking_strategy(documents)
            
            # Create ingestion pipeline with extractors
            transformations = [node_parser]
            
            # Add metadata extractors for enrichment
            if len(documents) > 1:  # Only for multiple documents to avoid overhead
                transformations.extend([
                    self.title_extractor,
                    self.keyword_extractor
                ])
            
            # Create ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=transformations,
                vector_store=vector_store
            )
            
            # Process documents through pipeline
            nodes = await asyncio.to_thread(pipeline.run, documents=documents)
            
            # Create index
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Create and configure query engine
            query_engine = self._create_query_engine(index, kb_id)
            
            # Store references
            self.indices[kb_id] = index
            self.query_engines[kb_id] = query_engine
            
            logger.info(f"Index created successfully for KB {kb_id} with {len(nodes)} nodes")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index for KB {kb_id}: {e}")
            raise
    
    def _create_query_engine(self, index: VectorStoreIndex, kb_id: str) -> RetrieverQueryEngine:
        """Create a configured query engine with advanced retrieval"""
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=settings.RERANK_TOP_K if self.enable_reranking else settings.FINAL_TOP_K,
            embed_model=self.embed_model
        )
        
        # Create query engine with postprocessors
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=self.postprocessors,
            response_synthesizer=index.as_query_engine().response_synthesizer
        )
        
        return query_engine
    
    async def query(
        self, 
        kb_id: str, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        enable_query_expansion: bool = None
    ) -> Dict[str, Any]:
        """Execute an advanced query with hybrid search and reranking"""
        
        try:
            if kb_id not in self.query_engines:
                raise ValueError(f"Knowledge base {kb_id} not found")
            
            query_engine = self.query_engines[kb_id]
            
            # Query expansion if enabled
            if enable_query_expansion or (enable_query_expansion is None and settings.ENABLE_QUERY_EXPANSION):
                expanded_query = await self._expand_query(query)
                logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
                query = expanded_query
            
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Apply filters if provided
            if filters:
                # Update retriever with filters
                retriever = query_engine.retriever
                if hasattr(retriever, '_similarity_top_k'):
                    # This is a way to pass filters in LlamaIndex
                    retriever._vector_store_kwargs = {"where": filters}
            
            # Execute query
            response = await asyncio.to_thread(query_engine.query, query_bundle)
            
            # Format response
            result = {
                "answer": str(response),
                "source_nodes": [
                    {
                        "content": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                        "metadata": node.node.metadata,
                        "score": node.score,
                        "node_id": node.node.node_id
                    }
                    for node in response.source_nodes[:top_k or settings.FINAL_TOP_K]
                ],
                "query": query,
                "kb_id": kb_id
            }
            
            logger.info(f"Query executed successfully for KB {kb_id}, returned {len(result['source_nodes'])} sources")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute query for KB {kb_id}: {e}")
            raise
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        try:
            expansion_prompt = f"""
            Given the user query: "{query}"
            
            Generate an expanded version that includes synonyms and related terms to improve search recall.
            Keep the expansion concise and relevant. Return only the expanded query without explanation.
            
            Expanded query:
            """
            
            response = await asyncio.to_thread(
                self.llm.complete,
                expansion_prompt
            )
            
            expanded = str(response).strip()
            
            # Fallback to original query if expansion fails
            if not expanded or len(expanded) > len(query) * 3:
                return query
                
            return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    async def update_index(self, kb_id: str, new_documents: List[LlamaDocument]) -> bool:
        """Update an existing index with new documents"""
        try:
            if kb_id not in self.indices:
                logger.warning(f"Index for KB {kb_id} not found, creating new index")
                await self.create_index(kb_id, new_documents)
                return True
            
            index = self.indices[kb_id]
            
            # Select optimal chunking for new documents
            node_parser = self._select_optimal_chunking_strategy(new_documents)
            
            # Process new documents
            nodes = []
            for doc in new_documents:
                doc_nodes = await asyncio.to_thread(node_parser.get_nodes_from_documents, [doc])
                nodes.extend(doc_nodes)
            
            # Insert new nodes
            await asyncio.to_thread(index.insert_nodes, nodes)
            
            # Recreate query engine
            self.query_engines[kb_id] = self._create_query_engine(index, kb_id)
            
            logger.info(f"Index updated successfully for KB {kb_id} with {len(nodes)} new nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update index for KB {kb_id}: {e}")
            return False
    
    async def delete_index(self, kb_id: str) -> bool:
        """Delete an index and its associated data"""
        try:
            # Remove from memory
            if kb_id in self.indices:
                del self.indices[kb_id]
            if kb_id in self.query_engines:
                del self.query_engines[kb_id]
            
            # Delete ChromaDB collection
            try:
                self.chroma_client.delete_collection(name=kb_id)
                logger.info(f"ChromaDB collection {kb_id} deleted successfully")
            except Exception as e:
                logger.warning(f"Failed to delete ChromaDB collection {kb_id}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index for KB {kb_id}: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available embedding and LLM models"""
        return {
            "embedding_models": settings.AVAILABLE_EMBEDDING_MODELS,
            "llm_models": settings.AVAILABLE_LLM_MODELS,
            "current_embedding_model": self.embedding_model,
            "current_llm_model": self.llm_model
        }
    
    async def get_index_stats(self, kb_id: str) -> Dict[str, Any]:
        """Get statistics about an index"""
        try:
            if kb_id not in self.indices:
                return {"error": f"Index for KB {kb_id} not found"}
            
            index = self.indices[kb_id]
            
            # Get collection stats from ChromaDB
            collection = self.chroma_client.get_collection(name=kb_id)
            collection_count = collection.count()
            
            stats = {
                "kb_id": kb_id,
                "total_nodes": collection_count,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "semantic_chunking_enabled": self.enable_semantic_chunking,
                "hybrid_search_enabled": self.enable_hybrid_search,
                "reranking_enabled": self.enable_reranking,
                "postprocessors_count": len(self.postprocessors)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for KB {kb_id}: {e}")
            return {"error": str(e)}
    
    async def semantic_search(
        self, 
        kb_id: str, 
        query: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search without LLM generation"""
        try:
            if kb_id not in self.indices:
                raise ValueError(f"Knowledge base {kb_id} not found")
            
            index = self.indices[kb_id]
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k,
                embed_model=self.embed_model
            )
            
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Retrieve nodes
            nodes = await asyncio.to_thread(retriever.retrieve, query_bundle)
            
            # Filter by similarity threshold and format results
            results = []
            for node in nodes:
                if node.score and node.score >= similarity_threshold:
                    results.append({
                        "content": node.node.text,
                        "metadata": node.node.metadata,
                        "score": node.score,
                        "node_id": node.node.node_id
                    })
            
            logger.info(f"Semantic search completed for KB {kb_id}, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search for KB {kb_id}: {e}")
            raise