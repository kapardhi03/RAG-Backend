import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from pathlib import Path

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter
)
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.core.schema import MetadataMode

from app.services.document_processor.parsers.pdf import PDFParser
from app.services.document_processor.parsers.docx import DocxParser
from app.services.document_processor.parsers.txt import TxtParser
from app.services.document_processor.parsers.md import MDParser
from app.services.llamaindex.engine import LlamaIndexRAGEngine
from app.utils.file_utils import get_file_extension
from app.core.config import settings

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced document processor with LlamaIndex integration and advanced chunking"""
    
    def __init__(self, rag_engine: Optional[LlamaIndexRAGEngine] = None):
        """Initialize with optional RAG engine for advanced processing"""
        self.rag_engine = rag_engine
        
        # Initialize file parsers
        self.parsers = {
            "pdf": PDFParser(),
            "docx": DocxParser(),
            "doc": DocxParser(),
            "txt": TxtParser(),
            "md": MDParser(),
        }
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize chunking strategies
        self._initialize_chunkers()
        
        # Initialize metadata extractors
        self._initialize_extractors()
        
        logger.info("Enhanced document processor initialized")
    
    def _initialize_chunkers(self):
        """Initialize different chunking strategies"""
        
        # Token-based chunker for precise token control
        self.token_chunker = TokenTextSplitter(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            separator=" ",
            backup_separators=["\n", "\t", ""]
        )
        
        # Sentence-based chunker for semantic coherence
        self.sentence_chunker = SentenceSplitter(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
        )
        
        # Semantic chunker (requires embedding model)
        if self.rag_engine and settings.ENABLE_SEMANTIC_CHUNKING:
            self.semantic_chunker = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD * 100,
                embed_model=self.rag_engine.embed_model
            )
        else:
            self.semantic_chunker = None
        
        logger.info("Chunking strategies initialized")
    
    def _initialize_extractors(self):
        """Initialize metadata extractors"""
        self.extractors = {}
        
        if self.rag_engine:
            # Title extractor
            self.extractors['title'] = TitleExtractor(
                nodes=5,
                llm=self.rag_engine.llm
            )
            
            # Keyword extractor
            self.extractors['keyword'] = KeywordExtractor(
                keywords=10,
                llm=self.rag_engine.llm
            )
            
            # Summary extractor for long documents
            self.extractors['summary'] = SummaryExtractor(
                summaries=["prev", "self"],
                llm=self.rag_engine.llm
            )
            
            # Questions answered extractor
            self.extractors['questions'] = QuestionsAnsweredExtractor(
                questions=5,
                llm=self.rag_engine.llm
            )
        
        logger.info(f"Metadata extractors initialized: {list(self.extractors.keys())}")
    
    async def process_file(self, file_content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process a file and return text content with metadata"""
        try:
            logger.info(f"Processing file: {filename}")
            
            # Get file extension and validate
            extension = get_file_extension(filename).lower()
            if extension not in self.parsers:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Parse file content
            parser = self.parsers[extension]
            text = await parser.parse(file_content)
            
            # Extract basic metadata
            metadata = {
                "filename": filename,
                "file_extension": extension,
                "file_size": len(file_content),
                "text_length": len(text),
                "estimated_tokens": len(self.tokenizer.encode(text))
            }
            
            # Analyze document characteristics
            doc_analysis = await self._analyze_document(text)
            metadata.update(doc_analysis)
            
            logger.info(f"File processed: {filename}, {metadata['text_length']} chars, {metadata['estimated_tokens']} tokens")
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
    
    async def _analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze document characteristics for optimal processing"""
        
        analysis = {}
        
        # Basic text statistics
        lines = text.split('\n')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        
        analysis.update({
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "avg_paragraph_length": len(words) / len(paragraphs) if paragraphs else 0
        })
        
        # Document complexity assessment
        if analysis["avg_sentence_length"] > 20:
            analysis["complexity"] = "high"
        elif analysis["avg_sentence_length"] > 15:
            analysis["complexity"] = "medium"
        else:
            analysis["complexity"] = "low"
        
        # Recommended chunking strategy
        if analysis["word_count"] > 5000 and self.semantic_chunker:
            analysis["recommended_chunking"] = "semantic"
        elif analysis["avg_sentence_length"] > 20:
            analysis["recommended_chunking"] = "sentence"
        else:
            analysis["recommended_chunking"] = "token"
        
        return analysis
    
    def _select_chunking_strategy(self, text: str, metadata: Dict[str, Any]) -> Any:
        """Select optimal chunking strategy based on document analysis"""
        
        recommended = metadata.get("recommended_chunking", "token")
        
        if recommended == "semantic" and self.semantic_chunker:
            logger.info("Using semantic chunking strategy")
            return self.semantic_chunker
        elif recommended == "sentence":
            logger.info("Using sentence-based chunking strategy")
            return self.sentence_chunker
        else:
            logger.info("Using token-based chunking strategy")
            return self.token_chunker
    
    async def create_document_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[LlamaDocument]:
        """Create optimally chunked documents with enhanced metadata"""
        try:
            # Override chunk parameters if provided
            if chunk_size:
                self._update_chunker_params(chunk_size, chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP)
            
            # Create initial LlamaIndex document
            document = LlamaDocument(
                text=text,
                metadata=metadata
            )
            
            # Select and apply chunking strategy
            chunker = self._select_chunking_strategy(text, metadata)
            
            # Generate chunks
            if self.semantic_chunker and chunker == self.semantic_chunker:
                # Semantic chunking (async)
                nodes = await asyncio.to_thread(chunker.get_nodes_from_documents, [document])
            else:
                # Token or sentence chunking (sync)
                nodes = chunker.get_nodes_from_documents([document])
            
            # Convert nodes to documents with enhanced metadata
            chunked_documents = []
            for i, node in enumerate(nodes):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_id": f"{metadata.get('filename', 'unknown')}_{i}",
                    "total_chunks": len(nodes),
                    "chunk_size": len(node.text),
                    "chunk_tokens": len(self.tokenizer.encode(node.text))
                })
                
                chunk_doc = LlamaDocument(
                    text=node.text,
                    metadata=chunk_metadata
                )
                chunked_documents.append(chunk_doc)
            
            # Apply metadata extraction if available
            if self.extractors and len(chunked_documents) > 1:
                chunked_documents = await self._extract_chunk_metadata(chunked_documents)
            
            logger.info(f"Created {len(chunked_documents)} chunks from document")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error creating document chunks: {e}")
            # Fallback to simple chunking
            return await self._fallback_chunking(text, metadata)
    
    def _update_chunker_params(self, chunk_size: int, chunk_overlap: int):
        """Update chunker parameters dynamically"""
        # Update token chunker
        self.token_chunker = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            backup_separators=["\n", "\t", ""]
        )
        
        # Update sentence chunker
        self.sentence_chunker = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
        )
    
    async def _extract_chunk_metadata(self, documents: List[LlamaDocument]) -> List[LlamaDocument]:
        """Extract metadata for chunks using LLM-based extractors"""
        try:
            if not self.extractors:
                return documents
            
            # Apply extractors selectively based on document characteristics
            enhanced_docs = []
            
            for doc in documents:
                enhanced_metadata = doc.metadata.copy()
                
                # Extract title for first chunk or long chunks
                if (doc.metadata.get('chunk_index', 0) == 0 or 
                    doc.metadata.get('chunk_tokens', 0) > 500):
                    
                    if 'title' in self.extractors:
                        try:
                            title_result = await asyncio.to_thread(
                                self.extractors['title'].extract, [doc]
                            )
                            if title_result and title_result[0].metadata.get('document_title'):
                                enhanced_metadata['extracted_title'] = title_result[0].metadata['document_title']
                        except Exception as e:
                            logger.warning(f"Title extraction failed: {e}")
                
                # Extract keywords for content-rich chunks
                if doc.metadata.get('chunk_tokens', 0) > 200 and 'keyword' in self.extractors:
                    try:
                        keyword_result = await asyncio.to_thread(
                            self.extractors['keyword'].extract, [doc]
                        )
                        if keyword_result and keyword_result[0].metadata.get('excerpt_keywords'):
                            enhanced_metadata['keywords'] = keyword_result[0].metadata['excerpt_keywords']
                    except Exception as e:
                        logger.warning(f"Keyword extraction failed: {e}")
                
                enhanced_doc = LlamaDocument(
                    text=doc.text,
                    metadata=enhanced_metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            logger.info(f"Enhanced metadata for {len(enhanced_docs)} chunks")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return documents
    
    async def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[LlamaDocument]:
        """Fallback to simple chunking if advanced methods fail"""
        try:
            # Simple token-based chunking
            chunk_size = settings.DEFAULT_CHUNK_SIZE
            chunk_overlap = settings.DEFAULT_CHUNK_OVERLAP
            
            # Split text into chunks
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "chunk_id": f"{metadata.get('filename', 'unknown')}_{len(chunks)}",
                    "chunk_size": len(chunk_text),
                    "chunk_tokens": len(self.tokenizer.encode(chunk_text)),
                    "chunking_method": "fallback"
                })
                
                chunk_doc = LlamaDocument(
                    text=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk_doc)
            
            # Update total chunks count
            for chunk in chunks:
                chunk.metadata["total_chunks"] = len(chunks)
            
            logger.info(f"Fallback chunking created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback chunking failed: {e}")
            raise
    
    async def process_and_index(
        self,
        kb_id: str,
        file_content: bytes,
        filename: str,
        document_id: str,
        tenant_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """Complete processing pipeline from file to indexed chunks"""
        try:
            logger.info(f"Starting complete processing for {filename}")
            
            # Step 1: Extract text and analyze document
            text, metadata = await self.process_file(file_content, filename)
            
            # Add processing metadata
            metadata.update({
                "document_id": document_id,
                "kb_id": kb_id,
                "tenant_id": tenant_id,
                "processing_timestamp": asyncio.get_event_loop().time()
            })
            
            # Step 2: Create optimized chunks
            documents = await self.create_document_chunks(
                text, metadata, chunk_size, chunk_overlap
            )
            
            # Step 3: Index documents if RAG engine is available
            if self.rag_engine:
                try:
                    # Check if index exists, create if not
                    if kb_id not in self.rag_engine.indices:
                        await self.rag_engine.create_index(kb_id, documents)
                    else:
                        await self.rag_engine.update_index(kb_id, documents)
                    
                    indexed = True
                    logger.info(f"Successfully indexed {len(documents)} chunks for KB {kb_id}")
                    
                except Exception as e:
                    logger.error(f"Indexing failed: {e}")
                    indexed = False
            else:
                indexed = False
                logger.warning("RAG engine not available, skipping indexing")
            
            # Return processing results
            result = {
                "success": True,
                "document_id": document_id,
                "kb_id": kb_id,
                "filename": filename,
                "text_length": len(text),
                "chunk_count": len(documents),
                "indexed": indexed,
                "metadata": metadata,
                "chunking_strategy": metadata.get("recommended_chunking", "token"),
                "processing_time": asyncio.get_event_loop().time() - metadata["processing_timestamp"]
            }
            
            logger.info(f"Processing completed for {filename}: {result['chunk_count']} chunks, indexed: {indexed}")
            return result
            
        except Exception as e:
            logger.error(f"Processing pipeline failed for {filename}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "kb_id": kb_id,
                "filename": filename
            }
    
    async def batch_process_files(
        self,
        kb_id: str,
        files: List[Tuple[bytes, str]],  # List of (file_content, filename) tuples
        tenant_id: str,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Process multiple files concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_single_file(file_content: bytes, filename: str):
            async with semaphore:
                # Generate document ID
                import uuid
                document_id = str(uuid.uuid4())
                
                result = await self.process_and_index(
                    kb_id=kb_id,
                    file_content=file_content,
                    filename=filename,
                    document_id=document_id,
                    tenant_id=tenant_id
                )
                results.append(result)
                return result
        
        # Create tasks for all files
        tasks = [
            process_single_file(file_content, filename) 
            for file_content, filename in files
        ]
        
        # Execute with progress logging
        for i, task in enumerate(asyncio.as_completed(tasks)):
            await task
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files")
        
        successful = len([r for r in results if r.get('success', False)])
        logger.info(f"Batch processing completed: {successful}/{len(files)} files successful")
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration"""
        stats = {
            "available_parsers": list(self.parsers.keys()),
            "chunking_strategies": ["token", "sentence"],
            "extractors_available": list(self.extractors.keys()) if self.extractors else [],
            "semantic_chunking_enabled": self.semantic_chunker is not None,
            "rag_engine_connected": self.rag_engine is not None,
            "default_chunk_size": settings.DEFAULT_CHUNK_SIZE,
            "default_chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
            "max_chunk_size": settings.MAX_CHUNK_SIZE,
            "min_chunk_size": settings.MIN_CHUNK_SIZE
        }
        
        if self.semantic_chunker:
            stats["chunking_strategies"].append("semantic")
        
        return stats
    
    async def optimize_chunk_parameters(
        self,
        sample_texts: List[str],
        target_chunk_count: int = None
    ) -> Dict[str, int]:
        """Analyze sample texts to recommend optimal chunk parameters"""
        try:
            if not sample_texts:
                return {
                    "chunk_size": settings.DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP
                }
            
            # Analyze text characteristics
            total_tokens = 0
            total_length = 0
            
            for text in sample_texts:
                total_tokens += len(self.tokenizer.encode(text))
                total_length += len(text)
            
            avg_tokens = total_tokens / len(sample_texts)
            avg_length = total_length / len(sample_texts)
            
            # Calculate optimal parameters
            if target_chunk_count:
                optimal_chunk_size = int(avg_tokens / target_chunk_count)
            else:
                # Base on text complexity
                if avg_tokens > 5000:
                    optimal_chunk_size = 1536  # Larger chunks for long documents
                elif avg_tokens > 2000:
                    optimal_chunk_size = 1024  # Medium chunks
                else:
                    optimal_chunk_size = 512   # Smaller chunks for short documents
            
            # Ensure within bounds
            optimal_chunk_size = max(
                settings.MIN_CHUNK_SIZE,
                min(optimal_chunk_size, settings.MAX_CHUNK_SIZE)
            )
            
            # Set overlap as 20% of chunk size
            optimal_overlap = int(optimal_chunk_size * 0.2)
            
            recommendations = {
                "chunk_size": optimal_chunk_size,
                "chunk_overlap": optimal_overlap,
                "reasoning": f"Based on {len(sample_texts)} samples, avg tokens: {avg_tokens:.0f}"
            }
            
            logger.info(f"Optimized chunk parameters: {recommendations}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {
                "chunk_size": settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
                "reasoning": "Using defaults due to optimization failure"
            }