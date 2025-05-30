import os
import logging
import traceback
from typing import List, Dict, Any, Optional
from app.services.document_processor.parsers.pdf import PDFParser
from app.services.document_processor.parsers.docx import DocxParser
from app.services.document_processor.parsers.txt import TxtParser
from app.services.document_processor.parsers.md import MDParser
from app.services.document_processor.chunker import TextChunker
from app.services.document_processor.models import TextChunk
from app.utils.file_utils import get_file_extension

# Setup logging
logger = logging.getLogger("document_processor")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

class DocumentProcessor:
    def __init__(self):
        self.parsers = {
            "pdf": PDFParser(),
            "docx": DocxParser(),
            "doc": DocxParser(),  # Use the same parser for .doc files
            "txt": TxtParser(),
            "md": MDParser(),
        }
        self.chunker = TextChunker()
        logger.info("DocumentProcessor initialized")
    
    async def process_file(self, file_content: bytes, filename: str) -> str:
        """Process a file and extract its text content"""
        try:
            logger.info(f"Processing file: {filename}")
            extension = get_file_extension(filename).lower()
            
            if extension not in self.parsers:
                error_msg = f"Unsupported file type: {extension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            parser = self.parsers[extension]
            logger.info(f"Using parser for {extension}")
            
            text = await parser.parse(file_content)
            logger.info(f"Successfully extracted text from {filename}: {len(text)} characters")
            
            # Log a small preview of the text
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"Text preview: {preview}")
            
            return text
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """Split text into chunks with metadata"""
        try:
            logger.info(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={chunk_overlap}")
            chunks = await self.chunker.chunk(text, chunk_size, chunk_overlap)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Log a sample of the first chunk
            if chunks:
                sample = chunks[0].text[:100] + "..." if len(chunks[0].text) > 100 else chunks[0].text
                logger.info(f"Sample chunk: {sample}")
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def process_and_embed(
        self, 
        file_content: bytes, 
        filename: str,
        document_id: str,
        kb_id: str,
        tenant_id: str,
        embedding_service: Any,
        vector_store: Any
    ) -> Dict[str, Any]:
        """
        Full processing pipeline: extract text -> chunk -> embed -> store vectors
        
        Args:
            file_content: Raw file bytes
            filename: Name of the file
            document_id: UUID of the document
            kb_id: Knowledge base ID
            tenant_id: Tenant ID
            embedding_service: Service to generate embeddings
            vector_store: Vector database service
            
        Returns:
            Processing status and debug info
        """
        try:
            logger.info(f"Starting full processing pipeline for {filename} (doc_id: {document_id}, kb: {kb_id})")
            
            # 1. Extract text
            text = await self.process_file(file_content, filename)
            logger.info(f"Text extraction complete: {len(text)} characters")
            
            # 2. Chunk text
            chunks = await self.chunk_text(text, chunk_size=1000, chunk_overlap=200)
            chunk_texts = [chunk.text for chunk in chunks]
            logger.info(f"Text chunking complete: {len(chunks)} chunks created")
            
            # 3. Generate embeddings
            try:
                logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
                embeddings = await embedding_service.embed_documents(chunk_texts)
                logger.info(f"Embedding generation complete: {len(embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to generate embeddings: {str(e)}")
            
            # 4. Prepare metadata for storage
            metadatas = []
            for i, _ in enumerate(chunks):
                metadatas.append({
                    "doc_id": document_id,
                    "kb_id": kb_id,
                    "tenant_id": tenant_id,
                    "position": i,
                    "source": filename
                })
            
            # 5. Check if collection exists, create if not
            try:
                collection_exists = await vector_store.collection_exists(kb_id)
                logger.info(f"Collection '{kb_id}' exists: {collection_exists}")
                
                if not collection_exists:
                    logger.info(f"Creating new collection '{kb_id}'")
                    vector_store.client.create_collection(name=kb_id)
                    logger.info(f"Created new collection '{kb_id}'")
            except Exception as e:
                logger.error(f"Error checking/creating collection: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to check/create ChromaDB collection: {str(e)}")
            
            # 6. Store vectors
            try:
                logger.info(f"Storing {len(chunk_texts)} vectors in collection '{kb_id}'")
                collection = vector_store.client.get_collection(name=kb_id)
                
                # Generate IDs for each chunk
                ids = [f"{document_id}_{i}" for i in range(len(chunk_texts))]
                
                # Add to ChromaDB
                collection.add(
                    documents=chunk_texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully stored {len(chunk_texts)} vectors in ChromaDB")
            except Exception as e:
                logger.error(f"Error storing vectors: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to store vectors in ChromaDB: {str(e)}")
            
            # Return processing results and debug info
            return {
                "success": True,
                "document_id": document_id,
                "kb_id": kb_id,
                "filename": filename,
                "text_length": len(text),
                "chunks": len(chunks),
                "vectors": len(embeddings),
                "collection": kb_id
            }
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "kb_id": kb_id,
                "filename": filename
            }