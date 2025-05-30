import logging
import traceback
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from app.services.document_processor.chunker import TextChunker
from app.services.document_processor.models import TextChunk

# Setup logging
logger = logging.getLogger("url_processor")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

class URLProcessor:
    def __init__(self):
        self.chunker = TextChunker()
        logger.info("URLProcessor initialized")
    
    async def fetch_url(self, url: str) -> str:
        """Fetch content from a URL and extract its text"""
        try:
            logger.info(f"Fetching URL: {url}")
            
            # Fetch URL content
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
                }) as response:
                    if response.status != 200:
                        error_msg = f"Failed to fetch URL: {url}, status code: {response.status}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    html = await response.text()
                    logger.info(f"Successfully fetched HTML from {url}: {len(html)} characters")
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                script_or_style.decompose()

            # Extract title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
                logger.info(f"Page title: {title}")
            
            # Get text from main content areas
            main_content = soup.find_all(['article', 'main', 'div.content', 'div.main'])
            main_text = ""
            
            if main_content:
                # Prioritize content from main sections
                for content in main_content:
                    main_text += content.get_text(separator='\n') + "\n\n"
            
            # If no main content found, get all text
            if not main_text:
                main_text = soup.get_text(separator='\n')
            
            # Clean text (remove excess whitespace)
            lines = (line.strip() for line in main_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Add title as first line if available
            if title:
                text = f"{title}\n\n{text}"
            
            logger.info(f"Extracted text from URL: {len(text)} characters")
            
            # Log a small preview of the text
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"Text preview: {preview}")
            
            return text
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
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
        url: str,
        document_id: str,
        kb_id: str,
        tenant_id: str,
        embedding_service: Any,
        vector_store: Any
    ) -> Dict[str, Any]:
        """
        Full processing pipeline for URL: fetch -> extract text -> chunk -> embed -> store vectors
        
        Args:
            url: URL to process
            document_id: UUID of the document
            kb_id: Knowledge base ID
            tenant_id: Tenant ID
            embedding_service: Service to generate embeddings
            vector_store: Vector database service
            
        Returns:
            Processing status and debug info
        """
        try:
            logger.info(f"Starting full URL processing pipeline for {url} (doc_id: {document_id}, kb: {kb_id})")
            
            # 1. Fetch and extract text from URL
            text = await self.fetch_url(url)
            logger.info(f"Text extraction complete: {len(text)} characters")
            
            # 2. Chunk text
            chunks = await self.chunk_text(text)
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
                    "source": url
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
                "url": url,
                "text_length": len(text),
                "chunks": len(chunks),
                "vectors": len(embeddings),
                "collection": kb_id
            }
            
        except Exception as e:
            logger.error(f"Error in URL processing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "kb_id": kb_id,
                "url": url
            }