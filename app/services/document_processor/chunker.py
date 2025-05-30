# app/services/document_processor/chunker.py
import logging
from typing import List, Dict, Any
import re
from .models import TextChunk

logger = logging.getLogger("text_chunker")

class TextChunker:
    async def chunk(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """
        Split text into semantically meaningful chunks
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of TextChunk objects
        """
        try:
            logger.info(f"Chunking text with size={chunk_size}, overlap={chunk_overlap}")
            
            # Clean the text - normalize whitespace
            text = self._clean_text(text)
            
            # Return empty list for empty text
            if not text:
                logger.warning("Empty text provided for chunking")
                return []
            
            # Split text into paragraphs
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Create chunks
            chunks = []
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                paragraph_size = len(paragraph)
                
                # If paragraph is larger than chunk_size, split it into sentences
                if paragraph_size > chunk_size:
                    sentences = self._split_into_sentences(paragraph)
                    
                    for sentence in sentences:
                        sentence_size = len(sentence)
                        
                        # If adding this sentence would exceed chunk_size, start a new chunk
                        if current_size + sentence_size > chunk_size and current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(TextChunk(text=chunk_text, metadata={}))
                            
                            # Create overlap by keeping some content from the previous chunk
                            overlap_size = 0
                            overlap_chunks = []
                            
                            # Add previous content until we reach desired overlap
                            for prev_chunk in reversed(current_chunk):
                                if overlap_size + len(prev_chunk) <= chunk_overlap:
                                    overlap_chunks.insert(0, prev_chunk)
                                    overlap_size += len(prev_chunk) + 1  # +1 for space
                                else:
                                    break
                            
                            current_chunk = overlap_chunks
                            current_size = overlap_size
                        
                        # Add sentence to current chunk
                        current_chunk.append(sentence)
                        current_size += sentence_size + 1  # +1 for space
                else:
                    # If adding this paragraph would exceed chunk_size, start a new chunk
                    if current_size + paragraph_size > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(TextChunk(text=chunk_text, metadata={}))
                        
                        # Create overlap by keeping some content from the previous chunk
                        overlap_size = 0
                        overlap_chunks = []
                        
                        # Add previous content until we reach desired overlap
                        for prev_chunk in reversed(current_chunk):
                            if overlap_size + len(prev_chunk) <= chunk_overlap:
                                overlap_chunks.insert(0, prev_chunk)
                                overlap_size += len(prev_chunk) + 1  # +1 for space
                            else:
                                break
                        
                        current_chunk = overlap_chunks
                        current_size = overlap_size
                    
                    # Add paragraph to current chunk
                    current_chunk.append(paragraph)
                    current_size += paragraph_size + 1  # +1 for space
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(text=chunk_text, metadata={}))
            
            logger.info(f"Created {len(chunks)} chunks from {len(text)} characters of text")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise ValueError(f"Failed to chunk text: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace"""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n+', '\n\n', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting by typical end-of-sentence markers
        # This is a simplified approach, a more sophisticated approach might use NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]