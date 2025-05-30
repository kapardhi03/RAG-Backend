from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.embedding.base import EmbeddingService

class SentenceTransformerEmbedding(EmbeddingService):
    """Embedding service using Sentence Transformers models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a specific model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                        Default is 'all-MiniLM-L6-v2' which is small and fast
        """
        self.model = SentenceTransformer(model_name)
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Convert to numpy arrays and then to list of lists for compatibility
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text)
        return embedding.tolist()