from abc import ABC, abstractmethod
from typing import List, Dict, Any

class EmbeddingService(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        pass