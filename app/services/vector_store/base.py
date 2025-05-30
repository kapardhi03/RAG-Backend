from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    @abstractmethod
    async def create_collection(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new collection (knowledge base)"""
        pass
    
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        pass
    
    @abstractmethod
    async def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        pass
    
    @abstractmethod
    async def add_documents(self, kb_id: str, texts: List[str], metadatas: List[Dict[str, Any]], embedding_service) -> str:
        """Add documents to a collection"""
        pass
    
    @abstractmethod
    async def get_documents(self, kb_id: str) -> List[Dict[str, Any]]:
        """Get all documents in a collection"""
        pass
    
    @abstractmethod
    async def delete_document(self, kb_id: str, doc_id: str) -> None:
        """Delete a document from a collection"""
        pass
    
    @abstractmethod
    async def similarity_search(self, kb_id: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        pass