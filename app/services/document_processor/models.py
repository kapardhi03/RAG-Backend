# app/services/document_processor/models.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

class TextChunk(BaseModel):
    """Represents a chunk of text with metadata"""
    text: str
    metadata: Dict[str, Any] = {}