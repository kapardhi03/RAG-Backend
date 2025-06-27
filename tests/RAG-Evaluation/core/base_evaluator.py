# tests/RAG-Evaluation/core/base_evaluator.py
"""
Base evaluator class for all RAG evaluation components
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    GuidelineEvaluator,
    PairwiseComparisonEvaluator,
    BatchEvalRunner
)
from llama_index.core.evaluation.retrieval import RetrieverEvaluator
from llama_index.core.llms import OpenAI
from llama_index.core.embeddings import OpenAIEmbedding

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Standard evaluation result format"""
    evaluator_name: str
    metric_name: str
    score: float
    feedback: str
    query: str
    response: str
    reference: Optional[str] = None
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class EvaluationDatapoint:
    """Standard format for evaluation data"""
    query: str
    response: str
    reference: Optional[str] = None
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseRAGEvaluator(ABC):
    """Base class for all RAG evaluators"""
    
    def __init__(self, llm: Optional[Any] = None, embedding_model: Optional[Any] = None):
        self.llm = llm or OpenAI(model="gpt-4-turbo-preview", temperature=0.0)
        self.embedding_model = embedding_model or OpenAIEmbedding()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate a list of datapoints"""
        pass
    
    @abstractmethod
    def get_evaluator_name(self) -> str:
        """Get the name of this evaluator"""
        pass
    
    async def evaluate_single(self, datapoint: EvaluationDatapoint) -> EvaluationResult:
        """Evaluate a single datapoint"""
        results = await self.evaluate([datapoint])
        return results[0] if results else None