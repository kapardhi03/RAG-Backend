# tests/RAG-Evaluation/core/embedding_similarity_evaluator.py
"""
Embedding similarity-based evaluation
"""
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class EmbeddingSimilarityEvaluator(BaseRAGEvaluator):
    """Evaluates semantic similarity using embeddings"""
    
    def get_evaluator_name(self) -> str:
        return "Embedding_Similarity"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate semantic similarity between response and reference/context"""
        results = []
        
        for datapoint in datapoints:
            try:
                # Determine what to compare against
                comparison_text = None
                comparison_type = "reference"
                
                if datapoint.reference:
                    comparison_text = datapoint.reference
                elif datapoint.contexts:
                    comparison_text = " ".join(datapoint.contexts)
                    comparison_type = "context"
                else:
                    self.logger.warning(f"No reference or context for similarity: {datapoint.query[:50]}...")
                    continue
                
                # Get embeddings
                response_embedding = await self.embedding_model.aget_text_embedding(datapoint.response)
                comparison_embedding = await self.embedding_model.aget_text_embedding(comparison_text)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [response_embedding], 
                    [comparison_embedding]
                )[0][0]
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name=f"similarity_to_{comparison_type}",
                    score=float(similarity),
                    feedback=f"Cosine similarity to {comparison_type}: {similarity:.3f}",
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    contexts=datapoint.contexts,
                    metadata={
                        **(datapoint.metadata or {}),
                        "comparison_type": comparison_type,
                        "comparison_text": comparison_text[:200] + "..." if len(comparison_text) > 200 else comparison_text
                    }
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error calculating embedding similarity: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="similarity_score",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
        
        return results