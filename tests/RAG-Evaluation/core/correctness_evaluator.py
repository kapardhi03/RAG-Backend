# tests/RAG-Evaluation/core/correctness_evaluator.py
"""
Correctness evaluation using LlamaIndex
"""
from typing import List, Optional
from llama_index.core.evaluation import CorrectnessEvaluator as LICorrectnessEvaluator

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class CorrectnessEvaluator(BaseRAGEvaluator):
    """Evaluates the correctness of RAG responses against reference answers"""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.evaluator = LICorrectnessEvaluator(llm=self.llm)
    
    def get_evaluator_name(self) -> str:
        return "Correctness"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate correctness of responses against references"""
        results = []
        
        for datapoint in datapoints:
            if not datapoint.reference:
                self.logger.warning(f"No reference answer provided for query: {datapoint.query[:50]}...")
                continue
            
            try:
                # Use LlamaIndex correctness evaluator
                eval_result = await self.evaluator.aevaluate(
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    contexts=datapoint.contexts
                )
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="correctness_score",
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating correctness: {e}")
                # Create a failed result
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="correctness_score",
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