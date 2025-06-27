# tests/RAG-Evaluation/core/pairwise_evaluator.py
"""
Pairwise comparison evaluation using LlamaIndex
"""
from typing import List, Tuple
from llama_index.core.evaluation import PairwiseComparisonEvaluator as LIPairwiseEvaluator

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class PairwiseEvaluator(BaseRAGEvaluator):
    """Evaluates responses using pairwise comparison"""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.evaluator = LIPairwiseEvaluator(llm=self.llm)
    
    def get_evaluator_name(self) -> str:
        return "Pairwise_Comparison"
    
    async def evaluate_pair(
        self, 
        query: str, 
        response_a: str, 
        response_b: str,
        contexts: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate a pair of responses"""
        try:
            # Use LlamaIndex pairwise evaluator
            eval_result = await self.evaluator.aevaluate(
                query=query,
                response=response_a,
                second_response=response_b,
                contexts=contexts
            )
            
            # Convert pairwise result to standard format
            # Score of 1.0 means response_a is better, 0.0 means response_b is better
            score = 1.0 if eval_result.score > 0 else 0.0
            
            result = EvaluationResult(
                evaluator_name=self.get_evaluator_name(),
                metric_name="pairwise_preference",
                score=score,
                feedback=eval_result.feedback,
                query=query,
                response=response_a,
                contexts=contexts,
                metadata={
                    "comparison_response": response_b,
                    "raw_score": eval_result.score
                }
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error in pairwise evaluation: {e}")
            return EvaluationResult(
                evaluator_name=self.get_evaluator_name(),
                metric_name="pairwise_preference",
                score=0.0,
                feedback=f"Evaluation failed: {str(e)}",
                query=query,
                response=response_a,
                contexts=contexts,
                metadata={"comparison_response": response_b}
            )
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Standard evaluate method - requires pairs in metadata"""
        results = []
        
        for datapoint in datapoints:
            if not datapoint.metadata or 'comparison_response' not in datapoint.metadata:
                self.logger.warning(f"No comparison response for pairwise evaluation: {datapoint.query[:50]}...")
                continue
            
            comparison_response = datapoint.metadata['comparison_response']
            result = await self.evaluate_pair(
                query=datapoint.query,
                response_a=datapoint.response,
                response_b=comparison_response,
                contexts=datapoint.contexts
            )
            results.append(result)
        
        return results