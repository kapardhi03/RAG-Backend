# tests/RAG-Evaluation/core/faithfulness_evaluator.py
"""
Faithfulness evaluation using LlamaIndex
"""
from typing import List
from llama_index.core.evaluation import FaithfulnessEvaluator as LIFaithfulnessEvaluator

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class FaithfulnessEvaluator(BaseRAGEvaluator):
    """Evaluates whether RAG responses are faithful to the retrieved contexts"""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.evaluator = LIFaithfulnessEvaluator(llm=self.llm)
    
    def get_evaluator_name(self) -> str:
        return "Faithfulness"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate faithfulness of responses to contexts"""
        results = []
        
        for datapoint in datapoints:
            if not datapoint.contexts:
                self.logger.warning(f"No contexts provided for query: {datapoint.query[:50]}...")
                continue
            
            try:
                # Use LlamaIndex faithfulness evaluator
                eval_result = await self.evaluator.aevaluate(
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts
                )
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="faithfulness_score",
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating faithfulness: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="faithfulness_score",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
        
        return results
