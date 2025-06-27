# tests/RAG-Evaluation/core/relevance_evaluator.py
"""
Answer relevance evaluation using LlamaIndex
"""
from typing import List
from llama_index.core.evaluation import RelevancyEvaluator as LIRelevancyEvaluator

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class AnswerRelevanceEvaluator(BaseRAGEvaluator):
    """Evaluates how relevant the answer is to the query"""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.evaluator = LIRelevancyEvaluator(llm=self.llm)
    
    def get_evaluator_name(self) -> str:
        return "Answer_Relevance"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate relevance of responses to queries"""
        results = []
        
        for datapoint in datapoints:
            try:
                # Use LlamaIndex relevancy evaluator
                eval_result = await self.evaluator.aevaluate(
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts
                )
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="relevance_score",
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating relevance: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="relevance_score",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata=datapoint.metadata
                )
                results.append(result)
        
        return results
