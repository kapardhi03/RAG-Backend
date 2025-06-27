# tests/RAG-Evaluation/core/guideline_evaluator.py
"""
Guideline-based evaluation using LlamaIndex
"""
from typing import List, Optional
from llama_index.core.evaluation import GuidelineEvaluator as LIGuidelineEvaluator

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class GuidelineEvaluator(BaseRAGEvaluator):
    """Evaluates responses against custom guidelines"""
    
    def __init__(self, guidelines: str, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.guidelines = guidelines
        self.evaluator = LIGuidelineEvaluator(
            llm=self.llm,
            guidelines=guidelines
        )
    
    def get_evaluator_name(self) -> str:
        return "Guideline_Based"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate responses against guidelines"""
        results = []
        
        for datapoint in datapoints:
            try:
                # Use LlamaIndex guideline evaluator
                eval_result = await self.evaluator.aevaluate(
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts
                )
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="guideline_score",
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata={
                        **(datapoint.metadata or {}),
                        "guidelines": self.guidelines
                    }
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating with guidelines: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="guideline_score",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    contexts=datapoint.contexts,
                    metadata={
                        **(datapoint.metadata or {}),
                        "guidelines": self.guidelines
                    }
                )
                results.append(result)
        
        return results