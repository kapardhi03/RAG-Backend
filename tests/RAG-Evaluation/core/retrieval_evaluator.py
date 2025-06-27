# tests/RAG-Evaluation/core/retrieval_evaluator.py
"""
Retrieval quality evaluation using LlamaIndex
"""
from typing import List, Dict, Any
from llama_index.core.evaluation.retrieval import RetrieverEvaluator as LIRetrieverEvaluator
from llama_index.core.evaluation.retrieval.metrics import (
    resolve_metrics,
    MRR,
    HitRate,
    NDCG
)

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class RetrievalEvaluator(BaseRAGEvaluator):
    """Evaluates retrieval quality using multiple metrics"""
    
    def __init__(self, llm=None, metrics=None, **kwargs):
        super().__init__(llm=llm)
        self.metrics = metrics or ["hit_rate", "mrr", "ndcg"]
        self.evaluator = LIRetrieverEvaluator.from_metric_names(
            metric_names=self.metrics
        )
    
    def get_evaluator_name(self) -> str:
        return "Retrieval_Quality"
    
    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_nodes: List[Any],
        expected_node_ids: List[str]
    ) -> List[EvaluationResult]:
        """Evaluate retrieval quality"""
        results = []
        
        try:
            # Evaluate retrieval
            eval_result = await self.evaluator.aevaluate(
                query=query,
                expected_ids=expected_node_ids,
                retrieved_ids=[node.id_ for node in retrieved_nodes]
            )
            
            # Create results for each metric
            for metric_name, score in eval_result.dict().items():
                if metric_name != "query":
                    result = EvaluationResult(
                        evaluator_name=self.get_evaluator_name(),
                        metric_name=metric_name,
                        score=score,
                        feedback=f"{metric_name}: {score:.3f}",
                        query=query,
                        response="",  # No response for retrieval eval
                        metadata={
                            "retrieved_count": len(retrieved_nodes),
                            "expected_count": len(expected_node_ids),
                            "retrieved_ids": [node.id_ for node in retrieved_nodes],
                            "expected_ids": expected_node_ids
                        }
                    )
                    results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating retrieval: {e}")
            for metric in self.metrics:
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name=metric,
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=query,
                    response="",
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Standard evaluate method - requires retrieval data in metadata"""
        results = []
        
        for datapoint in datapoints:
            if not datapoint.metadata or 'retrieved_nodes' not in datapoint.metadata:
                self.logger.warning(f"No retrieval data for evaluation: {datapoint.query[:50]}...")
                continue
            
            retrieved_nodes = datapoint.metadata['retrieved_nodes']
            expected_ids = datapoint.metadata.get('expected_node_ids', [])
            
            retrieval_results = await self.evaluate_retrieval(
                query=datapoint.query,
                retrieved_nodes=retrieved_nodes,
                expected_node_ids=expected_ids
            )
            results.extend(retrieval_results)
        
        return results