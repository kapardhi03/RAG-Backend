# tests/RAG-Evaluation/core/prometheus_evaluator.py
"""
Prometheus model-based evaluation
Note: This would require the Prometheus model which may not be locally available
"""
from typing import List, Optional
import requests
import json

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class PrometheusEvaluator(BaseRAGEvaluator):
    """
    Prometheus model evaluation
    Note: Requires Prometheus model API or local deployment
    """
    
    def __init__(self, 
                 prometheus_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 llm=None, 
                 **kwargs):
        super().__init__(llm=llm)
        self.prometheus_endpoint = prometheus_endpoint
        self.api_key = api_key
        self.use_fallback = not prometheus_endpoint
        
        if self.use_fallback:
            self.logger.warning("Prometheus endpoint not provided. Using LLM fallback.")
    
    def get_evaluator_name(self) -> str:
        return "Prometheus_Style"
    
    async def _prometheus_api_call(self, prompt: str) -> str:
        """Call Prometheus API if available"""
        if self.use_fallback:
            # Fallback to regular LLM with Prometheus-style prompt
            return str(await self.llm.acomplete(prompt))
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.prometheus_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"Prometheus API error: {response.status_code}")
                return str(await self.llm.acomplete(prompt))
                
        except Exception as e:
            self.logger.error(f"Prometheus API call failed: {e}")
            return str(await self.llm.acomplete(prompt))
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate using Prometheus-style criteria"""
        results = []
        
        for datapoint in datapoints:
            try:
                # Prometheus-style evaluation prompt
                eval_prompt = f"""
                You are an expert evaluator. Please evaluate the following response using these criteria:
                
                1. Factual Accuracy (0-5): Is the information correct?
                2. Completeness (0-5): Does it fully address the question?
                3. Clarity (0-5): Is it well-structured and clear?
                4. Relevance (0-5): How relevant is it to the question?
                5. Source Faithfulness (0-5): If sources are provided, is the response faithful to them?
                
                Question: {datapoint.query}
                Response: {datapoint.response}
                Context/Sources: {' '.join(datapoint.contexts) if datapoint.contexts else 'Not provided'}
                Reference: {datapoint.reference or 'Not provided'}
                
                Please provide:
                - Individual scores for each criterion (0-5)
                - Overall score (average)
                - Brief justification
                
                Format: Accuracy: X, Completeness: X, Clarity: X, Relevance: X, Faithfulness: X
                Overall: X.X
                Justification: [explanation]
                """
                
                eval_response = await self._prometheus_api_call(eval_prompt)
                
                # Parse the response to extract scores
                import re
                
                # Extract individual scores
                accuracy = self._extract_score(eval_response, "Accuracy")
                completeness = self._extract_score(eval_response, "Completeness")
                clarity = self._extract_score(eval_response, "Clarity")
                relevance = self._extract_score(eval_response, "Relevance")
                faithfulness = self._extract_score(eval_response, "Faithfulness")
                
                # Calculate overall score
                scores = [s for s in [accuracy, completeness, clarity, relevance, faithfulness] if s is not None]
                overall_score = sum(scores) / len(scores) / 5.0 if scores else 0.0  # Normalize to 0-1
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="prometheus_score",
                    score=overall_score,
                    feedback=eval_response,
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    contexts=datapoint.contexts,
                    metadata={
                        **(datapoint.metadata or {}),
                        "individual_scores": {
                            "accuracy": accuracy,
                            "completeness": completeness,
                            "clarity": clarity,
                            "relevance": relevance,
                            "faithfulness": faithfulness
                        }
                    }
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in Prometheus evaluation: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="prometheus_score",
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
    
    def _extract_score(self, text: str, criterion: str) -> Optional[float]:
        """Extract score for a specific criterion"""
        import re
        pattern = rf"{criterion}:\s*([0-5](?:\.[0-9])?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None