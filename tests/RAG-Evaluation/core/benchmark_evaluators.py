# tests/RAG-Evaluation/core/benchmark_evaluators.py
"""
Benchmark evaluators (MT-Bench style evaluations)
Note: These require external datasets which may not be available
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult

class MTBenchEvaluator(BaseRAGEvaluator):
    """
    MT-Bench style evaluation
    Note: Requires MT-Bench dataset which is not included
    """
    
    def __init__(self, llm=None, benchmark_data_path: Optional[str] = None, **kwargs):
        super().__init__(llm=llm)
        self.benchmark_data_path = benchmark_data_path
        self.benchmark_data = self._load_benchmark_data()
    
    def get_evaluator_name(self) -> str:
        return "MT_Bench_Style"
    
    def _load_benchmark_data(self) -> List[Dict[str, Any]]:
        """Load MT-Bench style data if available"""
        if not self.benchmark_data_path or not os.path.exists(self.benchmark_data_path):
            self.logger.warning("MT-Bench data not found. Using sample data.")
            return self._create_sample_benchmark_data()
        
        try:
            with open(self.benchmark_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
            return self._create_sample_benchmark_data()
    
    def _create_sample_benchmark_data(self) -> List[Dict[str, Any]]:
        """Create sample benchmark questions for testing"""
        return [
            {
                "category": "knowledge",
                "question": "What is the capital of France?",
                "reference_answer": "The capital of France is Paris.",
                "difficulty": "easy"
            },
            {
                "category": "reasoning",
                "question": "If a train travels at 60 mph for 2 hours, how far does it go?",
                "reference_answer": "The train travels 120 miles (60 mph × 2 hours = 120 miles).",
                "difficulty": "medium"
            },
            {
                "category": "analysis",
                "question": "Compare the advantages and disadvantages of renewable energy sources.",
                "reference_answer": "Renewable energy advantages include sustainability and environmental benefits. Disadvantages include intermittency and initial costs.",
                "difficulty": "hard"
            }
        ]
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Evaluate using MT-Bench style criteria"""
        results = []
        
        # Use benchmark questions if no datapoints provided
        if not datapoints:
            datapoints = [
                EvaluationDatapoint(
                    query=item["question"],
                    response="",  # Would need actual responses
                    reference=item["reference_answer"],
                    metadata={"category": item["category"], "difficulty": item["difficulty"]}
                )
                for item in self.benchmark_data
            ]
        
        for datapoint in datapoints:
            if not datapoint.response:
                self.logger.warning(f"No response to evaluate for: {datapoint.query[:50]}...")
                continue
            
            try:
                # MT-Bench style evaluation prompt
                eval_prompt = f"""
                Please evaluate the following response on a scale of 1-10 based on these criteria:
                - Accuracy and correctness
                - Completeness of the answer
                - Clarity and coherence
                - Relevance to the question
                
                Question: {datapoint.query}
                Response: {datapoint.response}
                Reference Answer: {datapoint.reference or "Not provided"}
                
                Provide a score (1-10) and brief explanation.
                """
                
                response = await self.llm.acomplete(eval_prompt)
                eval_text = str(response)
                
                # Extract score (simple regex approach)
                import re
                score_match = re.search(r'\b([1-9]|10)\b', eval_text)
                score = float(score_match.group(1)) / 10.0 if score_match else 0.5
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="mt_bench_score",
                    score=score,
                    feedback=eval_text,
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    metadata=datapoint.metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in MT-Bench evaluation: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="mt_bench_score",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    metadata=datapoint.metadata
                )
                results.append(result)
        
        return results

class MiniMTBenchEvaluator(BaseRAGEvaluator):
    """
    Mini MT-Bench evaluator with single grading
    """
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
    
    def get_evaluator_name(self) -> str:
        return "Mini_MT_Bench"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        """Simplified MT-Bench style evaluation"""
        results = []
        
        for datapoint in datapoints:
            try:
                # Simplified evaluation prompt
                eval_prompt = f"""
                Rate this response on a scale of 1-5:
                5 = Excellent (accurate, complete, clear)
                4 = Good (mostly accurate, adequate detail)
                3 = Fair (some accuracy, basic response)
                2 = Poor (limited accuracy or relevance)
                1 = Very Poor (inaccurate or irrelevant)
                
                Question: {datapoint.query}
                Response: {datapoint.response}
                
                Rating:
                """
                
                response = await self.llm.acomplete(eval_prompt)
                eval_text = str(response)
                
                # Extract rating
                import re
                rating_match = re.search(r'\b([1-5])\b', eval_text)
                score = float(rating_match.group(1)) / 5.0 if rating_match else 0.5
                
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="mini_bench_rating",
                    score=score,
                    feedback=eval_text,
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    metadata=datapoint.metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in Mini MT-Bench evaluation: {e}")
                result = EvaluationResult(
                    evaluator_name=self.get_evaluator_name(),
                    metric_name="mini_bench_rating",
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    query=datapoint.query,
                    response=datapoint.response,
                    reference=datapoint.reference,
                    metadata=datapoint.metadata
                )
                results.append(result)
        
        return results