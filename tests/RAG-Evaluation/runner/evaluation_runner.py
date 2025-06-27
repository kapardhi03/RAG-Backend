# tests/RAG-Evaluation/runner/evaluation_runner.py
"""
Main evaluation runner that orchestrates all evaluations
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path

from  core.base_evaluator import BaseRAGEvaluator, EvaluationDatapoint, EvaluationResult
from core.correctness_evaluator import CorrectnessEvaluator
from core.faithfulness_evaluator import FaithfulnessEvaluator
from core.relevance_evaluator import AnswerRelevanceEvaluator
from core.guideline_evaluator import GuidelineEvaluator
from core.pairwise_evaluator import PairwiseEvaluator
from core.embedding_similarity_evaluator import EmbeddingSimilarityEvaluator
from core.retrieval_evaluator import RetrievalEvaluator
from core.benchmark_evaluators import MTBenchEvaluator, MiniMTBenchEvaluator
from core.prometheus_evaluator import PrometheusEvaluator

logger = logging.getLogger(__name__)

class RAGEvaluationRunner:
    """Main runner for RAG evaluations"""
    
    def __init__(self, llm=None, embedding_model=None):
        self.llm = llm
        self.embedding_model = embedding_model
        self.evaluators: Dict[str, BaseRAGEvaluator] = {}
        self.results: List[EvaluationResult] = []
        
        # Initialize default evaluators
        self._setup_default_evaluators()
    
    def _setup_default_evaluators(self):
        """Set up default evaluators"""
        self.evaluators = {
            "correctness": CorrectnessEvaluator(llm=self.llm),
            "faithfulness": FaithfulnessEvaluator(llm=self.llm),
            "answer_relevance": AnswerRelevanceEvaluator(llm=self.llm),
            "embedding_similarity": EmbeddingSimilarityEvaluator(
                llm=self.llm, 
                embedding_model=self.embedding_model
            ),
            "mini_mt_bench": MiniMTBenchEvaluator(llm=self.llm),
        }
    
    def add_evaluator(self, name: str, evaluator: BaseRAGEvaluator):
        """Add a custom evaluator"""
        self.evaluators[name] = evaluator
    
    def add_guideline_evaluator(self, name: str, guidelines: str):
        """Add a guideline-based evaluator"""
        self.evaluators[name] = GuidelineEvaluator(
            guidelines=guidelines,
            llm=self.llm
        )
    
    def add_prometheus_evaluator(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """Add Prometheus evaluator"""
        self.evaluators["prometheus"] = PrometheusEvaluator(
            prometheus_endpoint=endpoint,
            api_key=api_key,
            llm=self.llm
        )
    
    def add_mt_bench_evaluator(self, benchmark_data_path: Optional[str] = None):
        """Add MT-Bench evaluator"""
        self.evaluators["mt_bench"] = MTBenchEvaluator(
            llm=self.llm,
            benchmark_data_path=benchmark_data_path
        )
    
    def add_retrieval_evaluator(self, metrics: Optional[List[str]] = None):
        """Add retrieval quality evaluator"""
        self.evaluators["retrieval"] = RetrievalEvaluator(
            llm=self.llm,
            metrics=metrics
        )
    
    async def run_evaluation(
        self,
        datapoints: List[EvaluationDatapoint],
        evaluator_names: Optional[List[str]] = None,
        save_results: bool = True,
        results_dir: str = "evaluation_results"
    ) -> Dict[str, List[EvaluationResult]]:
        """Run evaluation with specified evaluators"""
        
        if evaluator_names is None:
            evaluator_names = list(self.evaluators.keys())
        
        logger.info(f"Running evaluation with {len(evaluator_names)} evaluators on {len(datapoints)} datapoints")
        
        all_results = {}
        
        for evaluator_name in evaluator_names:
            if evaluator_name not in self.evaluators:
                logger.warning(f"Evaluator '{evaluator_name}' not found. Skipping.")
                continue
            
            logger.info(f"Running {evaluator_name} evaluation...")
            
            try:
                evaluator = self.evaluators[evaluator_name]
                results = await evaluator.evaluate(datapoints)
                all_results[evaluator_name] = results
                self.results.extend(results)
                
                logger.info(f"Completed {evaluator_name}: {len(results)} results")
                
            except Exception as e:
                logger.error(f"Error running {evaluator_name}: {e}")
                all_results[evaluator_name] = []
        
        if save_results:
            self._save_results(all_results, results_dir)
        
        return all_results
    
    def _save_results(self, results: Dict[str, List[EvaluationResult]], results_dir: str):
        """Save results to files"""
        Path(results_dir).mkdir(exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame for easy analysis
        all_data = []
        for evaluator_name, evaluator_results in results.items():
            for result in evaluator_results:
                all_data.append({
                    "timestamp": result.timestamp,
                    "evaluator": result.evaluator_name,
                    "metric": result.metric_name,
                    "score": result.score,
                    "query": result.query,
                    "response": result.response,
                    "reference": result.reference,
                    "feedback": result.feedback,
                    "contexts_count": len(result.contexts) if result.contexts else 0,
                    "metadata": str(result.metadata) if result.metadata else ""
                })
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(f"{results_dir}/evaluation_results_{timestamp}.csv", index=False)
            
            # Save summary statistics
            summary = df.groupby(['evaluator', 'metric'])['score'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            summary.to_csv(f"{results_dir}/evaluation_summary_{timestamp}.csv")
            
            logger.info(f"Results saved to {results_dir}/")
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics of all results"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                "evaluator": result.evaluator_name,
                "metric": result.metric_name,
                "score": result.score
            })
        
        df = pd.DataFrame(data)
        return df.groupby(['evaluator', 'metric'])['score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
    
    def clear_results(self):
        """Clear stored results"""
        self.results = []