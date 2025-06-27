# tests/RAG-Evaluation/utils/metrics_utils.py
"""
Metrics and analysis utilities
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from scipy import stats
import logging

from ..core.base_evaluator import EvaluationResult

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate additional metrics and statistics"""
    
    @staticmethod
    def calculate_agreement_metrics(results1: List[EvaluationResult], 
                                  results2: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate agreement metrics between two sets of results"""
        
        if len(results1) != len(results2):
            raise ValueError("Result sets must have the same length")
        
        scores1 = [r.score for r in results1]
        scores2 = [r.score for r in results2]
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(scores1, scores2)
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(scores1, scores2)
        
        # Mean absolute error
        mae = np.mean(np.abs(np.array(scores1) - np.array(scores2)))
        
        # Root mean square error
        rmse = np.sqrt(np.mean((np.array(scores1) - np.array(scores2)) ** 2))
        
        return {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "mean_absolute_error": mae,
            "root_mean_square_error": rmse
        }
    
    @staticmethod
    def calculate_reliability_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate reliability metrics for evaluation results"""
        
        scores = [r.score for r in results]
        
        metrics = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "median": np.median(scores),
            "q25": np.percentile(scores, 25),
            "q75": np.percentile(scores, 75),
            "iqr": np.percentile(scores, 75) - np.percentile(scores, 25),
            "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def detect_outliers(results: List[EvaluationResult], method: str = "iqr") -> List[bool]:
        """Detect outliers in evaluation results"""
        
        scores = np.array([r.score for r in results])
        
        if method == "iqr":
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return (scores < lower_bound) | (scores > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(scores))
            return z_scores > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

class ResultsAnalyzer:
    """Analyze evaluation results"""
    
    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from results"""
        data = []
        for result in self.results:
            data.append({
                "evaluator": result.evaluator_name,
                "metric": result.metric_name,
                "score": result.score,
                "query": result.query,
                "response": result.response,
                "reference": result.reference,
                "feedback": result.feedback,
                "timestamp": result.timestamp
            })
        
        return pd.DataFrame(data)
    
    def get_summary_by_evaluator(self) -> pd.DataFrame:
        """Get summary statistics by evaluator"""
        return self.df.groupby("evaluator")["score"].agg([
            "count", "mean", "std", "min", "max", "median"
        ]).round(3)
    
    def get_summary_by_metric(self) -> pd.DataFrame:
        """Get summary statistics by metric"""
        return self.df.groupby("metric")["score"].agg([
            "count", "mean", "std", "min", "max", "median"
        ]).round(3)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix between evaluators"""
        pivot_df = self.df.pivot_table(
            index=["query", "response"], 
            columns="evaluator", 
            values="score"
        )
        
        return pivot_df.corr()
    
    def identify_problematic_queries(self, threshold: float = 0.5) -> pd.DataFrame:
        """Identify queries with low scores across evaluators"""
        query_scores = self.df.groupby("query")["score"].agg(["mean", "std", "count"])
        problematic = query_scores[query_scores["mean"] < threshold]
        
        return problematic.sort_values("mean")
    
    def get_evaluator_agreement(self) -> Dict[str, float]:
        """Calculate agreement between evaluators"""
        pivot_df = self.df.pivot_table(
            index=["query", "response"], 
            columns="evaluator", 
            values="score"
        )
        
        agreement_scores = {}
        evaluators = pivot_df.columns.tolist()
        
        for i, eval1 in enumerate(evaluators):
            for eval2 in evaluators[i+1:]:
                if not pivot_df[eval1].isna().all() and not pivot_df[eval2].isna().all():
                    valid_pairs = ~(pivot_df[eval1].isna() | pivot_df[eval2].isna())
                    if valid_pairs.sum() > 0:
                        corr = pivot_df.loc[valid_pairs, eval1].corr(pivot_df.loc[valid_pairs, eval2])
                        agreement_scores[f"{eval1}_vs_{eval2}"] = corr
        
        return agreement_scores