# tests/RAG-Evaluation/runner/batch_evaluator.py
"""
Batch evaluation utilities
"""
import asyncio
from typing import List, Dict, Any, Callable, Optional
import json
from pathlib import Path

from ..core.base_evaluator import EvaluationDatapoint
from .evaluation_runner import RAGEvaluationRunner

class BatchEvaluator:
    """Utilities for batch evaluation"""
    
    def __init__(self, runner: RAGEvaluationRunner):
        self.runner = runner
    
    @staticmethod
    def load_datapoints_from_json(file_path: str) -> List[EvaluationDatapoint]:
        """Load evaluation datapoints from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        datapoints = []
        for item in data:
            datapoint = EvaluationDatapoint(
                query=item["query"],
                response=item.get("response", ""),
                reference=item.get("reference"),
                contexts=item.get("contexts"),
                metadata=item.get("metadata")
            )
            datapoints.append(datapoint)
        
        return datapoints
    
    @staticmethod
    def load_datapoints_from_csv(file_path: str) -> List[EvaluationDatapoint]:
        """Load evaluation datapoints from CSV file"""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        datapoints = []
        
        for _, row in df.iterrows():
            # Parse contexts if it's a string representation of a list
            contexts = None
            if 'contexts' in row and pd.notna(row['contexts']):
                try:
                    contexts = eval(row['contexts']) if isinstance(row['contexts'], str) else row['contexts']
                except:
                    contexts = [str(row['contexts'])]
            
            datapoint = EvaluationDatapoint(
                query=row["query"],
                response=row.get("response", ""),
                reference=row.get("reference") if pd.notna(row.get("reference")) else None,
                contexts=contexts,
                metadata={"source_file": file_path}
            )
            datapoints.append(datapoint)
        
        return datapoints
    
    async def evaluate_rag_system(
        self,
        rag_query_function: Callable[[str], Dict[str, Any]],
        test_queries: List[str],
        reference_answers: Optional[List[str]] = None,
        evaluator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system end-to-end
        
        Args:
            rag_query_function: Function that takes a query and returns RAG response
            test_queries: List of test queries
            reference_answers: Optional reference answers
            evaluator_names: Evaluators to run
            
        Returns:
            Evaluation results
        """
        # Generate responses using the RAG system
        datapoints = []
        
        for i, query in enumerate(test_queries):
            try:
                # Call the RAG system
                rag_response = await rag_query_function(query)
                
                # Extract response and contexts
                response = rag_response.get("answer", "")
                contexts = [source.get("content", "") for source in rag_response.get("sources", [])]
                
                reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None
                
                datapoint = EvaluationDatapoint(
                    query=query,
                    response=response,
                    reference=reference,
                    contexts=contexts,
                    metadata={
                        "query_index": i,
                        "source_count": len(contexts)
                    }
                )
                datapoints.append(datapoint)
                
            except Exception as e:
                logger.error(f"Error getting RAG response for query {i}: {e}")
                # Create a failed datapoint
                datapoint = EvaluationDatapoint(
                    query=query,
                    response="[ERROR: RAG system failed]",
                    reference=reference_answers[i] if reference_answers and i < len(reference_answers) else None,
                    contexts=[],
                    metadata={"query_index": i, "error": str(e)}
                )
                datapoints.append(datapoint)
        
        # Run evaluation
        results = await self.runner.run_evaluation(
            datapoints=datapoints,
            evaluator_names=evaluator_names
        )
        
        return {
            "datapoints": datapoints,
            "results": results,
            "summary": self.runner.get_summary_stats()
        }