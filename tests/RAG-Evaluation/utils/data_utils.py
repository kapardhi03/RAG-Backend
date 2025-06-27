# tests/RAG-Evaluation/utils/data_utils.py
"""
Data utilities for RAG evaluation
"""
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..core.base_evaluator import EvaluationDatapoint, EvaluationResult

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load datasets from various formats"""
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> List[EvaluationDatapoint]:
        """Load evaluation datapoints from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            datapoints = []
            for item in data:
                datapoint = EvaluationDatapoint(
                    query=item["query"],
                    response=item.get("response", ""),
                    reference=item.get("reference"),
                    contexts=item.get("contexts"),
                    metadata=item.get("metadata", {})
                )
                datapoints.append(datapoint)
            
            logger.info(f"Loaded {len(datapoints)} datapoints from {file_path}")
            return datapoints
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> List[EvaluationDatapoint]:
        """Load evaluation datapoints from CSV file"""
        try:
            df = pd.read_csv(file_path)
            datapoints = []
            
            for _, row in df.iterrows():
                # Handle contexts column
                contexts = None
                if 'contexts' in row and pd.notna(row['contexts']):
                    try:
                        # Try to parse as JSON array
                        contexts = json.loads(row['contexts'])
                    except json.JSONDecodeError:
                        # Fallback: split by delimiter or use as single context
                        contexts = [str(row['contexts'])]
                
                # Handle metadata column
                metadata = {}
                if 'metadata' in row and pd.notna(row['metadata']):
                    try:
                        metadata = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        metadata = {"raw_metadata": str(row['metadata'])}
                
                datapoint = EvaluationDatapoint(
                    query=row["query"],
                    response=row.get("response", ""),
                    reference=row.get("reference") if pd.notna(row.get("reference")) else None,
                    contexts=contexts,
                    metadata=metadata
                )
                datapoints.append(datapoint)
            
            logger.info(f"Loaded {len(datapoints)} datapoints from {file_path}")
            return datapoints
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_from_excel(file_path: Union[str, Path], sheet_name: str = None) -> List[EvaluationDatapoint]:
        """Load evaluation datapoints from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return DatasetLoader._dataframe_to_datapoints(df)
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise
    
    @staticmethod
    def _dataframe_to_datapoints(df: pd.DataFrame) -> List[EvaluationDatapoint]:
        """Convert DataFrame to evaluation datapoints"""
        datapoints = []
        
        for _, row in df.iterrows():
            # Handle contexts
            contexts = None
            if 'contexts' in row and pd.notna(row['contexts']):
                if isinstance(row['contexts'], str):
                    try:
                        contexts = json.loads(row['contexts'])
                    except json.JSONDecodeError:
                        contexts = [row['contexts']]
                elif isinstance(row['contexts'], list):
                    contexts = row['contexts']
            
            # Handle metadata
            metadata = {}
            for col in df.columns:
                if col not in ['query', 'response', 'reference', 'contexts']:
                    if pd.notna(row[col]):
                        metadata[col] = row[col]
            
            datapoint = EvaluationDatapoint(
                query=row["query"],
                response=row.get("response", ""),
                reference=row.get("reference") if pd.notna(row.get("reference")) else None,
                contexts=contexts,
                metadata=metadata
            )
            datapoints.append(datapoint)
        
        return datapoints

class DatasetSaver:
    """Save evaluation results to various formats"""
    
    @staticmethod
    def save_results_to_json(results: List[EvaluationResult], file_path: Union[str, Path]):
        """Save evaluation results to JSON file"""
        try:
            results_data = []
            for result in results:
                result_dict = {
                    "evaluator_name": result.evaluator_name,
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "feedback": result.feedback,
                    "query": result.query,
                    "response": result.response,
                    "reference": result.reference,
                    "contexts": result.contexts,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                }
                results_data.append(result_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(results)} results to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}")
            raise
    
    @staticmethod
    def save_results_to_csv(results: List[EvaluationResult], file_path: Union[str, Path]):
        """Save evaluation results to CSV file"""
        try:
            data = []
            for result in results:
                row = {
                    "evaluator_name": result.evaluator_name,
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "feedback": result.feedback,
                    "query": result.query,
                    "response": result.response,
                    "reference": result.reference,
                    "contexts": json.dumps(result.contexts) if result.contexts else None,
                    "metadata": json.dumps(result.metadata) if result.metadata else None,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Saved {len(results)} results to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            raise

class SampleDataGenerator:
    """Generate sample datasets for testing"""
    
    @staticmethod
    def create_basic_sample() -> List[EvaluationDatapoint]:
        """Create basic sample dataset"""
        return [
            EvaluationDatapoint(
                query="What is the capital of France?",
                response="The capital of France is Paris.",
                reference="Paris is the capital and most populous city of France.",
                contexts=[
                    "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents.",
                    "France is a country in Western Europe with Paris as its capital city."
                ],
                metadata={"category": "geography", "difficulty": "easy"}
            ),
            EvaluationDatapoint(
                query="How does photosynthesis work?",
                response="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                reference="Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose.",
                contexts=[
                    "Photosynthesis is the process by which plants and other organisms use sunlight to synthesize nutrients.",
                    "The process occurs in chloroplasts and involves the conversion of carbon dioxide and water into glucose using light energy."
                ],
                metadata={"category": "biology", "difficulty": "medium"}
            ),
            EvaluationDatapoint(
                query="Explain quantum computing principles",
                response="Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in quantum bits (qubits).",
                reference="Quantum computing leverages quantum mechanics to solve certain computational problems more efficiently than classical computers.",
                contexts=[
                    "Quantum computing is a type of computation that harnesses quantum mechanical phenomena.",
                    "Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously through superposition."
                ],
                metadata={"category": "technology", "difficulty": "hard"}
            )
        ]
    
    @staticmethod
    def create_domain_specific_sample(domain: str) -> List[EvaluationDatapoint]:
        """Create domain-specific sample dataset"""
        
        samples = {
            "medical": [
                EvaluationDatapoint(
                    query="What are the symptoms of diabetes?",
                    response="Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
                    reference="Diabetes symptoms include polydipsia (excessive thirst), polyuria (frequent urination), and fatigue.",
                    contexts=[
                        "Type 1 and Type 2 diabetes share common symptoms including increased thirst and urination.",
                        "Early diabetes symptoms may include fatigue, blurred vision, and slow-healing wounds."
                    ],
                    metadata={"domain": "medical", "condition": "diabetes"}
                )
            ],
            "legal": [
                EvaluationDatapoint(
                    query="What is intellectual property?",
                    response="Intellectual property refers to creations of the mind, including inventions, literary works, designs, and symbols used in commerce.",
                    reference="Intellectual property (IP) is a category of property that includes intangible creations of the human intellect.",
                    contexts=[
                        "Intellectual property law protects creators' rights over their inventions, creative works, and business innovations.",
                        "Common types of intellectual property include patents, copyrights, trademarks, and trade secrets."
                    ],
                    metadata={"domain": "legal", "area": "intellectual_property"}
                )
            ],
            "finance": [
                EvaluationDatapoint(
                    query="What is compound interest?",
                    response="Compound interest is interest calculated on the initial principal and also on the accumulated interest of previous periods.",
                    reference="Compound interest is the addition of interest to the principal sum of a loan or deposit.",
                    contexts=[
                        "Compound interest differs from simple interest in that it earns interest on previously earned interest.",
                        "The frequency of compounding affects the total amount of compound interest earned."
                    ],
                    metadata={"domain": "finance", "concept": "compound_interest"}
                )
            ]
        }
        
        return samples.get(domain, SampleDataGenerator.create_basic_sample())