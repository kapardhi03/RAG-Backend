# tests/RAG-Evaluation/integration/rag_system_evaluator.py
"""
Integration example with your existing RAG system
This shows how to evaluate your current RAG pipeline without modifying it
"""
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
import logging

from ..core.base_evaluator import EvaluationDatapoint
from ..runner.evaluation_runner import RAGEvaluationRunner
from ..runner.batch_evaluator import BatchEvaluator

logger = logging.getLogger(__name__)

class ExistingRAGSystemEvaluator:
    """Evaluate your existing RAG system via API calls"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000/api",
                 auth_token: Optional[str] = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def call_rag_system(self, query: str, kb_id: str) -> Dict[str, Any]:
        """
        Call your existing RAG system
        Adjust this method to match your API endpoints
        """
        try:
            # Example call to your existing /api/kb/{kb_id}/chat endpoint
            url = f"{self.base_url}/kb/{kb_id}/chat"
            payload = {
                "query": query,
                "top_k": 5
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "answer": data.get("answer", ""),
                        "sources": data.get("sources", []),
                        "success": True
                    }
                else:
                    logger.error(f"RAG API error: {response.status}")
                    return {
                        "answer": "",
                        "sources": [],
                        "success": False,
                        "error": f"API error: {response.status}"
                    }
        except Exception as e:
            logger.error(f"Error calling RAG system: {e}")
            return {
                "answer": "",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    async def evaluate_system_performance(self,
                                        kb_id: str,
                                        test_queries: List[str],
                                        reference_answers: Optional[List[str]] = None,
                                        evaluator_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate your RAG system end-to-end
        
        Args:
            kb_id: Knowledge base ID to test
            test_queries: List of test questions
            reference_answers: Optional ground truth answers
            evaluator_names: Evaluators to run
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"🔄 Evaluating RAG system for KB: {kb_id}")
        print(f"📝 Running {len(test_queries)} test queries...")
        
        # Generate responses from your RAG system
        datapoints = []
        successful_queries = 0
        
        for i, query in enumerate(test_queries):
            print(f"Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            rag_response = await self.call_rag_system(query, kb_id)
            
            if rag_response["success"]:
                successful_queries += 1
                
                # Extract contexts from sources
                contexts = []
                for source in rag_response["sources"]:
                    if isinstance(source, dict):
                        contexts.append(source.get("content", ""))
                    else:
                        contexts.append(str(source))
                
                reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None
                
                datapoint = EvaluationDatapoint(
                    query=query,
                    response=rag_response["answer"],
                    reference=reference,
                    contexts=contexts,
                    metadata={
                        "kb_id": kb_id,
                        "query_index": i,
                        "source_count": len(contexts),
                        "api_success": True
                    }
                )
            else:
                # Create failed datapoint for analysis
                datapoint = EvaluationDatapoint(
                    query=query,
                    response="[RAG SYSTEM ERROR]",
                    reference=reference_answers[i] if reference_answers and i < len(reference_answers) else None,
                    contexts=[],
                    metadata={
                        "kb_id": kb_id,
                        "query_index": i,
                        "source_count": 0,
                        "api_success": False,
                        "error": rag_response.get("error", "Unknown error")
                    }
                )
            
            datapoints.append(datapoint)
        
        print(f"✅ Generated {successful_queries}/{len(test_queries)} successful responses")
        
        # Run evaluation
        runner = RAGEvaluationRunner()
        
        # Add custom evaluators if needed
        if successful_queries > 0:
            print("🔧 Running evaluation...")
            results = await runner.run_evaluation(
                datapoints=datapoints,
                evaluator_names=evaluator_names,
                save_results=True,
                results_dir=f"evaluation_results/kb_{kb_id}"
            )
            
            # Calculate system-level metrics
            system_metrics = self._calculate_system_metrics(datapoints, results)
            
            return {
                "kb_id": kb_id,
                "total_queries": len(test_queries),
                "successful_queries": successful_queries,
                "success_rate": successful_queries / len(test_queries),
                "datapoints": datapoints,
                "evaluation_results": results,
                "system_metrics": system_metrics,
                "summary": runner.get_summary_stats()
            }
        else:
            print("❌ No successful queries to evaluate")
            return {
                "kb_id": kb_id,
                "total_queries": len(test_queries),
                "successful_queries": 0,
                "success_rate": 0.0,
                "error": "No successful RAG responses"
            }
    
    def _calculate_system_metrics(self, datapoints: List[EvaluationDatapoint], 
                                results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system-level performance metrics"""
        
        # Basic system health metrics
        total_queries = len(datapoints)
        successful_queries = sum(1 for dp in datapoints if dp.metadata.get("api_success", False))
        
        # Response quality metrics
        avg_response_length = sum(len(dp.response) for dp in datapoints) / total_queries
        avg_context_count = sum(len(dp.contexts) if dp.contexts else 0 for dp in datapoints) / total_queries
        
        # Evaluation score aggregation
        all_scores = []
        evaluator_averages = {}
        
        for evaluator_name, evaluator_results in results.items():
            scores = [r.score for r in evaluator_results]
            if scores:
                evaluator_averages[evaluator_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
                all_scores.extend(scores)
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "system_availability": successful_queries / total_queries,
            "overall_quality_score": overall_score,
            "avg_response_length": avg_response_length,
            "avg_context_count": avg_context_count,
            "evaluator_breakdown": evaluator_averages,
            "total_evaluated_responses": len(all_scores)
        }

# Example usage and test scenarios
class RAGSystemTestSuite:
    """Pre-built test suites for different scenarios"""
    
    @staticmethod
    def get_general_knowledge_tests() -> List[str]:
        """General knowledge test queries"""
        return [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What are the main causes of climate change?",
            "Explain the concept of machine learning",
            "What is the difference between DNA and RNA?",
            "How do vaccines work?",
            "What is quantum computing?",
            "Explain the water cycle",
            "What causes earthquakes?",
            "How do solar panels generate electricity?"
        ]
    
    @staticmethod
    def get_reasoning_tests() -> List[str]:
        """Reasoning and analysis test queries"""
        return [
            "Compare the advantages and disadvantages of renewable energy",
            "What are the potential impacts of artificial intelligence on employment?",
            "Analyze the causes of the 2008 financial crisis",
            "Explain the relationship between inflation and unemployment",
            "What factors contribute to successful startup companies?",
            "How has the internet changed global communication?",
            "What are the ethical implications of genetic engineering?",
            "Compare different approaches to education reform",
            "Analyze the impact of social media on democracy",
            "What strategies can address income inequality?"
        ]
    
    @staticmethod
    def get_domain_specific_tests(domain: str) -> List[str]:
        """Domain-specific test queries"""
        
        domain_tests = {
            "medical": [
                "What are the symptoms of diabetes?",
                "How is blood pressure measured?",
                "What is the difference between bacteria and viruses?",
                "Explain how antibiotics work",
                "What are the stages of wound healing?"
            ],
            "legal": [
                "What is intellectual property?",
                "Explain the concept of due process",
                "What are the elements of a valid contract?",
                "How does copyright law work?",
                "What is the difference between civil and criminal law?"
            ],
            "finance": [
                "What is compound interest?",
                "How do stock markets work?",
                "What is the difference between debt and equity?",
                "Explain diversification in investing",
                "How is credit score calculated?"
            ],
            "technology": [
                "What is cloud computing?",
                "How does blockchain technology work?",
                "What is the difference between HTTP and HTTPS?",
                "Explain database normalization",
                "How do neural networks learn?"
            ]
        }
        
        return domain_tests.get(domain, RAGSystemTestSuite.get_general_knowledge_tests())

# Integration script example
async def evaluate_existing_rag_system():
    """Example script to evaluate your existing RAG system"""
    
    print("🎯 RAG System Evaluation Script")
    print("=" * 50)
    
    # Configuration
    KB_ID = "your-knowledge-base-id"  # Replace with your actual KB ID
    API_BASE_URL = "http://localhost:8000/api"  # Your API endpoint
    AUTH_TOKEN = "your-auth-token"  # Your authentication token
    
    # Test scenarios
    test_scenarios = {
        "general_knowledge": {
            "queries": RAGSystemTestSuite.get_general_knowledge_tests()[:5],  # Limit for demo
            "evaluators": ["correctness", "faithfulness", "answer_relevance"]
        },
        "reasoning": {
            "queries": RAGSystemTestSuite.get_reasoning_tests()[:3],
            "evaluators": ["faithfulness", "answer_relevance", "mini_mt_bench"]
        }
    }
    
    async with ExistingRAGSystemEvaluator(API_BASE_URL, AUTH_TOKEN) as evaluator:
        
        all_results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            print(f"\n🔍 Running {scenario_name} evaluation...")
            
            try:
                results = await evaluator.evaluate_system_performance(
                    kb_id=KB_ID,
                    test_queries=scenario_config["queries"],
                    evaluator_names=scenario_config["evaluators"]
                )
                
                all_results[scenario_name] = results
                
                # Print summary
                if "system_metrics" in results:
                    metrics = results["system_metrics"]
                    print(f"✅ {scenario_name} Results:")
                    print(f"   Success Rate: {results['success_rate']:.1%}")
                    print(f"   Overall Quality: {metrics['overall_quality_score']:.3f}")
                    print(f"   Avg Response Length: {metrics['avg_response_length']:.0f} chars")
                    print(f"   Avg Context Count: {metrics['avg_context_count']:.1f}")
                
            except Exception as e:
                print(f"❌ Error in {scenario_name}: {e}")
                all_results[scenario_name] = {"error": str(e)}
        
        # Overall summary
        print(f"\n📊 Overall Evaluation Summary")
        print("=" * 50)
        
        successful_scenarios = [name for name, result in all_results.items() if "error" not in result]
        
        if successful_scenarios:
            # Calculate aggregate metrics
            total_queries = sum(all_results[name]["total_queries"] for name in successful_scenarios)
            total_successful = sum(all_results[name]["successful_queries"] for name in successful_scenarios)
            
            overall_success_rate = total_successful / total_queries if total_queries > 0 else 0
            
            print(f"Scenarios Evaluated: {len(successful_scenarios)}/{len(test_scenarios)}")
            print(f"Total Queries: {total_queries}")
            print(f"Overall Success Rate: {overall_success_rate:.1%}")
            
            # Quality breakdown by evaluator
            print(f"\n📈 Quality Metrics by Evaluator:")
            evaluator_scores = {}
            
            for scenario_name in successful_scenarios:
                if "system_metrics" in all_results[scenario_name]:
                    breakdown = all_results[scenario_name]["system_metrics"]["evaluator_breakdown"]
                    for evaluator, metrics in breakdown.items():
                        if evaluator not in evaluator_scores:
                            evaluator_scores[evaluator] = []
                        evaluator_scores[evaluator].append(metrics["mean"])
            
            for evaluator, scores in evaluator_scores.items():
                avg_score = sum(scores) / len(scores)
                print(f"   {evaluator}: {avg_score:.3f}")
        
        else:
            print("❌ No successful evaluations completed")
        
        return all_results

# Standalone evaluation script
if __name__ == "__main__":
    # Run the evaluation
    results = asyncio.run(evaluate_existing_rag_system())
    
    # Save results
    import json
    with open("rag_system_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: rag_system_evaluation_results.json")