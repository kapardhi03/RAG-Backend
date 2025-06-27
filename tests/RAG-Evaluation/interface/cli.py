# tests/RAG-Evaluation/interface/cli.py
"""
Command Line Interface for RAG Evaluation Suite
"""
import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.base_evaluator import EvaluationDatapoint
from runner.evaluation_runner import RAGEvaluationRunner
from runner.batch_evaluator import BatchEvaluator
from llama_index.core.llms import OpenAI
from llama_index.core.embeddings import OpenAIEmbedding

class RAGEvaluationCLI:
    """Command Line Interface for RAG evaluation"""
    
    def __init__(self):
        self.runner = None
        self.batch_evaluator = None
    
    def setup_runner(self, 
                    openai_api_key: Optional[str] = None,
                    llm_model: str = "gpt-4-turbo-preview",
                    embedding_model: str = "text-embedding-3-large"):
        """Setup the evaluation runner"""
        
        # Use provided API key or get from environment
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: No OpenAI API key provided. Some evaluators may not work.")
        
        # Initialize LLM and embedding model
        llm = OpenAI(model=llm_model, api_key=api_key, temperature=0.0) if api_key else None
        embedding = OpenAIEmbedding(model=embedding_model, api_key=api_key) if api_key else None
        
        # Create runner
        self.runner = RAGEvaluationRunner(llm=llm, embedding_model=embedding)
        self.batch_evaluator = BatchEvaluator(self.runner)
        
        print(f"✅ Evaluation runner initialized with {llm_model} and {embedding_model}")
    
    def create_sample_data(self, output_file: str = "sample_evaluation_data.json"):
        """Create sample evaluation data"""
        sample_data = [
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "reference": "Paris is the capital and most populous city of France.",
                "contexts": [
                    "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents.",
                    "France is a country in Western Europe with Paris as its capital city."
                ],
                "metadata": {"category": "geography", "difficulty": "easy"}
            },
            {
                "query": "How does photosynthesis work?",
                "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                "reference": "Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose.",
                "contexts": [
                    "Photosynthesis is the process by which plants and other organisms use sunlight to synthesize nutrients.",
                    "The process occurs in chloroplasts and involves the conversion of carbon dioxide and water into glucose using light energy."
                ],
                "metadata": {"category": "biology", "difficulty": "medium"}
            },
            {
                "query": "Explain quantum computing principles",
                "response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in quantum bits (qubits).",
                "reference": "Quantum computing leverages quantum mechanics to solve certain computational problems more efficiently than classical computers.",
                "contexts": [
                    "Quantum computing is a type of computation that harnesses quantum mechanical phenomena.",
                    "Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously through superposition."
                ],
                "metadata": {"category": "technology", "difficulty": "hard"}
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Sample data created: {output_file}")
        return output_file
    
    async def run_basic_evaluation(self, 
                                  data_file: str,
                                  evaluators: Optional[List[str]] = None,
                                  output_dir: str = "evaluation_results"):
        """Run basic evaluation on data file"""
        
        if not self.runner:
            print("❌ Runner not initialized. Call setup_runner() first.")
            return
        
        print(f"📊 Loading data from {data_file}...")
        
        # Load data
        if data_file.endswith('.json'):
            datapoints = BatchEvaluator.load_datapoints_from_json(data_file)
        elif data_file.endswith('.csv'):
            datapoints = BatchEvaluator.load_datapoints_from_csv(data_file)
        else:
            print("❌ Unsupported file format. Use .json or .csv")
            return
        
        print(f"📋 Loaded {len(datapoints)} datapoints")
        
        # Run evaluation
        print(f"🔄 Running evaluation with evaluators: {evaluators or 'all available'}")
        
        results = await self.runner.run_evaluation(
            datapoints=datapoints,
            evaluator_names=evaluators,
            results_dir=output_dir
        )
        
        # Print summary
        print("\n📈 Evaluation Summary:")
        summary = self.runner.get_summary_stats()
        print(summary)
        
        print(f"\n✅ Results saved to {output_dir}/")
        
        return results
    
    async def run_custom_evaluation(self):
        """Run interactive custom evaluation"""
        
        if not self.runner:
            print("❌ Runner not initialized. Call setup_runner() first.")
            return
        
        print("\n🎯 Custom Evaluation Mode")
        print("Enter your evaluation data interactively.")
        
        datapoints = []
        
        while True:
            print(f"\n📝 Datapoint {len(datapoints) + 1}:")
            
            query = input("Query: ").strip()
            if not query:
                break
                
            response = input("Response: ").strip()
            reference = input("Reference (optional): ").strip() or None
            
            # Contexts
            contexts = []
            print("Contexts (press Enter twice to finish):")
            while True:
                context = input(f"Context {len(contexts) + 1}: ").strip()
                if not context:
                    break
                contexts.append(context)
            
            datapoint = EvaluationDatapoint(
                query=query,
                response=response,
                reference=reference,
                contexts=contexts if contexts else None,
                metadata={"source": "interactive"}
            )
            datapoints.append(datapoint)
            
            continue_input = input("\nAdd another datapoint? (y/N): ").strip().lower()
            if continue_input != 'y':
                break
        
        if not datapoints:
            print("No datapoints provided.")
            return
        
        # Select evaluators
        print(f"\n🔧 Available evaluators: {list(self.runner.evaluators.keys())}")
        evaluator_input = input("Select evaluators (comma-separated, or 'all'): ").strip()
        
        if evaluator_input.lower() == 'all':
            evaluators = None
        else:
            evaluators = [e.strip() for e in evaluator_input.split(',') if e.strip()]
        
        # Run evaluation
        print(f"\n🔄 Running evaluation...")
        results = await self.runner.run_evaluation(
            datapoints=datapoints,
            evaluator_names=evaluators
        )
        
        # Show results
        print("\n📊 Results:")
        for evaluator_name, evaluator_results in results.items():
            print(f"\n{evaluator_name}:")
            for result in evaluator_results:
                print(f"  {result.metric_name}: {result.score:.3f}")
                if len(result.feedback) < 100:
                    print(f"  Feedback: {result.feedback}")
        
        return results
    
    def add_custom_evaluators(self):
        """Add custom evaluators interactively"""
        
        if not self.runner:
            print("❌ Runner not initialized. Call setup_runner() first.")
            return
        
        print("\n🛠️  Custom Evaluator Setup")
        
        # Guideline evaluator
        add_guideline = input("Add guideline-based evaluator? (y/N): ").strip().lower()
        if add_guideline == 'y':
            name = input("Evaluator name: ").strip()
            print("Enter guidelines (press Enter twice to finish):")
            guidelines = []
            while True:
                line = input()
                if not line:
                    break
                guidelines.append(line)
            
            guidelines_text = '\n'.join(guidelines)
            self.runner.add_guideline_evaluator(name, guidelines_text)
            print(f"✅ Added guideline evaluator: {name}")
        
        # Prometheus evaluator
        add_prometheus = input("Add Prometheus evaluator? (y/N): ").strip().lower()
        if add_prometheus == 'y':
            endpoint = input("Prometheus endpoint (optional): ").strip() or None
            api_key = input("API key (optional): ").strip() or None
            self.runner.add_prometheus_evaluator(endpoint, api_key)
            print("✅ Added Prometheus evaluator")
        
        # MT-Bench evaluator
        add_mtbench = input("Add MT-Bench evaluator? (y/N): ").strip().lower()
        if add_mtbench == 'y':
            data_path = input("MT-Bench data path (optional): ").strip() or None
            self.runner.add_mt_bench_evaluator(data_path)
            print("✅ Added MT-Bench evaluator")
        
        # Retrieval evaluator
        add_retrieval = input("Add retrieval evaluator? (y/N): ").strip().lower()
        if add_retrieval == 'y':
            metrics_input = input("Metrics (comma-separated, default: hit_rate,mrr,ndcg): ").strip()
            metrics = [m.strip() for m in metrics_input.split(',')] if metrics_input else None
            self.runner.add_retrieval_evaluator(metrics)
            print("✅ Added retrieval evaluator")
    
    def list_evaluators(self):
        """List available evaluators"""
        if not self.runner:
            print("❌ Runner not initialized. Call setup_runner() first.")
            return
        
        print("\n📋 Available Evaluators:")
        for name, evaluator in self.runner.evaluators.items():
            print(f"  • {name}: {evaluator.__class__.__name__}")
    
    async def run_interactive_mode(self):
        """Run interactive mode"""
        print("🎯 RAG Evaluation Suite - Interactive Mode")
        print("=" * 50)
        
        # Setup
        api_key = input("OpenAI API Key (or press Enter to use env): ").strip()
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        llm_model = input("LLM model (default: gpt-4-turbo-preview): ").strip() or "gpt-4-turbo-preview"
        embedding_model = input("Embedding model (default: text-embedding-3-large): ").strip() or "text-embedding-3-large"
        
        self.setup_runner(api_key, llm_model, embedding_model)
        
        while True:
            print("\n" + "=" * 50)
            print("Main Menu:")
            print("1. Create sample data")
            print("2. Run evaluation from file")
            print("3. Run custom evaluation")
            print("4. Add custom evaluators")
            print("5. List available evaluators")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                filename = input("Output filename (default: sample_data.json): ").strip() or "sample_data.json"
                self.create_sample_data(filename)
                
            elif choice == "2":
                data_file = input("Data file path: ").strip()
                if not os.path.exists(data_file):
                    print(f"❌ File not found: {data_file}")
                    continue
                
                evaluators_input = input("Evaluators (comma-separated, or 'all'): ").strip()
                evaluators = None if evaluators_input.lower() == 'all' else [e.strip() for e in evaluators_input.split(',')]
                
                output_dir = input("Output directory (default: evaluation_results): ").strip() or "evaluation_results"
                
                await self.run_basic_evaluation(data_file, evaluators, output_dir)
                
            elif choice == "3":
                await self.run_custom_evaluation()
                
            elif choice == "4":
                self.add_custom_evaluators()
                
            elif choice == "5":
                self.list_evaluators()
                
            elif choice == "6":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite CLI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--data", "-d", help="Data file for evaluation")
    parser.add_argument("--evaluators", "-e", help="Comma-separated list of evaluators")
    parser.add_argument("--output", "-o", default="evaluation_results", help="Output directory")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data file")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--llm-model", default="gpt-4-turbo-preview", help="LLM model")
    parser.add_argument("--embedding-model", default="text-embedding-3-large", help="Embedding model")
    
    args = parser.parse_args()
    
    cli = RAGEvaluationCLI()
    
    if args.interactive:
        asyncio.run(cli.run_interactive_mode())
    elif args.create_sample:
        filename = input("Output filename (default: sample_data.json): ").strip() or "sample_data.json"
        cli.create_sample_data(filename)
    elif args.data:
        cli.setup_runner(args.api_key, args.llm_model, args.embedding_model)
        evaluators = [e.strip() for e in args.evaluators.split(',')] if args.evaluators else None
        asyncio.run(cli.run_basic_evaluation(args.data, evaluators, args.output))
    else:
        print("Use --interactive for interactive mode or --help for options")

if __name__ == "__main__":
    main()