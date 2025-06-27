# tests/RAG-Evaluation/examples/basic_usage.py
"""
Basic usage examples for the RAG evaluation suite
"""
import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tests.RAG_Evaluation.core.base_evaluator import EvaluationDatapoint
from tests.RAG_Evaluation.runner.evaluation_runner import RAGEvaluationRunner
from llama_index.core.llms import OpenAI
from llama_index.core.embeddings import OpenAIEmbedding

async def basic_evaluation_example():
    """Basic evaluation example"""
    print("🎯 Basic RAG Evaluation Example")
    print("=" * 50)
    
    # Setup (replace with your API key)
    api_key = "your-openai-api-key"  # Replace with actual key
    
    if api_key == "your-openai-api-key":
        print("⚠️ Please set your OpenAI API key in the script")
        return
    
    # Initialize components
    llm = OpenAI(model="gpt-4-turbo-preview", api_key=api_key, temperature=0.0)
    embedding = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
    
    # Create evaluation runner
    runner = RAGEvaluationRunner(llm=llm, embedding_model=embedding)
    
    # Create sample data
    datapoints = [
        EvaluationDatapoint(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            reference="Paris is the capital and most populous city of France.",
            contexts=[
                "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents.",
                "France is a country in Western Europe with Paris as its capital city."
            ],
            metadata={"category": "geography"}
        ),
        EvaluationDatapoint(
            query="How does photosynthesis work?",
            response="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            reference="Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose.",
            contexts=[
                "Photosynthesis is the process by which plants and other organisms use sunlight to synthesize nutrients from carbon dioxide and water.",
                "The process occurs in chloroplasts and involves the conversion of carbon dioxide and water into glucose using light energy."
            ],
            metadata={"category": "biology"}
        )
    ]
    
    print(f"📋 Evaluating {len(datapoints)} datapoints...")
    
    # Run evaluation with default evaluators
    results = await runner.run_evaluation(
        datapoints=datapoints,
        evaluator_names=["correctness", "faithfulness", "answer_relevance"],
        save_results=True
    )
    
    # Print results
    print("\n📊 Results Summary:")
    for evaluator_name, evaluator_results in results.items():
        print(f"\n{evaluator_name}:")
        for result in evaluator_results:
            print(f"  Query: {result.query[:50]}...")
            print(f"  {result.metric_name}: {result.score:.3f}")
            print(f"  Feedback: {result.feedback[:100]}...")
            print()
    
    # Get summary statistics
    summary = runner.get_summary_stats()
    print("\n📈 Summary Statistics:")
    print(summary)

async def advanced_evaluation_example():
    """Advanced evaluation with custom evaluators"""
    print("\n🔬 Advanced RAG Evaluation Example")
    print("=" * 50)
    
    # Setup (replace with your API key)
    api_key = "your-openai-api-key"  # Replace with actual key
    
    if api_key == "your-openai-api-key":
        print("⚠️ Please set your OpenAI API key in the script")
        return
    
    # Initialize components
    llm = OpenAI(model="gpt-4-turbo-preview", api_key=api_key, temperature=0.0)
    embedding = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
    
    # Create evaluation runner
    runner = RAGEvaluationRunner(llm=llm, embedding_model=embedding)
    
    # Add custom evaluators
    custom_guidelines = """
    Evaluate responses based on:
    1. Accuracy: Is the information factually correct?
    2. Completeness: Does it address all parts of the question?
    3. Clarity: Is the explanation clear and well-structured?
    4. Relevance: Is the response directly relevant to the query?
    """
    
    runner.add_guideline_evaluator("custom_quality", custom_guidelines)
    runner.add_prometheus_evaluator()  # Uses LLM fallback
    runner.add_mt_bench_evaluator()    # Uses sample data
    
    # Create more complex datapoints
    datapoints = [
        EvaluationDatapoint(
            query="Compare renewable and non-renewable energy sources",
            response="Renewable energy sources like solar and wind are sustainable and environmentally friendly but can be intermittent. Non-renewable sources like fossil fuels are reliable but finite and polluting.",
            reference="Renewable energy sources are naturally replenished and include solar, wind, and hydro power. Non-renewable sources like coal, oil, and gas are finite and contribute to pollution.",
            contexts=[
                "Renewable energy comes from natural sources that are constantly replenished, such as sunlight, wind, rain, tides, waves, and geothermal heat.",
                "Non-renewable energy sources are fossil fuels: coal, petroleum, and natural gas. These energy sources are finite and will eventually be depleted.",
                "Advantages of renewable energy include sustainability and reduced environmental impact. Disadvantages include higher upfront costs and intermittency."
            ],
            metadata={"category": "environment", "complexity": "high"}
        )
    ]
    
    print(f"📋 Running advanced evaluation on {len(datapoints)} datapoints...")
    
    # Run evaluation with all evaluators
    results = await runner.run_evaluation(
        datapoints=datapoints,
        save_results=True
    )
    
    # Print detailed results
    print("\n📊 Detailed Results:")
    for evaluator_name, evaluator_results in results.items():
        print(f"\n{'='*20} {evaluator_name} {'='*20}")
        for result in evaluator_results:
            print(f"Metric: {result.metric_name}")
            print(f"Score: {result.score:.3f}")
            print(f"Query: {result.query}")
            print(f"Response: {result.response}")
            print(f"Feedback: {result.feedback}")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
            print("-" * 50)

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = [
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "reference": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
            "contexts": [
                "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience.",
                "Machine learning algorithms build models based on training data in order to make predictions or decisions without being explicitly programmed to do so."
            ],
            "metadata": {"category": "technology", "difficulty": "medium"}
        },
        {
            "query": "Explain the water cycle",
            "response": "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection processes on Earth.",
            "reference": "The water cycle describes how water moves continuously through the Earth's oceans, atmosphere, and land in a closed system.",
            "contexts": [
                "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth.",
                "The main processes of the water cycle include evaporation, transpiration, condensation, precipitation, and runoff."
            ],
            "metadata": {"category": "science", "difficulty": "easy"}
        }
    ]
    
    # Save to file
    with open("sample_evaluation_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample dataset created: sample_evaluation_data.json")

if __name__ == "__main__":
    print("RAG Evaluation Suite Examples")
    print("Choose an example to run:")
    print("1. Basic evaluation")
    print("2. Advanced evaluation")
    print("3. Create sample dataset")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(basic_evaluation_example())
    elif choice == "2":
        asyncio.run(advanced_evaluation_example())
    elif choice == "3":
        create_sample_dataset()
    else:
        print("Invalid choice")