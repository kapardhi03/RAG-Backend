RAG Evaluation Suite
===================

A comprehensive toolkit for evaluating Retrieval-Augmented Generation (RAG) systems using LlamaIndex.

## Features

- **Correctness Evaluation**: Compare responses against reference answers
- **Faithfulness Evaluation**: Ensure responses are faithful to retrieved contexts
- **Answer Relevance**: Measure how relevant answers are to queries
- **Guideline-based Evaluation**: Custom evaluation criteria
- **Benchmark Evaluators**: MT-Bench style evaluations
- **Pairwise Comparison**: Compare multiple responses
- **Embedding Similarity**: Semantic similarity metrics
- **Retrieval Quality**: Evaluate retrieval performance

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### 2. CLI Usage

```bash
# Interactive mode
python -m tests.RAG_Evaluation.interface.cli --interactive

# Evaluate from file
python -m tests.RAG_Evaluation.interface.cli --data sample_data.json --evaluators correctness,faithfulness

# Create sample data
python -m tests.RAG_Evaluation.interface.cli --create-sample
```

### 3. Web Interface

```bash
# Run Streamlit app
streamlit run tests/RAG-Evaluation/interface/web_interface.py
```

### 4. Programmatic Usage

```python
from tests.RAG_Evaluation.core.base_evaluator import EvaluationDatapoint
from tests.RAG_Evaluation.runner.evaluation_runner import RAGEvaluationRunner

# Create evaluation data
datapoints = [
    EvaluationDatapoint(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris is the capital of France.",
        contexts=["Paris is the capital and largest city of France."]
    )
]

# Run evaluation
runner = RAGEvaluationRunner()
results = await runner.run_evaluation(datapoints)
```

## Directory Structure

```
tests/RAG-Evaluation/
├── core/                    # Core evaluation components
│   ├── base_evaluator.py   # Base classes and interfaces
│   ├── correctness_evaluator.py
│   ├── faithfulness_evaluator.py
│   ├── relevance_evaluator.py
│   ├── guideline_evaluator.py
│   ├── pairwise_evaluator.py
│   ├── embedding_similarity_evaluator.py
│   ├── retrieval_evaluator.py
│   ├── benchmark_evaluators.py
│   └── prometheus_evaluator.py
├── runner/                  # Evaluation orchestration
│   ├── evaluation_runner.py
│   └── batch_evaluator.py
├── interface/              # User interfaces
│   ├── cli.py             # Command line interface
│   └── web_interface.py   # Streamlit web app
├── utils/                  # Utilities
│   ├── data_utils.py      # Data loading/saving
│   └── metrics_utils.py   # Additional metrics
├── config/                 # Configuration
│   └── settings.py
├── examples/              # Usage examples
│   └── basic_usage.py
└── README.md
```

## Evaluators

### Core Evaluators

1. **CorrectnessEvaluator**: Evaluates factual correctness against reference answers
2. **FaithfulnessEvaluator**: Checks if responses are faithful to retrieved contexts
3. **AnswerRelevanceEvaluator**: Measures relevance of answers to queries
4. **EmbeddingSimilarityEvaluator**: Semantic similarity using embeddings

### Advanced Evaluators

1. **GuidelineEvaluator**: Custom evaluation based on specific guidelines
2. **PairwiseEvaluator**: Compare multiple responses pairwise
3. **MTBenchEvaluator**: MT-Bench style human judgment evaluation
4. **MiniMTBenchEvaluator**: Simplified MT-Bench evaluation
5. **PrometheusEvaluator**: Prometheus model-based evaluation
6. **RetrievalEvaluator**: Evaluate retrieval quality (Hit Rate, MRR, NDCG)

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key  # Optional, for reranking
```

### Configuration File

Create `evaluation_config.json`:

```json
{
    "llm_model": "gpt-4-turbo-preview",
    "embedding_model": "text-embedding-3-large",
    "default_evaluators": ["correctness", "faithfulness", "answer_relevance"],
    "batch_size": 10,
    "results_dir": "evaluation_results"
}
```

## Data Format

### JSON Format

```json
[
    {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI...",
        "reference": "Machine learning is a method of data analysis...",
        "contexts": [
            "Machine learning is a subset of artificial intelligence...",
            "ML algorithms build models based on training data..."
        ],
        "metadata": {
            "category": "technology",
            "difficulty": "medium"
        }
    }
]
```

### CSV Format

| query | response | reference | contexts | metadata |
|-------|----------|-----------|----------|----------|
| What is ML? | ML is... | ML is a method... | ["context1", "context2"] | {"category": "tech"} |

## Advanced Features

### Custom Evaluators

```python
# Add custom guideline evaluator
guidelines = "Evaluate based on: 1. Accuracy 2. Completeness 3. Clarity"
runner.add_guideline_evaluator("custom_quality", guidelines)

# Add Prometheus evaluator
runner.add_prometheus_evaluator(endpoint="http://localhost:8000")

# Add retrieval evaluator
runner.add_retrieval_evaluator(metrics=["hit_rate", "mrr", "ndcg"])
```

### Batch Evaluation

```python
from tests.RAG_Evaluation.runner.batch_evaluator import BatchEvaluator

batch_evaluator = BatchEvaluator(runner)

# Evaluate your RAG system
async def my_rag_function(query):
    # Your RAG implementation
    return {"answer": "...", "sources": [...]}

results = await batch_evaluator.evaluate_rag_system(
    rag_query_function=my_rag_function,
    test_queries=["What is...?", "How does...?"],
    reference_answers=["Answer 1", "Answer 2"]
)
```

## Output

Results are saved in multiple formats:

- **CSV**: Detailed results with all metrics
- **JSON**: Structured results for programmatic access
- **Summary**: Statistical summary by evaluator and metric

### Sample Output

```
Evaluator        Metric           Count  Mean   Std    Min    Max
correctness      correctness_score   10  0.850  0.120  0.600  1.000
faithfulness     faithfulness_score  10  0.920  0.080  0.750  1.000
answer_relevance relevance_score     10  0.880  0.100  0.700  1.000
```

## Notes

- **MT-Bench Evaluator**: Uses sample data if full dataset not available
- **Prometheus Evaluator**: Falls back to LLM if Prometheus endpoint not available
- **Retrieval Evaluator**: Requires retrieval data in metadata
- **API Keys**: Most evaluators require OpenAI API key

## Contributing

To add new evaluators:

1. Inherit from `BaseRAGEvaluator`
2. Implement `evaluate()` and `get_evaluator_name()` methods
3. Add to `RAGEvaluationRunner`

Example:

```python
class CustomEvaluator(BaseRAGEvaluator):
    def get_evaluator_name(self) -> str:
        return "Custom"
    
    async def evaluate(self, datapoints: List[EvaluationDatapoint]) -> List[EvaluationResult]:
        # Your evaluation logic
        pass
```