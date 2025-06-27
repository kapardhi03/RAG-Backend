# tests/RAG-Evaluation/config/settings.py
"""
Configuration settings for RAG evaluation
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation"""
    
    # API Configuration
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Model Configuration
    llm_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    
    # Evaluation Configuration
    default_evaluators: List[str] = None
    batch_size: int = 10
    max_retries: int = 3
    timeout: int = 60
    
    # Output Configuration
    results_dir: str = "evaluation_results"
    save_detailed_results: bool = True
    save_summary_stats: bool = True
    
    # Advanced Configuration
    use_cache: bool = True
    cache_dir: str = ".evaluation_cache"
    parallel_evaluation: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.default_evaluators is None:
            self.default_evaluators = [
                "correctness", 
                "faithfulness", 
                "answer_relevance", 
                "embedding_similarity"
            ]
        
        # Load from environment if not set
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.cohere_api_key:
            self.cohere_api_key = os.getenv("COHERE_API_KEY")

class ConfigManager:
    """Manage evaluation configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> EvaluationConfig:
        """Load configuration from file or environment"""
        if self.config_file and os.path.exists(self.config_file):
            import json
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            return EvaluationConfig(**config_data)
        else:
            return EvaluationConfig()
    
    def save_config(self, config: EvaluationConfig):
        """Save configuration to file"""
        if not self.config_file:
            self.config_file = "evaluation_config.json"
        
        import json
        from dataclasses import asdict
        
        with open(self.config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def get_config(self) -> EvaluationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)