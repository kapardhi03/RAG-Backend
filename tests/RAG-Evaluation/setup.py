# tests/RAG-Evaluation/setup.py
"""
Setup script for RAG Evaluation Suite
"""
from setuptools import setup, find_packages

setup(
    name="rag-evaluation-suite",
    version="1.0.0",
    description="Comprehensive RAG evaluation toolkit using LlamaIndex",
    packages=find_packages(),
    install_requires=[
        "llama-index>=0.11.20",
        "llama-index-core>=0.11.20",
        "llama-index-embeddings-openai>=0.2.5",
        "llama-index-llms-openai>=0.2.9",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "click>=8.1.0",
        "openai>=1.0.0",
        "numpy>=1.24.0"
    ],
    extras_require={
        "advanced": [
            "cohere>=5.0.0",
            "selenium>=4.0.0",
            "transformers>=4.30.0"
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rag-eval=tests.RAG_Evaluation.interface.cli:main",
        ],
    },
)