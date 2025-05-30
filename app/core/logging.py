# app/core/logging.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# Make sure logs directory exists
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Create query logs directory
query_logs_dir = logs_dir / "queries"
query_logs_dir.mkdir(exist_ok=True)

def setup_logging():
    """
    Configure logging for the application
    """
    # Set log level (INFO by default)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Main application log file
    file_handler = RotatingFileHandler(
        './logs/app.log',
        maxBytes=10485760,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    
    # Error log file
    error_file_handler = RotatingFileHandler(
        './logs/errors.log',
        maxBytes=10485760,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_file_handler)
    
    # RAG Queries log file
    query_logger = logging.getLogger("rag_queries")
    query_logger.setLevel(logging.INFO)
    query_logger.propagate = False  # Don't propagate to root logger
    
    query_handler = RotatingFileHandler(
        './logs/queries/rag_queries.log',
        maxBytes=10485760,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    query_handler.setFormatter(formatter)
    query_logger.addHandler(query_handler)
    
    # Set lower level for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info(f"Logging initialized with level: {log_level}")

def get_query_logger():
    """
    Get the specialized RAG query logger
    """
    return logging.getLogger("rag_queries")

def log_rag_query(query_text, kb_id, is_file_query=False, specific_file=None, results_count=0, execution_time_ms=0):
    """
    Log a RAG query with basic information
    """
    logger = get_query_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Basic log format
    log_message = (
        f"QUERY: '{query_text}' | "
        f"KB: {kb_id} | "
        f"RESULTS: {results_count} | "
        f"TIME: {execution_time_ms}ms"
    )
    
    # Add file-specific info if applicable
    if is_file_query:
        log_message += f" | FILE QUERY: {specific_file or 'unspecified'}"
    
    logger.info(log_message)