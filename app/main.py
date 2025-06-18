# app/main.py - Updated with enhanced initialization
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.deps import cleanup_services, get_rag_engine, get_document_processor
# Initialize logging first
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade RAG Microservices API with LlamaIndex",
    version="2.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Startup tasks with service initialization"""
    logger.info("=== Starting RAG Microservices Application ===")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        from app.db.session import init_db
        await init_db()
        logger.info("✓ Database initialization completed")
        
        # Initialize core services
        logger.info("Initializing core services...")
        
        # Initialize RAG engine (this will create embeddings and LLM)
        logger.info("Initializing RAG engine...")
        rag_engine = get_rag_engine()
        logger.info(f"✓ RAG engine initialized with models: {rag_engine.embedding_model}, {rag_engine.llm_model}")
        
        # Initialize document processor
        logger.info("Initializing document processor...")
        doc_processor = get_document_processor()
        logger.info("✓ Document processor initialized")
        
        # Log configuration
        logger.info("=== Current Configuration ===")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Default Embedding Model: {settings.DEFAULT_EMBEDDING_MODEL}")
        logger.info(f"Default LLM Model: {settings.DEFAULT_LLM_MODEL}")
        logger.info(f"Semantic Chunking: {settings.ENABLE_SEMANTIC_CHUNKING}")
        logger.info(f"Hybrid Search: {settings.ENABLE_HYBRID_SEARCH}")
        logger.info(f"Query Expansion: {settings.ENABLE_QUERY_EXPANSION}")
        logger.info(f"Reranking: {settings.ENABLE_RERANKING}")
        logger.info(f"Advanced Scraping: {settings.ENABLE_ADVANCED_SCRAPING}")
        logger.info("=============================")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f" Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks"""
    logger.info("=== Shutting down RAG Microservices Application ===")
    
    try:
        # Cleanup services
        await cleanup_services()
        logger.info("✓ Services cleaned up")
        
        logger.info("👋 Application shutdown completed")
        
    except Exception as e:
        logger.error(f" Error during shutdown: {str(e)}")

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    error_msg = f"Unhandled exception: {str(exc)}"
    logger.error(error_msg, exc_info=True)
    logger.error(f"Error occurred at path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "environment": settings.ENVIRONMENT
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    
    try:
        # Check database
        from app.db.session import get_db
        db_status = "healthy"
        try:
            async for db in get_db():
                await db.execute("SELECT 1")
                break
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check RAG engine
        rag_status = "healthy"
        try:
            rag_engine = get_rag_engine()
            available_models = rag_engine.get_available_models()
        except Exception as e:
            rag_status = f"error: {str(e)}"
            available_models = {}
        
        # Check S3 storage
        s3_status = "healthy"
        try:
            from app.api.deps import get_s3_storage
            s3_storage = get_s3_storage()
        except Exception as e:
            s3_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": time.time(),
            "environment": settings.ENVIRONMENT,
            "services": {
                "database": db_status,
                "rag_engine": rag_status,
                "s3_storage": s3_status
            },
            "configuration": {
                "default_embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
                "default_llm_model": settings.DEFAULT_LLM_MODEL,
                "semantic_chunking": settings.ENABLE_SEMANTIC_CHUNKING,
                "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
                "query_expansion": settings.ENABLE_QUERY_EXPANSION,
                "reranking": settings.ENABLE_RERANKING,
                "advanced_scraping": settings.ENABLE_ADVANCED_SCRAPING
            },
            "available_models": available_models,
            "performance": {
                "default_chunk_size": settings.DEFAULT_CHUNK_SIZE,
                "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
                "retrieval_timeout": settings.RETRIEVAL_TIMEOUT,
                "generation_timeout": settings.GENERATION_TIMEOUT
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    
    try:
        # Get basic metrics
        from app.api.deps import get_query_engine
        query_engine = get_query_engine()
        
        metrics = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - startup_time if 'startup_time' in globals() else 0,
            "engine_stats": query_engine.get_engine_stats(),
            "system_config": {
                "chunk_size": settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
                "max_workers": settings.MAX_WORKERS,
                "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e), "timestamp": time.time()}

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Microservices API v2.0",
        "description": "Production-grade RAG system with LlamaIndex",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "detailed_health": "/health/detailed",
        "metrics": "/metrics",
        "features": [
            "Advanced chunking with semantic splitting",
            "Hybrid search with vector and keyword",
            "Query expansion and intent analysis",
            "Configurable embedding and LLM models",
            "Advanced web scraping with anti-blocking",
            "Re-ranking with Cohere",
            "Batch query processing",
            "Real-time processing pipeline"
        ]
    }

# Store startup time for metrics
startup_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )