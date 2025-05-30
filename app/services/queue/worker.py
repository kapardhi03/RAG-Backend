import asyncio
import json
import logging
import aioredis
import traceback
import signal
import sys
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.queue.redis_queue import (
    DOCUMENT_PROCESSING_QUEUE, 
    URL_PROCESSING_QUEUE,
    update_task_status
)
from app.services.document_processor.processor import DocumentProcessor
from app.services.vector_store.chroma import ChromaVectorStore
from app.db.session import init_db

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Create global instances
vector_store = None
document_processor = None
redis_client = None

async def process_document_task(task_data: Dict[str, Any]) -> bool:
    """
    Process a document task
    
    Args:
        task_data: Task data from queue
        
    Returns:
        True if successful, False otherwise
    """
    task_id = task_data.get("task_id")
    document_id = task_data.get("document_id")
    
    try:
        # Update task status to processing
        await update_task_status(redis_client, task_id, "processing")
        
        # Process document
        result = await document_processor.process_file(document_id)
        
        # Update task status to completed
        await update_task_status(
            redis_client, 
            task_id, 
            "completed", 
            {"success": True}
        )
        
        logger.info(f"Successfully processed document {document_id} (task {task_id})")
        return True
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Error processing document {document_id} (task {task_id}): {error_msg}\n{tb}")
        
        # Update task status to failed
        await update_task_status(
            redis_client, 
            task_id, 
            "failed", 
            {
                "error": error_msg,
                "traceback": tb
            }
        )
        return False

async def process_url_task(task_data: Dict[str, Any]) -> bool:
    """
    Process a URL task
    
    Args:
        task_data: Task data from queue
        
    Returns:
        True if successful, False otherwise
    """
    task_id = task_data.get("task_id")
    document_id = task_data.get("document_id")
    url = task_data.get("url")
    
    try:
        # Update task status to processing
        await update_task_status(redis_client, task_id, "processing")
        
        # Process URL
        result = await document_processor.process_url(document_id)
        
        # Update task status to completed
        await update_task_status(
            redis_client, 
            task_id, 
            "completed", 
            {"success": True}
        )
        
        logger.info(f"Successfully processed URL {url} for document {document_id} (task {task_id})")
        return True
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Error processing URL {url} for document {document_id} (task {task_id}): {error_msg}\n{tb}")
        
        # Update task status to failed
        await update_task_status(
            redis_client, 
            task_id, 
            "failed", 
            {
                "error": error_msg,
                "traceback": tb
            }
        )
        return False

async def document_worker():
    """
    Worker for processing document tasks
    """
    logger.info("Starting document worker...")
    
    while True:
        try:
            # Pop a task from the queue (blocking with timeout)
            task = await redis_client.brpop(DOCUMENT_PROCESSING_QUEUE, timeout=1)
            if task:
                _, task_json = task
                task_data = json.loads(task_json)
                
                # Process the task
                await process_document_task(task_data)
            else:
                # No tasks, sleep briefly
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Document worker shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in document worker: {str(e)}")
            await asyncio.sleep(1)  # Sleep to avoid tight loop on error

async def url_worker():
    """
    Worker for processing URL tasks
    """
    logger.info("Starting URL worker...")
    
    while True:
        try:
            # Pop a task from the queue (blocking with timeout)
            task = await redis_client.brpop(URL_PROCESSING_QUEUE, timeout=1)
            if task:
                _, task_json = task
                task_data = json.loads(task_json)
                
                # Process the task
                await process_url_task(task_data)
            else:
                # No tasks, sleep briefly
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("URL worker shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in URL worker: {str(e)}")
            await asyncio.sleep(1)  # Sleep to avoid tight loop on error

async def startup():
    """
    Initialize worker dependencies
    """
    global vector_store, document_processor, redis_client
    
    # Initialize database
    await init_db()
    
    # Initialize Redis client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize vector store
    vector_store = ChromaVectorStore(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)
    
    # Initialize document processor
    document_processor = DocumentProcessor(vector_store=vector_store)
    
    logger.info("Worker initialization complete")

async def shutdown():
    """
    Clean up resources
    """
    global redis_client
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    
    logger.info("Worker shutdown complete")

async def main():
    """
    Main worker function
    """
    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_signal()))
    
    try:
        # Initialize dependencies
        await startup()
        
        # Determine number of workers
        num_workers = settings.MAX_WORKERS
        
        # Start workers
        worker_tasks = []
        
        # Add document workers
        for i in range(num_workers // 2 or 1):
            worker_tasks.append(asyncio.create_task(document_worker()))
        
        # Add URL workers
        for i in range(num_workers // 2 or 1):
            worker_tasks.append(asyncio.create_task(url_worker()))
        
        logger.info(f"Started {len(worker_tasks)} worker tasks")
        
        # Wait for workers to complete (they run indefinitely)
        await asyncio.gather(*worker_tasks)
    
    except Exception as e:
        logger.error(f"Error in worker main: {str(e)}")
    finally:
        await shutdown()

async def shutdown_signal():
    """
    Handle shutdown signal
    """
    logger.info("Received shutdown signal")
    
    # Cancel all worker tasks
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    
    # Exit the process
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())