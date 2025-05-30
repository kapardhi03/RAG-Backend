import json
import logging
import aioredis
import uuid
from typing import Dict, Any, List, Optional
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

# Queue names
DOCUMENT_PROCESSING_QUEUE = "document_processing"
URL_PROCESSING_QUEUE = "url_processing"

async def init_redis_pool() -> aioredis.Redis:
    """Initialize Redis connection pool"""
    try:
        redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Test connection
        await redis.ping()
        logger.info(f"Connected to Redis at {settings.REDIS_URL}")
        return redis
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

async def close_redis_pool(redis: aioredis.Redis) -> None:
    """Close Redis connection pool"""
    try:
        await redis.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

async def add_document_processing_task(
    redis: aioredis.Redis, 
    document_id: str,
    tenant_id: str,
    kb_id: str,
    priority: int = 1
) -> str:
    """
    Add document processing task to the queue
    
    Args:
        redis: Redis client
        document_id: Document ID
        tenant_id: Tenant ID
        kb_id: Knowledge Base ID
        priority: Task priority (1-10, higher is more important)
        
    Returns:
        Task ID
    """
    try:
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "document_id": document_id, 
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "priority": priority,
            "type": "document"
        }
        
        # Add task to the processing queue
        await redis.lpush(DOCUMENT_PROCESSING_QUEUE, json.dumps(task_data))
        
        # Add task to the task set with status
        await redis.hset(f"task:{task_id}", mapping={
            "status": "pending",
            "document_id": document_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "type": "document",
            "created_at": asyncio.get_event_loop().time()
        })
        
        logger.info(f"Added document processing task {task_id} for document {document_id}")
        return task_id
    except Exception as e:
        logger.error(f"Error adding document processing task: {str(e)}")
        raise

async def add_url_processing_task(
    redis: aioredis.Redis, 
    document_id: str,
    tenant_id: str,
    kb_id: str,
    url: str,
    priority: int = 1
) -> str:
    """
    Add URL processing task to the queue
    
    Args:
        redis: Redis client
        document_id: Document ID
        tenant_id: Tenant ID
        kb_id: Knowledge Base ID
        url: URL to process
        priority: Task priority (1-10, higher is more important)
        
    Returns:
        Task ID
    """
    try:
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "document_id": document_id, 
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "url": url,
            "priority": priority,
            "type": "url"
        }
        
        # Add task to the processing queue
        await redis.lpush(URL_PROCESSING_QUEUE, json.dumps(task_data))
        
        # Add task to the task set with status
        await redis.hset(f"task:{task_id}", mapping={
            "status": "pending",
            "document_id": document_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "url": url,
            "type": "url",
            "created_at": asyncio.get_event_loop().time()
        })
        
        logger.info(f"Added URL processing task {task_id} for document {document_id}, URL: {url}")
        return task_id
    except Exception as e:
        logger.error(f"Error adding URL processing task: {str(e)}")
        raise

async def get_task_status(redis: aioredis.Redis, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task
    
    Args:
        redis: Redis client
        task_id: Task ID
    
    Returns:
        Task status information or None if not found
    """
    try:
        task_data = await redis.hgetall(f"task:{task_id}")
        return task_data if task_data else None
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return None

async def update_task_status(
    redis: aioredis.Redis, 
    task_id: str, 
    status: str, 
    result: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update the status of a task
    
    Args:
        redis: Redis client
        task_id: Task ID
        status: New status (pending, processing, completed, failed)
        result: Optional result data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if task exists
        task_exists = await redis.exists(f"task:{task_id}")
        if not task_exists:
            logger.warning(f"Attempted to update non-existent task {task_id}")
            return False
        
        # Update status
        await redis.hset(f"task:{task_id}", "status", status)
        if status == "completed" or status == "failed":
            await redis.hset(f"task:{task_id}", "completed_at", asyncio.get_event_loop().time())
        
        # If result provided, update result
        if result:
            await redis.hset(f"task:{task_id}", "result", json.dumps(result))
        
        logger.info(f"Updated task {task_id} status to {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating task status: {str(e)}")
        return False

# This module will be expanded with worker implementations for processing tasks