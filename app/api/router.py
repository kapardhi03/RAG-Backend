from fastapi import APIRouter
from app.api.endpoints import auth, knowledge_base, documents, urls, query, vectors

# Create API router
api_router = APIRouter()

# Include routers from endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(knowledge_base.router, prefix="/kb", tags=["Knowledge Base"])
api_router.include_router(documents.router, prefix="/kb", tags=["Documents"])
api_router.include_router(urls.router, prefix="/urls", tags=["URLs"])
api_router.include_router(vectors.router, prefix="/kb", tags=["vectors"])
api_router.include_router(query.router, prefix="/kb", tags=["Query"])