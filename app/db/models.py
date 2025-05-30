#db.models.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from passlib.context import CryptContext

from app.db.base_class import Base

# Setup passlib context for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Tenant(Base):
    """Tenant model for authentication and multi-tenancy"""
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    knowledge_bases = relationship("KnowledgeBase", back_populates="tenant", cascade="all, delete-orphan")
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash from plaintext password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str) -> bool:
        """Verify plaintext password against stored hash"""
        return pwd_context.verify(plain_password, self.hashed_password)
    
    @classmethod
    async def get_by_email(cls, email: str, db: AsyncSession) -> "Tenant":
        """Get tenant by email"""
        query = select(cls).filter(cls.email == email)
        result = await db.execute(query)
        return result.scalars().first()
    
    @classmethod
    async def get_by_id(cls, tenant_id: uuid.UUID, db: AsyncSession) -> "Tenant":
        """Get tenant by ID"""
        query = select(cls).filter(cls.id == tenant_id)
        result = await db.execute(query)
        return result.scalars().first()

class KnowledgeBase(Base):
    """Knowledge base model"""
    __tablename__ = "knowledge_bases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    
    @classmethod
    async def get_by_id_and_tenant(cls, kb_id, tenant_id, db: AsyncSession):
        """Get knowledge base by ID and tenant ID"""
        query = select(cls).filter(cls.id == kb_id, cls.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalars().first()
    
    @classmethod
    async def get_by_tenant(cls, tenant_id, db: AsyncSession):
        """Get all knowledge bases for a tenant"""
        query = select(cls).filter(cls.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalars().all()

class Document(Base):
    """Document model"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(Text, nullable=False)
    type = Column(Enum("file", "url", name="document_type"), nullable=False)
    source_url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    content_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    status = Column(
        Enum("pending", "processing", "completed", "error", "processed", name="document_status"), 
        default="pending"
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
     # Add this field to your Document model
    job_id = Column(String, nullable=True, index=True)
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    @classmethod
    async def get_by_id_kb_and_tenant(cls, doc_id, kb_id, tenant_id, db: AsyncSession):
        """Get document by ID, KB ID, and tenant ID"""
        query = select(cls).filter(
            cls.id == doc_id,
            cls.kb_id == kb_id,
            cls.tenant_id == tenant_id
        )
        result = await db.execute(query)
        return result.scalars().first()
    
    @classmethod
    async def get_by_kb_and_tenant(cls, kb_id, tenant_id, db: AsyncSession):
        """Get all documents in a KB for a tenant"""
        query = select(cls).filter(
            cls.kb_id == kb_id,
            cls.tenant_id == tenant_id
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def get_by_id_and_tenant(cls, doc_id, tenant_id, db: AsyncSession):
        """Get document by ID and tenant ID"""
        query = select(cls).filter(
            cls.id == doc_id,
            cls.tenant_id == tenant_id
        )
        result = await db.execute(query)
        return result.scalars().first()
    
    @classmethod
    async def get_by_type_and_tenant(cls, doc_type, tenant_id, db: AsyncSession):
        """Get documents by type and tenant ID"""
        query = select(cls).filter(
            cls.type == doc_type,
            cls.tenant_id == tenant_id
        )
        result = await db.execute(query)
        return result.scalars().all()
   
class Chunk(Base):
    """Chunk model for document segments"""
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    kb_id = Column(UUID(as_uuid=True), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    text = Column(Text, nullable=False)
    position = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=False)
    meta_data = Column(Text, nullable=True)  # JSON metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    @classmethod
    async def get_by_document(cls, document_id, db: AsyncSession):
        """Get all chunks for a document"""
        query = select(cls).filter(cls.document_id == document_id)
        result = await db.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def get_by_document_and_tenant(cls, document_id, tenant_id, db: AsyncSession):
        """Get chunks for a document that belongs to a tenant"""
        query = select(cls).filter(
            cls.document_id == document_id,
            cls.tenant_id == tenant_id
        )
        result = await db.execute(query)
        return result.scalars().all()