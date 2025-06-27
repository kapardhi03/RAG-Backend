#!/usr/bin/env python3
"""
Test database connection script
Run this to verify your PostgreSQL setup is working correctly
"""

import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

async def test_direct_connection():
    """Test direct asyncpg connection"""
    try:
        conn = await asyncpg.connect(
            user="postgres",
            password="postgres", 
            database="rag_microservices",
            host="localhost",
            port=5432
        )
        
        # Test query
        result = await conn.fetchval("SELECT version()")
        print(f"✅ Direct connection successful!")
        print(f"PostgreSQL version: {result}")
        
        await conn.close()
        return True
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False

async def test_sqlalchemy_connection():
    """Test SQLAlchemy async connection"""
    try:
        engine = create_async_engine(settings.DATABASE_URL)
        
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            print("✅ SQLAlchemy connection successful!")
        
        await engine.dispose()
        return True
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        return False

async def test_database_creation():
    """Test database and table creation"""
    try:
        from app.db.session import init_db
        await init_db()
        print("✅ Database initialization successful!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

async def main():
    print("🔍 Testing PostgreSQL setup...")
    print("=" * 50)
    
    # Test 1: Direct connection
    print("Test 1: Direct asyncpg connection")
    success1 = await test_direct_connection()
    print()
    
    # Test 2: SQLAlchemy connection  
    print("Test 2: SQLAlchemy connection")
    success2 = await test_sqlalchemy_connection()
    print()
    
    # Test 3: Database initialization
    print("Test 3: Database initialization")
    success3 = await test_database_creation()
    print()
    
    print("=" * 50)
    if success1 and success2 and success3:
        print("All tests passed! Your PostgreSQL setup is working correctly.")
    else:
        print("Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())