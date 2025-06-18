#!/usr/bin/env python3
"""
JWT functionality test script
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_jwt_import():
    """Test JWT library import"""
    print("🔍 Testing JWT library import...")
    
    try:
        import jwt
        print(f"✅ JWT library imported successfully")
        print(f"📦 JWT version: {jwt.__version__}")
        print(f"🔧 Available methods: {[method for method in dir(jwt) if not method.startswith('_')]}")
        
        # Test basic functionality
        if hasattr(jwt, 'encode') and hasattr(jwt, 'decode'):
            print("✅ JWT encode/decode methods available")
            return True
        else:
            print("❌ JWT encode/decode methods NOT available")
            print("This indicates python-jwt is installed instead of PyJWT")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import JWT: {e}")
        return False

def test_token_creation():
    """Test JWT token creation"""
    print("\n🔍 Testing JWT token creation...")
    
    try:
        from app.auth.jwt import create_access_token
        from uuid import uuid4
        
        # Create a test token
        test_tenant_id = uuid4()
        test_email = "test@example.com"
        
        token = create_access_token(test_tenant_id, test_email)
        print(f"✅ Token created successfully")
        print(f"📝 Token (first 50 chars): {token[:50]}...")
        
        return token
        
    except Exception as e:
        print(f"❌ Token creation failed: {e}")
        import traceback
        print(f"🔍 Full error: {traceback.format_exc()}")
        return None

def test_token_decoding(token):
    """Test JWT token decoding"""
    print("\n🔍 Testing JWT token decoding...")
    
    try:
        from app.auth.jwt import decode_token
        
        payload = decode_token(token)
        print(f"✅ Token decoded successfully")
        print(f"📝 Payload: {payload}")
        
        required_fields = ['sub', 'tenant_id', 'email', 'exp', 'iat', 'jti', 'type']
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
        
        return True
        
    except Exception as e:
        print(f"❌ Token decoding failed: {e}")
        return False

def test_auth_endpoints():
    """Test authentication endpoints"""
    print("\n🔍 Testing authentication endpoints...")
    
    try:
        from app.api.endpoints.auth import UserCreate
        
        # Test user creation schema
        test_user = UserCreate(
            email="test@example.com",
            password="testpassword123",
            name="Test User"
        )
        print("✅ UserCreate schema works")
        
        return True
        
    except Exception as e:
        print(f"❌ Auth endpoints test failed: {e}")
        return False

def main():
    """Run all JWT tests"""
    print("🔐 JWT Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        ("JWT Import", test_jwt_import),
        ("Token Creation", test_token_creation),
        ("Auth Endpoints", test_auth_endpoints),
    ]
    
    results = []
    token = None
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if name == "Token Creation":
            result = test_func()
            if isinstance(result, str):  # Token returned
                token = result
                results.append((name, True))
            else:
                results.append((name, False))
        else:
            result = test_func()
            results.append((name, result))
    
    # Test token decoding if we have a token
    if token:
        print(f"\n--- Token Decoding ---")
        decode_result = test_token_decoding(token)
        results.append(("Token Decoding", decode_result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All JWT tests passed!")
        print("Your authentication should work now.")
    else:
        print("⚠️  Some tests failed. Follow the fix instructions below.")
        print("\nFix Instructions:")
        print("1. pip uninstall python-jwt jwt PyJWT -y")
        print("2. pip install PyJWT==2.10.1")
        print("3. Restart your application")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)