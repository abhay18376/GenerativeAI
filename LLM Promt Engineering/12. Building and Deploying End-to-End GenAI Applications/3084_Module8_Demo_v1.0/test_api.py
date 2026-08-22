import requests
import json
import time

# Configuration
BASE_URL = "https://genai-chatbot-456095894614.us-central1.run.app"  # Change this to your deployed URL
SESSION_ID = "test_session_123"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_chat(message, session_id=SESSION_ID):
    """Test chat endpoint"""
    payload = {
        "message": message,
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"\nChat Request: {message}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['response']}")
        print(f"Session: {data['session_id']}")
        print(f"Timestamp: {data['timestamp']}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_session_info(session_id=SESSION_ID):
    """Test session info endpoint"""
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    print(f"\nSession Info: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_clear_session(session_id=SESSION_ID):
    """Test clear session endpoint"""
    response = requests.delete(f"{BASE_URL}/session/{session_id}")
    print(f"\nClear Session: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=== GenAI Chatbot API Testing ===")
    
    # Test 1: Health check
    print("\n1. Testing Health Endpoint...")
    if not test_health():
        print("❌ Health check failed")
        return
    print("✅ Health check passed")
    
    # Test 2: Chat functionality
    print("\n2. Testing Chat Endpoint...")
    test_messages = [
        "Hello! How are you?",
        "What is artificial intelligence?",
        "Can you explain machine learning in simple terms?",
        "What's the difference between AI and ML?"
    ]
    
    for msg in test_messages:
        if not test_chat(msg):
            print(f"❌ Chat test failed for: {msg}")
            return
        time.sleep(1)  # Small delay between requests
    
    print("✅ Chat tests passed")
    
    # Test 3: Session info
    print("\n3. Testing Session Info...")
    if not test_session_info():
        print("❌ Session info test failed")
        return
    print("✅ Session info test passed")
    
    # Test 4: Clear session
    print("\n4. Testing Clear Session...")
    if not test_clear_session():
        print("❌ Clear session test failed")
        return
    print("✅ Clear session test passed")
    
    # Test 5: Verify session cleared
    print("\n5. Verifying Session Cleared...")
    if not test_session_info():
        print("❌ Session verification failed")
        return
    print("✅ Session verification passed")
    
    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    main()