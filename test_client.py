"""
Simple test client to verify the server is working
Run this AFTER starting the server
"""
import requests
import json

SERVER_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate(prompt):
    """Test generate endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing /generate endpoint...")
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    try:
        data = {
            "prompt": prompt,
            "max_new_tokens": 100
        }
        
        response = requests.post(
            f"{SERVER_URL}/generate",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Response:")
            print(f"   {result['response']}")
        else:
            print(f"❌ Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🧪 Starting Server Tests")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        print("   Start server with: python simple_test_server.py")
        exit(1)
    
    # Test 2: Sample questions
    test_questions = [
        "What are the temple visiting hours?",
        "How do I book darshan tickets?",
        "Is free food available at Tirumala?",
        "How long does it take to climb Alipiri steps?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing with sample questions...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}/{len(test_questions)}]")
        test_generate(question)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)