"""
Quick test script for Salahsense API endpoints.
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api():
    """Test all API endpoints."""
    
    print("=" * 60)
    print("SALAHSENSE API TEST")
    print("=" * 60)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Message: {response.json()['message']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test 2: Categories endpoint
    print("\n2. Testing categories endpoint...")
    try:
        response = requests.get(f"{API_BASE}/categories")
        categories = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Namaz categories: {categories['namaz']}")
        print(f"✓ Non-namaz categories: {categories['non_namaz']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Status endpoint
    print("\n3. Testing status endpoint...")
    try:
        response = requests.get(f"{API_BASE}/status")
        status = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Model trained: {status['model_trained']}")
        print(f"✓ Total files: {status['data_statistics']['total_files']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Upload endpoint (with sample data)
    print("\n4. Testing upload endpoint...")
    
    # Create a small sample CSV
    sample_csv = """timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z
0.00,-0.177,0.216,-0.137,0.123,-0.456,9.807
0.01,-0.244,-0.468,-0.277,0.145,-0.432,9.812
0.02,-0.154,-0.228,-0.277,0.167,-0.398,9.803
0.03,-0.089,-0.189,-0.213,0.189,-0.376,9.798
0.04,-0.122,-0.267,-0.189,0.201,-0.354,9.791
0.05,-0.089,-0.228,-0.166,0.223,-0.332,9.784
"""
    
    try:
        files = {'file': ('test_sample.csv', sample_csv, 'text/csv')}
        data = {
            'category_type': 'namaz',
            'category': 'standing_loose_clothing'
        }
        
        response = requests.post(f"{API_BASE}/upload", files=files, data=data)
        result = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Upload success: {result['success']}")
        print(f"✓ File saved as: {result['file_info']['filename']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Training endpoint
    print("\n5. Testing training endpoint...")
    try:
        response = requests.post(f"{API_BASE}/train")
        result = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Training started: {result['success']}")
        print(f"✓ Job ID: {result['job_id']}")
        
        # Test training status
        time.sleep(2)  # Wait a bit
        status_response = requests.get(f"{API_BASE}/train_status/{result['job_id']}")
        print(f"✓ Training status: {status_response.json()['status']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("API TEST COMPLETE")
    print("=" * 60)
    print("\nAPI Documentation: API_DOCUMENTATION.md")
    print("Interactive Docs: http://localhost:8000/docs")
    print("\nExample usage from your mobile app:")
    print("POST http://localhost:8000/upload")
    print("  - file: your_recording.csv")
    print("  - category_type: namaz")
    print("  - category: standing_loose_clothing")

if __name__ == "__main__":
    test_api()