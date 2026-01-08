
import requests
import json
import time

url = "http://localhost:8081/infer"
data = {
    "input": [0.1, 0.5, 0.9],
    "latency_budget": "500ms",
    "confidence_target": 0.8
}

print(f"Sending request to {url}...")
try:
    # Wait for server to start
    time.sleep(3) 
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
