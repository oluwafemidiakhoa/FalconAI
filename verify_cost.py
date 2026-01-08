
import requests
import json
import time

url = "http://localhost:8081/costs/budget"

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print("Budget Status:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
