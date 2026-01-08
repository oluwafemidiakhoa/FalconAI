
import requests
import json
import time

url = "http://localhost:8081/infer"
data = {
    "input": "test_input_for_cache",
    "latency_budget": "500ms"
}

print("1. Sending FIRST request (Cache MISS expected)...")
t0 = time.time()
resp1 = requests.post(url, json=data)
print(f"Time: {time.time()-t0:.3f}s, Trace: {resp1.json().get('trace_id')}, Cached: {resp1.json().get('metadata', {}).get('cached')}")

print("\n2. Sending SECOND request (Cache HIT expected)...")
t0 = time.time()
resp2 = requests.post(url, json=data)
print(f"Time: {time.time()-t0:.3f}s, Trace: {resp2.json().get('trace_id')}, Cached: {resp2.json().get('metadata', {}).get('cached')}")
