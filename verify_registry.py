"""
Verify ModelRegistry Integration with Runtime API.

This script tests:
1. ModelRegistry endpoints
2. Model listing and filtering
3. Model performance tracking
4. Registry statistics

Run the Runtime API first:
    falcon-ai runtime --config configs/inference.yaml --port 8000

Then run this script:
    python verify_registry.py
"""

import requests
import json
import time
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_json(data: Dict[str, Any]):
    """Print formatted JSON."""
    print(json.dumps(data, indent=2))


def test_health():
    """Test health endpoint."""
    print_header("1. HEALTH CHECK")

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ API Status: {data['status']}")
        print(f"✓ System initialized")
    else:
        print(f"✗ Health check failed")
        return False

    return True


def test_list_models():
    """Test model listing."""
    print_header("2. LIST ALL MODELS")

    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Total models: {data['total']}")

        for model in data['models']:
            print(f"\n  Model: {model['name']}")
            print(f"    Provider: {model['provider']}")
            print(f"    Latency: {model['avg_latency_ms']}ms")
            print(f"    Cost: ${model['cost_per_1k_tokens']}/1K tokens")
            print(f"    Capabilities: {', '.join(model['capabilities'])}")
            print(f"    Enabled: {model['enabled']}")
    else:
        print(f"✗ Failed to list models")
        return False

    return True


def test_filter_models():
    """Test model filtering."""
    print_header("3. FILTER MODELS")

    # Filter by capability
    print("\nFilter by capability: classification")
    response = requests.get(f"{BASE_URL}/models?capability=classification")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total']} classification models")
        for model in data['models']:
            print(f"  • {model['name']}")
    else:
        print(f"✗ Failed to filter by capability")

    # Filter by provider
    print("\nFilter by provider: local")
    response = requests.get(f"{BASE_URL}/models?provider=local")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total']} local models")
        for model in data['models']:
            print(f"  • {model['name']}")
    else:
        print(f"✗ Failed to filter by provider")


def test_get_model_details():
    """Test getting model details."""
    print_header("4. GET MODEL DETAILS")

    model_name = "falcon-heuristic"
    print(f"\nFetching details for: {model_name}")

    response = requests.get(f"{BASE_URL}/models/{model_name}")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Model: {data['name']}")
        print(f"  Provider: {data['provider']}")
        print(f"  Model ID: {data['model_id']}")
        print(f"  Avg Latency: {data['avg_latency_ms']}ms")
        print(f"  Cost: ${data['cost_per_1k_tokens']}/1K tokens")
        print(f"  Max Tokens: {data['max_tokens']}")
        print(f"  Capabilities: {', '.join(data['capabilities'])}")
        print(f"  Avg Confidence: {data['avg_confidence']}")
        print(f"  Success Rate: {data['success_rate']}")
        print(f"  Enabled: {data['enabled']}")

        if data.get('metadata'):
            print(f"  Metadata: {data['metadata']}")
    else:
        print(f"✗ Failed to get model details")
        return False

    return True


def test_run_inferences():
    """Test running inferences to generate performance data."""
    print_header("5. RUN INFERENCES")

    test_cases = [
        {"input": {"value": 0.95}, "confidence_target": 0.8, "risk_level": "high"},
        {"input": {"value": 0.3}, "confidence_target": 0.8, "risk_level": "low"},
        {"input": {"value": 0.75}, "confidence_target": 0.85, "risk_level": "medium"},
    ]

    print("\nRunning 3 test inferences...")

    for i, test_data in enumerate(test_cases, 1):
        response = requests.post(f"{BASE_URL}/infer", json=test_data)

        if response.status_code == 200:
            data = response.json()
            cached = data['metadata'].get('cached', False)
            cache_hit = data['metadata'].get('cache_hit', False)

            print(f"\n  Test {i}: Input value={test_data['input']['value']}")
            print(f"    ✓ Trace ID: {data['trace_id']}")
            print(f"    ✓ Model: {data['model']}")
            print(f"    ✓ Output: {data['output']}")
            print(f"    ✓ Confidence: {data['confidence']:.2f}")
            print(f"    ✓ Latency: {data['latency_ms']:.2f}ms")
            print(f"    ✓ Cost: ${data['est_cost_usd']:.6f}")
            print(f"    ✓ Cached: {cache_hit}")
        else:
            print(f"  Test {i}: ✗ Failed (status {response.status_code})")

        time.sleep(0.1)  # Small delay between requests

    return True


def test_model_performance():
    """Test model performance tracking."""
    print_header("6. MODEL PERFORMANCE")

    model_name = "falcon-heuristic"
    print(f"\nFetching performance for: {model_name}")

    response = requests.get(f"{BASE_URL}/models/{model_name}/performance")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Model: {data['model']}")
        print(f"  Total Requests: {data['total_requests']}")
        print(f"  Success Rate: {data['success_rate']:.1%}")
        print(f"  Avg Latency: {data['avg_latency_ms']:.2f}ms")
        print(f"  Avg Cost: ${data['avg_cost_usd']:.6f}")
        print(f"  Avg Confidence: {data['avg_confidence']:.2f}")
        print(f"  Total Cost: ${data['total_cost_usd']:.6f}")
    elif response.status_code == 404:
        print(f"  No performance data yet (run some inferences first)")
    else:
        print(f"✗ Failed to get performance")
        return False

    return True


def test_registry_stats():
    """Test registry statistics."""
    print_header("7. REGISTRY STATISTICS")

    response = requests.get(f"{BASE_URL}/registry/stats")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Total Models: {data['total_models']}")
        print(f"✓ Enabled Models: {data['enabled_models']}")
        print(f"✓ Providers: {', '.join(data['providers'])}")
        print(f"✓ Capabilities: {', '.join(data['capabilities'])}")

        print("\nModels Summary:")
        for name, info in data['models'].items():
            print(f"  • {name}: {info['provider']} ({info['avg_latency_ms']}ms, ${info['cost_per_1k_tokens']}/1K)")
    else:
        print(f"✗ Failed to get registry stats")
        return False

    return True


def test_cache_stats():
    """Test cache statistics."""
    print_header("8. CACHE STATISTICS")

    response = requests.get(f"{BASE_URL}/cache/stats")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Cache Hits: {data['hits']}")
        print(f"✓ Cache Misses: {data['misses']}")
        print(f"✓ Hit Rate: {data['hit_rate']:.1%}")
        print(f"✓ Cache Size: {data['cache_size']}/{data['max_size']}")
        print(f"✓ TTL: {data['ttl_seconds']}s")
        print(f"✓ Total Requests: {data['total_requests']}")
    else:
        print(f"✗ Failed to get cache stats")
        return False

    return True


def test_cost_tracking():
    """Test cost tracking."""
    print_header("9. COST TRACKING")

    response = requests.get(f"{BASE_URL}/costs/budget")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print("\nDaily Budget:")
        print(f"  Spend: ${data['daily']['spend']:.4f}")
        print(f"  Limit: ${data['daily']['limit']:.2f}")
        print(f"  Remaining: ${data['daily']['remaining']:.4f}")
        print(f"  Utilization: {data['daily']['utilization']:.1%}")

        print("\nMonthly Budget:")
        print(f"  Spend: ${data['monthly']['spend']:.4f}")
        print(f"  Limit: ${data['monthly']['limit']:.2f}")
        print(f"  Remaining: ${data['monthly']['remaining']:.4f}")
        print(f"  Utilization: {data['monthly']['utilization']:.1%}")
    else:
        print(f"✗ Failed to get cost tracking")
        return False

    return True


def main():
    """Run all verification tests."""

    print("=" * 70)
    print("FALCON RUNTIME API - ModelRegistry Verification")
    print("=" * 70)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the Runtime API is running:")
    print("  falcon-ai runtime --config configs/inference.yaml --port 8000")

    try:
        # Run tests
        results = {
            "Health Check": test_health(),
            "List Models": test_list_models(),
            "Filter Models": test_filter_models(),
            "Model Details": test_get_model_details(),
            "Run Inferences": test_run_inferences(),
            "Model Performance": test_model_performance(),
            "Registry Stats": test_registry_stats(),
            "Cache Stats": test_cache_stats(),
            "Cost Tracking": test_cost_tracking(),
        }

        # Summary
        print_header("VERIFICATION SUMMARY")

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        print(f"\nPassed: {passed}/{total}")

        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {test_name}")

        if passed == total:
            print("\n🎉 All tests passed! ModelRegistry integration is working.")
        else:
            print(f"\n⚠ {total - passed} test(s) failed. Check the output above.")

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API")
        print("Make sure the Runtime API is running:")
        print("  falcon-ai runtime --config configs/inference.yaml --port 8000")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    main()
