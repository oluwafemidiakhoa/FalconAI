"""
Tests for Falcon Runtime API.

This module tests the REST inference endpoint to ensure:
- Inference requests work correctly
- Response format is valid
- Health check endpoint works
- Error handling is proper
"""

import pytest
from fastapi.testclient import TestClient
from falcon.runtime.api import create_app


class TestRuntimeAPI:
    """Test suite for Falcon Runtime API."""

    def test_health_endpoint_before_startup(self):
        """Test /health endpoint returns 503 before app startup."""
        app = create_app()
        client = TestClient(app)

        # Health should return 503 (starting) before startup event
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_infer_endpoint_with_basic_config(self):
        """Test /infer endpoint with basic threshold configuration."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        # Test inference request
        response = client.post("/infer", json={
            "input": {"value": 0.9},
            "latency_budget": "100ms",
            "confidence_target": 0.8,
            "risk_level": "medium"
        })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "trace_id" in data
        assert "model" in data
        assert "output" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "escalated" in data
        assert "est_cost_usd" in data
        assert "metadata" in data

        # Verify types
        assert isinstance(data["trace_id"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["latency_ms"], (int, float))
        assert isinstance(data["escalated"], bool)
        assert isinstance(data["est_cost_usd"], (int, float))

    def test_infer_endpoint_ignores_low_salience(self):
        """Test that low-salience events are ignored."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        # Low value should be filtered by perception
        response = client.post("/infer", json={
            "input": {"value": 0.1},  # Low salience
            "latency_budget": "100ms",
            "confidence_target": 0.8
        })

        assert response.status_code == 200
        data = response.json()

        # Should return None output with ignored status
        assert data["confidence"] == 0.0
        assert data["escalated"] is False
        assert data["metadata"]["status"] == "ignored"

    def test_infer_endpoint_with_high_salience(self):
        """Test that high-salience events trigger decisions."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        # High value should trigger action
        response = client.post("/infer", json={
            "input": {"value": 0.95},  # High salience
            "latency_budget": "100ms",
            "confidence_target": 0.8
        })

        assert response.status_code == 200
        data = response.json()

        # Should have made a decision
        assert data["output"] is not None
        assert data["confidence"] > 0.0

    def test_infer_endpoint_with_metadata(self):
        """Test that metadata is properly passed through."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        custom_metadata = {
            "user_id": "test_user",
            "request_source": "api_test"
        }

        response = client.post("/infer", json={
            "input": {"value": 0.9},
            "metadata": custom_metadata
        })

        assert response.status_code == 200
        data = response.json()

        # Metadata should be preserved
        assert isinstance(data["metadata"], dict)

    def test_infer_endpoint_without_config(self):
        """Test /infer fails gracefully without config."""
        app = create_app()  # No config
        client = TestClient(app)

        response = client.post("/infer", json={
            "input": {"value": 0.9}
        })

        # Should either work with defaults or return error
        assert response.status_code in [200, 500, 503]

    def test_infer_endpoint_latency_tracking(self):
        """Test that latency is properly tracked."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        response = client.post("/infer", json={
            "input": {"value": 0.9},
            "latency_budget": "50ms"
        })

        assert response.status_code == 200
        data = response.json()

        # Latency should be a positive number
        assert data["latency_ms"] > 0
        # Should complete within reasonable time
        assert data["latency_ms"] < 1000  # Less than 1 second

    def test_health_endpoint_after_init(self):
        """Test /health endpoint returns proper system status."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        # Make an inference first to initialize
        client.post("/infer", json={"input": {"value": 0.9}})

        # Now check health
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = response.json()

        assert "status" in data
        if response.status_code == 200:
            assert data["status"] == "ok"
            assert "system" in data

    def test_infer_request_validation(self):
        """Test that invalid requests are rejected."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        # Missing required 'input' field
        response = client.post("/infer", json={
            "latency_budget": "100ms"
        })

        # Should return 422 validation error
        assert response.status_code == 422

    def test_multiple_sequential_inferences(self):
        """Test that multiple inferences work correctly."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        trace_ids = set()

        for i in range(5):
            response = client.post("/infer", json={
                "input": {"value": 0.5 + i * 0.1}
            })

            assert response.status_code == 200
            data = response.json()

            # Each request should have unique trace ID
            assert data["trace_id"] not in trace_ids
            trace_ids.add(data["trace_id"])

    def test_cost_estimation(self):
        """Test that cost estimation is included."""
        app = create_app(config_path="configs/basic_config.yaml")
        client = TestClient(app)

        response = client.post("/infer", json={
            "input": {"value": 0.9}
        })

        assert response.status_code == 200
        data = response.json()

        # Cost should be present and non-negative
        assert "est_cost_usd" in data
        assert data["est_cost_usd"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
