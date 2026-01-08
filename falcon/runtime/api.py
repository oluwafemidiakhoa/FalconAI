"""
Falcon Runtime API.

This module provides the REST interface for the FalconAI runtime,
allowing external applications to request inference and decisions.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import load_config, normalize_config, build_falcon, FalconSpec
from ..core import FalconAI
from ..ml.inference_cache import InferenceCache
from ..ml.cost_manager import CostManager
from ..ml.model_registry import (
    ModelRegistry, ModelSpec, ModelProvider, ModelCapability,
    RoutingContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("falcon.runtime")

class InferRequest(BaseModel):
    """Request payload for inference."""
    input: Any = Field(..., description="Input data for the model (text, json, etc.)")
    latency_budget: Optional[str] = Field("350ms", description="Latency budget string (e.g. '350ms', '1s')")
    confidence_target: Optional[float] = Field(0.82, description="Target confidence score (0.0-1.0)")
    risk_level: Optional[str] = Field("medium", description="Risk tolerance: low, medium, high")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context metadata")

class InferResponse(BaseModel):
    """Response payload for inference."""
    trace_id: str
    model: str
    output: Any
    confidence: float
    latency_ms: float
    escalated: bool
    est_cost_usd: float
    metadata: Dict[str, Any]

def create_app(config_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Falcon Runtime API", version="0.2.0")

    # Global state for the Falcon instance
    falcon_instance: Optional[FalconAI] = None

    # Initialize cache, cost manager, and model registry
    cache = InferenceCache(ttl_seconds=3600, max_size=1000)
    cost_manager = CostManager(daily_limit=100.0, monthly_limit=2000.0)
    registry = ModelRegistry()

    @app.on_event("startup")
    async def startup_event():
        nonlocal falcon_instance
        logger.info(f"Loading Falcon configuration from {config_path or 'default'}")

        # Load and build Falcon
        try:
            raw_config = load_config(config_path) if config_path else {}
            app_cfg = normalize_config(raw_config)
            falcon_instance = build_falcon(app_cfg.falcon)
            logger.info("Falcon Runtime initialized successfully")
            logger.info(f"Cache enabled: TTL={cache.ttl}s, Max={cache.max_size}")
            logger.info(f"Cost limits: Daily=${cost_manager.daily_limit}, Monthly=${cost_manager.monthly_limit}")

            # Register default local model (FALCON heuristic)
            def falcon_inference(input_data: Any, context: Dict) -> Dict:
                """Default FALCON inference function."""
                decision = falcon_instance.process(input_data, context)
                if decision is None:
                    return {
                        "output": None,
                        "confidence": 0.0,
                        "cost_usd": 0.0,
                        "metadata": {"status": "ignored"}
                    }
                return {
                    "output": decision.action.value if hasattr(decision.action, "value") else str(decision.action),
                    "confidence": decision.confidence,
                    "cost_usd": decision.metadata.get("cost_usd", 0.0001),
                    "metadata": decision.metadata or {}
                }

            registry.register(ModelSpec(
                name="falcon-heuristic",
                provider=ModelProvider.LOCAL,
                model_id="falcon-default",
                avg_latency_ms=10.0,
                cost_per_1k_tokens=0.0,
                capabilities=[ModelCapability.CLASSIFICATION, ModelCapability.REASONING],
                inference_fn=falcon_inference,
                metadata={"description": "Fast local FALCON heuristic engine"}
            ))
            logger.info("Registered default model: falcon-heuristic")

        except Exception as e:
            logger.error(f"Failed to initialize Falcon: {e}")
            raise RuntimeError(f"Could not load Falcon config: {e}")

    @app.post("/infer", response_model=InferResponse)
    async def infer(request: InferRequest):
        if falcon_instance is None:
            raise HTTPException(status_code=503, detail="Falcon Runtime not initialized")

        trace_id = uuid.uuid4().hex
        start_time = time.perf_counter()

        # Check cache first
        request_dict = request.dict()
        cached_response = cache.get(request_dict)
        if cached_response:
            # Cache hit - return immediately
            cached_response["metadata"]["cached"] = True
            cached_response["metadata"]["cache_hit"] = True
            cached_response["trace_id"] = trace_id  # New trace ID even for cached
            logger.info(f"Cache HIT for trace {trace_id}")
            return InferResponse(**cached_response)

        # Prepare context for Falcon
        context = request.metadata or {}
        context.update({
            "trace_id": trace_id,
            "latency_budget": request.latency_budget,
            "confidence_target": request.confidence_target,
            "risk_level": request.risk_level,
        })

        try:
            # Run Inference -> FalconAI.process(data, context)
            decision = falcon_instance.process(request.input, context)

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Default response if no decision/actionable event
            if decision is None:
                response = InferResponse(
                    trace_id=trace_id,
                    model="system",  # Default/Gatekeeper
                    output=None,
                    confidence=0.0,
                    latency_ms=latency_ms,
                    escalated=False,
                    est_cost_usd=0.0,
                    metadata={"status": "ignored", "reason": "Not salient", "cached": False}
                )
            else:
                # Map Decision to Response
                output = decision.action.value if hasattr(decision.action, "value") else str(decision.action)

                # Determine if escalation occurred
                escalated = decision.metadata.get("escalated", False)
                model_used = decision.metadata.get("model", "default_core")
                cost = decision.metadata.get("cost_usd", 0.0001)  # constant fallback

                response = InferResponse(
                    trace_id=trace_id,
                    model=model_used,
                    output=output,
                    confidence=decision.confidence,
                    latency_ms=latency_ms,
                    escalated=escalated,
                    est_cost_usd=cost,
                    metadata={**(decision.metadata or {}), "cached": False}
                )

            # Record cost
            cost_manager.record_cost(
                cost_usd=response.est_cost_usd,
                trace_id=trace_id,
                model=response.model,
                metadata={"latency_ms": latency_ms}
            )

            # Cache the response (without trace_id for deduplication)
            response_dict = response.dict()
            response_dict["trace_id"] = "cached"  # Placeholder for cache
            cache.set(request_dict, response_dict)

            logger.info(f"Cache MISS for trace {trace_id}, cost=${response.est_cost_usd}")
            return response

        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        if falcon_instance is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return {"status": "ok", "system": falcon_instance.get_status()}

    @app.get("/cache/stats")
    async def cache_stats():
        """Get cache performance statistics."""
        return cache.get_stats()

    @app.get("/cache/clear")
    async def cache_clear():
        """Clear all cached entries."""
        cache.clear()
        return {"status": "ok", "message": "Cache cleared"}

    @app.get("/costs/budget")
    async def costs_budget():
        """Get current budget status."""
        return cost_manager.get_budget_status()

    @app.get("/costs/report")
    async def costs_report(days: int = 7):
        """Get cost report for the last N days."""
        return cost_manager.get_report(days=days)

    @app.get("/costs/by-model")
    async def costs_by_model(days: int = 7):
        """Get cost breakdown by model."""
        return cost_manager.get_cost_by_model(days=days)

    @app.get("/costs/alert")
    async def costs_alert(threshold: float = 0.8):
        """Check if spending is approaching limits."""
        return cost_manager.should_alert(threshold=threshold)

    # Model Registry Endpoints

    @app.get("/models")
    async def list_models(capability: Optional[str] = None,
                         provider: Optional[str] = None,
                         enabled_only: bool = True):
        """
        List registered models.

        Query parameters:
        - capability: Filter by capability (classification, generation, chat, etc.)
        - provider: Filter by provider (local, openai, anthropic, etc.)
        - enabled_only: Only show enabled models (default: true)
        """
        cap = ModelCapability(capability) if capability else None
        prov = ModelProvider(provider) if provider else None

        models = registry.list_models(capability=cap, provider=prov, enabled_only=enabled_only)

        return {
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider.value,
                    "model_id": m.model_id,
                    "avg_latency_ms": m.avg_latency_ms,
                    "cost_per_1k_tokens": m.cost_per_1k_tokens,
                    "capabilities": [c.value for c in m.capabilities],
                    "enabled": m.enabled,
                    "metadata": m.metadata
                }
                for m in models
            ],
            "total": len(models)
        }

    @app.get("/models/{model_name}")
    async def get_model(model_name: str):
        """Get details for a specific model."""
        model = registry.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        return {
            "name": model.name,
            "provider": model.provider.value,
            "model_id": model.model_id,
            "avg_latency_ms": model.avg_latency_ms,
            "cost_per_1k_tokens": model.cost_per_1k_tokens,
            "max_tokens": model.max_tokens,
            "capabilities": [c.value for c in model.capabilities],
            "avg_confidence": model.avg_confidence,
            "success_rate": model.success_rate,
            "rate_limit_rpm": model.rate_limit_rpm,
            "enabled": model.enabled,
            "metadata": model.metadata
        }

    @app.get("/models/{model_name}/performance")
    async def get_model_performance(model_name: str):
        """Get runtime performance statistics for a model."""
        perf = registry.get_performance(model_name)
        if not perf:
            raise HTTPException(status_code=404, detail=f"No performance data for: {model_name}")
        return perf

    @app.get("/registry/stats")
    async def registry_stats():
        """Get overall registry statistics."""
        return registry.get_registry_stats()

    @app.post("/models/{model_name}/enable")
    async def enable_model(model_name: str):
        """Enable a model."""
        model = registry.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        model.enabled = True
        return {"status": "ok", "message": f"Model {model_name} enabled"}

    @app.post("/models/{model_name}/disable")
    async def disable_model(model_name: str):
        """Disable a model."""
        model = registry.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        model.enabled = False
        return {"status": "ok", "message": f"Model {model_name} disabled"}

    return app
