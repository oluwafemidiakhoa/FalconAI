"""
Model Registry for FALCON Runtime API.

This module manages multiple AI models and provides intelligent routing
based on cost, latency, capabilities, and historical performance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import logging
from collections import defaultdict

logger = logging.getLogger("falcon.ml.registry")


class ModelCapability(Enum):
    """Capabilities that models can provide."""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    CHAT = "chat"
    REASONING = "reasoning"
    CODE = "code"
    VISION = "vision"


class ModelProvider(Enum):
    """Supported model providers."""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    name: str
    provider: ModelProvider
    model_id: str  # Provider-specific model identifier

    # Performance characteristics
    avg_latency_ms: float = 100.0
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 4096

    # Capabilities
    capabilities: List[ModelCapability] = field(default_factory=list)

    # Quality metrics
    avg_confidence: float = 0.85
    success_rate: float = 0.99

    # Usage limits
    rate_limit_rpm: Optional[int] = None  # Requests per minute
    enabled: bool = True

    # Custom inference function (for local models)
    inference_fn: Optional[Callable] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Runtime performance tracking for a model."""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    avg_confidence: float = 0.0

    def update(self, latency_ms: float, cost_usd: float,
               confidence: float, success: bool = True):
        """Update performance metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd

        # Running average of confidence
        n = self.successful_requests
        if n > 0:
            self.avg_confidence = (self.avg_confidence * (n - 1) + confidence) / n

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "model": self.model_name,
            "total_requests": self.total_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0,
            "avg_latency_ms": self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0,
            "avg_cost_usd": self.total_cost_usd / self.total_requests if self.total_requests > 0 else 0.0,
            "avg_confidence": self.avg_confidence,
            "total_cost_usd": self.total_cost_usd
        }


@dataclass
class RoutingContext:
    """Context for model routing decisions."""
    # Requirements
    latency_budget_ms: Optional[float] = None
    cost_budget_usd: Optional[float] = None
    confidence_target: Optional[float] = None
    required_capabilities: List[ModelCapability] = field(default_factory=list)

    # Preferences
    prefer_speed: bool = False
    prefer_cost: bool = False
    prefer_quality: bool = False

    # Input characteristics
    input_tokens: Optional[int] = None
    expected_output_tokens: Optional[int] = None

    def estimate_cost(self, model: ModelSpec) -> float:
        """Estimate cost for this model."""
        if self.input_tokens and self.expected_output_tokens:
            total_tokens = self.input_tokens + self.expected_output_tokens
            return (total_tokens / 1000.0) * model.cost_per_1k_tokens
        return model.cost_per_1k_tokens  # Rough estimate


class ModelRegistry:
    """
    Registry for managing multiple AI models with intelligent routing.

    Features:
    - Register models with specs (cost, latency, capabilities)
    - Select best model based on requirements
    - Track model performance over time
    - Support multiple providers (local, OpenAI, Anthropic, etc.)
    - Route requests based on cost/latency/quality tradeoffs

    Example:
        registry = ModelRegistry()

        # Register models
        registry.register(ModelSpec(
            name="fast-heuristic",
            provider=ModelProvider.LOCAL,
            model_id="heuristic",
            avg_latency_ms=10.0,
            cost_per_1k_tokens=0.0,
            capabilities=[ModelCapability.CLASSIFICATION],
            inference_fn=my_heuristic_function
        ))

        registry.register(ModelSpec(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            model_id="gpt-3.5-turbo",
            avg_latency_ms=800.0,
            cost_per_1k_tokens=0.002,
            capabilities=[ModelCapability.CHAT, ModelCapability.GENERATION]
        ))

        # Select best model for requirements
        context = RoutingContext(
            latency_budget_ms=500.0,
            required_capabilities=[ModelCapability.CLASSIFICATION],
            prefer_cost=True
        )

        model = registry.select_model(context)
        result = registry.invoke(model.name, input_data)
    """

    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, ModelSpec] = {}
        self.performance: Dict[str, ModelPerformance] = defaultdict(
            lambda: ModelPerformance(model_name="unknown")
        )
        self._last_request_time: Dict[str, float] = {}  # For rate limiting

    def register(self, spec: ModelSpec):
        """
        Register a model.

        Args:
            spec: Model specification
        """
        self.models[spec.name] = spec
        self.performance[spec.name] = ModelPerformance(model_name=spec.name)
        logger.info(f"Registered model: {spec.name} ({spec.provider.value})")

    def unregister(self, name: str):
        """
        Unregister a model.

        Args:
            name: Model name
        """
        if name in self.models:
            del self.models[name]
            logger.info(f"Unregistered model: {name}")

    def get_model(self, name: str) -> Optional[ModelSpec]:
        """
        Get model by name.

        Args:
            name: Model name

        Returns:
            ModelSpec if found, None otherwise
        """
        return self.models.get(name)

    def list_models(self,
                    capability: Optional[ModelCapability] = None,
                    provider: Optional[ModelProvider] = None,
                    enabled_only: bool = True) -> List[ModelSpec]:
        """
        List registered models.

        Args:
            capability: Filter by capability
            provider: Filter by provider
            enabled_only: Only include enabled models

        Returns:
            List of matching model specs
        """
        models = list(self.models.values())

        if enabled_only:
            models = [m for m in models if m.enabled]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        if provider:
            models = [m for m in models if m.provider == provider]

        return models

    def select_model(self, context: RoutingContext) -> Optional[ModelSpec]:
        """
        Select best model for given context.

        This implements intelligent routing based on:
        - Required capabilities
        - Latency budget
        - Cost budget
        - Quality requirements
        - User preferences (speed/cost/quality)

        Args:
            context: Routing context with requirements

        Returns:
            Selected model spec, or None if no suitable model
        """
        # Filter by requirements
        candidates = self.list_models(enabled_only=True)

        # Filter by capabilities
        if context.required_capabilities:
            candidates = [
                m for m in candidates
                if all(cap in m.capabilities for cap in context.required_capabilities)
            ]

        if not candidates:
            logger.warning("No models match required capabilities")
            return None

        # Filter by latency budget
        if context.latency_budget_ms:
            candidates = [
                m for m in candidates
                if m.avg_latency_ms <= context.latency_budget_ms
            ]

        if not candidates:
            logger.warning("No models meet latency budget")
            return None

        # Filter by cost budget
        if context.cost_budget_usd:
            candidates = [
                m for m in candidates
                if context.estimate_cost(m) <= context.cost_budget_usd
            ]

        if not candidates:
            logger.warning("No models meet cost budget")
            return None

        # Filter by confidence target
        if context.confidence_target:
            candidates = [
                m for m in candidates
                if m.avg_confidence >= context.confidence_target
            ]

        if not candidates:
            logger.warning("No models meet confidence target")
            return None

        # Score remaining candidates
        scored = []
        for model in candidates:
            score = self._score_model(model, context)
            scored.append((model, score))

        # Sort by score (higher is better)
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = scored[0][0]
        logger.info(f"Selected model: {selected.name} (score: {scored[0][1]:.3f})")
        return selected

    def _score_model(self, model: ModelSpec, context: RoutingContext) -> float:
        """
        Score a model for given context.

        Higher score = better match for requirements.
        """
        score = 0.0

        # Performance data
        perf = self.performance.get(model.name)
        if perf and perf.total_requests > 0:
            stats = perf.get_stats()
            actual_latency = stats["avg_latency_ms"]
            actual_cost = stats["avg_cost_usd"]
            actual_confidence = stats["avg_confidence"]
            success_rate = stats["success_rate"]
        else:
            # Use spec values
            actual_latency = model.avg_latency_ms
            actual_cost = context.estimate_cost(model)
            actual_confidence = model.avg_confidence
            success_rate = model.success_rate

        # Normalize metrics (lower is better for cost/latency)
        # Assume max latency = 10000ms, max cost = $1.0
        latency_score = 1.0 - min(actual_latency / 10000.0, 1.0)
        cost_score = 1.0 - min(actual_cost / 1.0, 1.0)
        quality_score = actual_confidence
        reliability_score = success_rate

        # Weight based on preferences
        if context.prefer_speed:
            score += latency_score * 2.0
            score += cost_score * 0.5
            score += quality_score * 1.0
        elif context.prefer_cost:
            score += cost_score * 2.0
            score += latency_score * 0.5
            score += quality_score * 1.0
        elif context.prefer_quality:
            score += quality_score * 2.0
            score += reliability_score * 1.0
            score += cost_score * 0.5
            score += latency_score * 0.5
        else:
            # Balanced scoring
            score += latency_score * 1.0
            score += cost_score * 1.0
            score += quality_score * 1.5
            score += reliability_score * 0.5

        return score

    def invoke(self, model_name: str, input_data: Any,
               context: Optional[Dict] = None) -> Any:
        """
        Invoke a model by name.

        Args:
            model_name: Name of model to invoke
            input_data: Input for the model
            context: Additional context

        Returns:
            Model output

        Raises:
            ValueError: If model not found or not enabled
            RuntimeError: If inference fails
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")

        if not model.enabled:
            raise ValueError(f"Model not enabled: {model_name}")

        # Check rate limiting
        self._check_rate_limit(model)

        start_time = time.perf_counter()

        try:
            # Route to appropriate provider
            if model.provider == ModelProvider.LOCAL:
                if model.inference_fn is None:
                    raise RuntimeError(f"Local model {model_name} has no inference function")
                result = model.inference_fn(input_data, context or {})
            else:
                # External provider - will be implemented in provider integrations
                raise NotImplementedError(f"Provider {model.provider.value} not yet implemented")

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract metrics from result
            confidence = result.get("confidence", 0.85) if isinstance(result, dict) else 0.85
            cost_usd = result.get("cost_usd", 0.0) if isinstance(result, dict) else 0.0

            # Update performance
            self.performance[model_name].update(
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                confidence=confidence,
                success=True
            )

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Record failure
            self.performance[model_name].update(
                latency_ms=latency_ms,
                cost_usd=0.0,
                confidence=0.0,
                success=False
            )

            logger.exception(f"Model {model_name} inference failed")
            raise RuntimeError(f"Inference failed: {e}")

    def _check_rate_limit(self, model: ModelSpec):
        """Check and enforce rate limiting."""
        if model.rate_limit_rpm is None:
            return  # No rate limit

        now = time.time()
        last_request = self._last_request_time.get(model.name, 0)
        min_interval = 60.0 / model.rate_limit_rpm

        elapsed = now - last_request
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            logger.warning(f"Rate limit: waiting {wait_time:.2f}s for {model.name}")
            time.sleep(wait_time)

        self._last_request_time[model.name] = time.time()

    def get_performance(self, model_name: Optional[str] = None) -> Dict:
        """
        Get performance statistics.

        Args:
            model_name: Specific model name, or None for all models

        Returns:
            Performance statistics
        """
        if model_name:
            perf = self.performance.get(model_name)
            if perf:
                return perf.get_stats()
            return {}
        else:
            return {
                name: perf.get_stats()
                for name, perf in self.performance.items()
                if perf.total_requests > 0
            }

    def reset_performance(self, model_name: Optional[str] = None):
        """
        Reset performance statistics.

        Args:
            model_name: Specific model to reset, or None for all
        """
        if model_name:
            self.performance[model_name] = ModelPerformance(model_name=model_name)
        else:
            for name in self.performance:
                self.performance[name] = ModelPerformance(model_name=name)

    def get_registry_stats(self) -> Dict:
        """Get overall registry statistics."""
        return {
            "total_models": len(self.models),
            "enabled_models": len([m for m in self.models.values() if m.enabled]),
            "providers": list(set(m.provider.value for m in self.models.values())),
            "capabilities": list(set(
                cap.value
                for m in self.models.values()
                for cap in m.capabilities
            )),
            "models": {
                name: {
                    "provider": spec.provider.value,
                    "enabled": spec.enabled,
                    "capabilities": [c.value for c in spec.capabilities],
                    "avg_latency_ms": spec.avg_latency_ms,
                    "cost_per_1k_tokens": spec.cost_per_1k_tokens
                }
                for name, spec in self.models.items()
            }
        }
