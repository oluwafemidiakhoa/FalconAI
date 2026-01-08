"""
Multi-Model Demo with FALCON ModelRegistry.

This script demonstrates:
1. Registering multiple models (local FALCON, OpenAI GPT, Anthropic Claude)
2. Intelligent routing based on cost/latency/quality requirements
3. Automatic fallback strategies
4. Performance tracking across models

Requirements:
    pip install openai anthropic  # Optional, for external providers

Environment variables (optional):
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any

# FALCON imports
from falcon.ml.model_registry import (
    ModelRegistry, ModelSpec, ModelProvider, ModelCapability,
    RoutingContext
)
from falcon.api.providers import (
    OpenAIProvider, AnthropicProvider,
    create_openai_inference_fn, create_anthropic_inference_fn,
    ProviderConfig
)


def register_local_models(registry: ModelRegistry):
    """Register fast local models."""

    def simple_heuristic(input_data: Any, context: Dict) -> Dict:
        """Ultra-fast rule-based classification."""
        if isinstance(input_data, dict):
            value = input_data.get("value", 0.0)
        else:
            value = float(input_data)

        # Simple threshold rule
        if value > 0.8:
            output = "ALERT"
            confidence = 0.85
        elif value > 0.5:
            output = "MONITOR"
            confidence = 0.75
        else:
            output = "IGNORE"
            confidence = 0.90

        return {
            "output": output,
            "confidence": confidence,
            "cost_usd": 0.0,
            "metadata": {"model": "simple-heuristic", "rule": "threshold"}
        }

    registry.register(ModelSpec(
        name="simple-heuristic",
        provider=ModelProvider.LOCAL,
        model_id="heuristic-v1",
        avg_latency_ms=1.0,  # Ultra-fast
        cost_per_1k_tokens=0.0,  # Free
        capabilities=[ModelCapability.CLASSIFICATION],
        avg_confidence=0.80,
        success_rate=0.95,
        metadata={"description": "Ultra-fast rule-based classifier"}
    ))
    registry.models["simple-heuristic"].inference_fn = simple_heuristic

    print("[OK] Registered local model: simple-heuristic (1ms, $0)")


def register_openai_models(registry: ModelRegistry):
    """Register OpenAI models (if API key available)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[SKIP] Skipping OpenAI models (no API key)")
        return

    try:
        provider = OpenAIProvider()

        # GPT-4o-mini - Fast and cheap
        inference_fn = create_openai_inference_fn(
            provider=provider,
            model_id="gpt-4o-mini",
            temperature=0.3,  # More deterministic
            max_tokens=50
        )

        registry.register(ModelSpec(
            name="gpt-4o-mini-classifier",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o-mini",
            avg_latency_ms=600.0,
            cost_per_1k_tokens=0.0003,  # Very cheap
            capabilities=[ModelCapability.CHAT, ModelCapability.CLASSIFICATION, ModelCapability.REASONING],
            avg_confidence=0.90,
            success_rate=0.98,
            max_tokens=4096,
            metadata={"description": "Fast OpenAI model for classification"}
        ))
        registry.models["gpt-4o-mini-classifier"].inference_fn = inference_fn

        print("[OK] Registered OpenAI model: gpt-4o-mini-classifier (600ms, $0.0003/1K)")

        # GPT-4o - Powerful reasoning
        inference_fn = create_openai_inference_fn(
            provider=provider,
            model_id="gpt-4o",
            temperature=0.7,
            max_tokens=150
        )

        registry.register(ModelSpec(
            name="gpt-4o-analyzer",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            avg_latency_ms=1200.0,
            cost_per_1k_tokens=0.005,  # More expensive
            capabilities=[ModelCapability.CHAT, ModelCapability.REASONING, ModelCapability.GENERATION],
            avg_confidence=0.95,
            success_rate=0.99,
            max_tokens=8192,
            metadata={"description": "Powerful reasoning model"}
        ))
        registry.models["gpt-4o-analyzer"].inference_fn = inference_fn

        print("[OK] Registered OpenAI model: gpt-4o-analyzer (1200ms, $0.005/1K)")

    except Exception as e:
        print(f"[ERROR] Error registering OpenAI models: {e}")


def register_anthropic_models(registry: ModelRegistry):
    """Register Anthropic Claude models (if API key available)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[SKIP] Skipping Anthropic models (no API key)")
        return

    try:
        provider = AnthropicProvider()

        # Claude 3.5 Haiku - Fast and cheap
        inference_fn = create_anthropic_inference_fn(
            provider=provider,
            model_id="claude-3-5-haiku-20241022",
            max_tokens=200,
            temperature=0.5
        )

        registry.register(ModelSpec(
            name="claude-haiku-classifier",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-haiku-20241022",
            avg_latency_ms=800.0,
            cost_per_1k_tokens=0.0024,  # Cheap
            capabilities=[ModelCapability.CHAT, ModelCapability.CLASSIFICATION, ModelCapability.REASONING],
            avg_confidence=0.92,
            success_rate=0.98,
            max_tokens=8192,
            metadata={"description": "Fast Claude model for classification"}
        ))
        registry.models["claude-haiku-classifier"].inference_fn = inference_fn

        print("[OK] Registered Anthropic model: claude-haiku-classifier (800ms, $0.0024/1K)")

        # Claude 3.5 Sonnet - Deep reasoning
        inference_fn = create_anthropic_inference_fn(
            provider=provider,
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=1.0
        )

        registry.register(ModelSpec(
            name="claude-sonnet-analyst",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            avg_latency_ms=1500.0,
            cost_per_1k_tokens=0.009,  # Premium
            capabilities=[ModelCapability.CHAT, ModelCapability.REASONING, ModelCapability.GENERATION],
            avg_confidence=0.96,
            success_rate=0.99,
            max_tokens=200000,
            metadata={"description": "Powerful reasoning and analysis"}
        ))
        registry.models["claude-sonnet-analyst"].inference_fn = inference_fn

        print("[OK] Registered Anthropic model: claude-sonnet-analyst (1500ms, $0.009/1K)")

    except Exception as e:
        print(f"[WARN] Error registering Anthropic models: {e}")


def demo_routing_scenarios(registry: ModelRegistry):
    """Demonstrate different routing scenarios."""

    print("\n" + "=" * 70)
    print("INTELLIGENT ROUTING SCENARIOS")
    print("=" * 70)

    test_input = {"value": 0.95, "source": "sensor_42", "type": "anomaly"}

    # Scenario 1: Need speed (real-time monitoring)
    print("\n1. REAL-TIME MONITORING (prefer speed)")
    print("-" * 70)
    context = RoutingContext(
        latency_budget_ms=50.0,  # Very tight budget
        confidence_target=0.75,
        required_capabilities=[ModelCapability.CLASSIFICATION],
        prefer_speed=True
    )

    model = registry.select_model(context)
    if model:
        print(f"   Selected: {model.name}")
        print(f"   Latency: {model.avg_latency_ms}ms")
        print(f"   Cost: ${model.cost_per_1k_tokens:.6f}/1K tokens")
        print(f"   Provider: {model.provider.value}")

        # Run inference
        try:
            result = registry.invoke(model.name, test_input, {})
            print(f"   Result: {result['output']} (confidence: {result['confidence']:.2f})")
        except Exception as e:
            print(f"   Error: {e}")

    # Scenario 2: Need accuracy (critical decisions)
    print("\n2. CRITICAL DECISION (prefer quality)")
    print("-" * 70)
    context = RoutingContext(
        latency_budget_ms=2000.0,  # Can wait longer
        confidence_target=0.93,  # High confidence needed
        required_capabilities=[ModelCapability.REASONING],
        prefer_quality=True
    )

    model = registry.select_model(context)
    if model:
        print(f"   Selected: {model.name}")
        print(f"   Latency: {model.avg_latency_ms}ms")
        print(f"   Cost: ${model.cost_per_1k_tokens:.6f}/1K tokens")
        print(f"   Confidence: {model.avg_confidence:.2f}")

    # Scenario 3: Need cost efficiency (batch processing)
    print("\n3. BATCH PROCESSING (prefer cost)")
    print("-" * 70)
    context = RoutingContext(
        latency_budget_ms=1000.0,
        confidence_target=0.85,
        required_capabilities=[ModelCapability.CLASSIFICATION],
        prefer_cost=True
    )

    model = registry.select_model(context)
    if model:
        print(f"   Selected: {model.name}")
        print(f"   Cost: ${model.cost_per_1k_tokens:.6f}/1K tokens")
        print(f"   Latency: {model.avg_latency_ms}ms")

    # Scenario 4: Balanced (no specific preference)
    print("\n4. BALANCED DECISION (no preference)")
    print("-" * 70)
    context = RoutingContext(
        latency_budget_ms=1000.0,
        confidence_target=0.85,
        required_capabilities=[ModelCapability.CLASSIFICATION]
    )

    model = registry.select_model(context)
    if model:
        print(f"   Selected: {model.name}")
        print(f"   Score: Balanced across speed/cost/quality")


def demo_performance_tracking(registry: ModelRegistry):
    """Demonstrate performance tracking."""

    print("\n" + "=" * 70)
    print("PERFORMANCE TRACKING")
    print("=" * 70)

    # Run several inferences
    test_inputs = [
        {"value": 0.95},
        {"value": 0.3},
        {"value": 0.7},
        {"value": 0.85},
        {"value": 0.1}
    ]

    model_name = "simple-heuristic"
    print(f"\nRunning 5 inferences on {model_name}...")

    for i, input_data in enumerate(test_inputs, 1):
        try:
            result = registry.invoke(model_name, input_data, {})
            print(f"  {i}. Input: {input_data['value']:.2f} -> Output: {result['output']}")
        except Exception as e:
            print(f"  {i}. Error: {e}")

    # Show performance stats
    print(f"\nPerformance Statistics for {model_name}:")
    print("-" * 70)
    stats = registry.get_performance(model_name)
    if stats:
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  Avg cost: ${stats['avg_cost_usd']:.6f}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"  Total cost: ${stats['total_cost_usd']:.6f}")


def demo_registry_stats(registry: ModelRegistry):
    """Show overall registry statistics."""

    print("\n" + "=" * 70)
    print("REGISTRY STATISTICS")
    print("=" * 70)

    stats = registry.get_registry_stats()

    print(f"\nTotal models: {stats['total_models']}")
    print(f"Enabled models: {stats['enabled_models']}")
    print(f"Providers: {', '.join(stats['providers'])}")
    print(f"Capabilities: {', '.join(stats['capabilities'])}")

    print("\nRegistered Models:")
    print("-" * 70)
    for name, info in stats['models'].items():
        print(f"\n  {name}")
        print(f"    Provider: {info['provider']}")
        print(f"    Enabled: {info['enabled']}")
        print(f"    Latency: {info['avg_latency_ms']}ms")
        print(f"    Cost: ${info['cost_per_1k_tokens']:.6f}/1K")
        print(f"    Capabilities: {', '.join(info['capabilities'])}")


def main():
    """Run the multi-model demo."""

    print("=" * 70)
    print("FALCON-AI MULTI-MODEL DEMO")
    print("=" * 70)

    # Create registry
    registry = ModelRegistry()

    # Register models
    print("\nRegistering models...")
    print("-" * 70)
    register_local_models(registry)
    register_openai_models(registry)
    register_anthropic_models(registry)

    # Show registry stats
    demo_registry_stats(registry)

    # Demonstrate routing
    demo_routing_scenarios(registry)

    # Show performance tracking
    demo_performance_tracking(registry)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\n[TIP] Key Takeaways:")
    print("  * Registry automatically routes to the best model for your requirements")
    print("  * Choose speed, cost, or quality optimization")
    print("  * Performance tracking helps optimize over time")
    print("  * Seamless integration of local + cloud models")

    print("\n[INFO] Next Steps:")
    print("  1. Set API keys: export OPENAI_API_KEY=... or ANTHROPIC_API_KEY=...")
    print("  2. Customize routing logic in ModelRegistry.select_model()")
    print("  3. Add your own custom models to the registry")
    print("  4. Integrate with FALCON Runtime API for production use")


if __name__ == "__main__":
    main()
