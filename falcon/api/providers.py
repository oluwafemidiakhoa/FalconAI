"""
Provider integrations for external AI services.

This module provides integrations with:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude models)
- Custom providers

Use these to register external models in the ModelRegistry.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("falcon.api.providers")


@dataclass
class ProviderConfig:
    """Configuration for AI provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OpenAIProvider:
    """
    Integration with OpenAI API.

    Supports: GPT-3.5-Turbo, GPT-4, GPT-4-Turbo, etc.

    Example:
        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

        result = provider.inference(
            model_id="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Classify this event..."}],
            temperature=0.7,
            max_tokens=100
        )
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize OpenAI provider."""
        self.config = config or ProviderConfig()

        # Try to get API key from config or environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")

        if not self.config.api_key:
            logger.warning("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client

    def inference(self,
                  model_id: str,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Run inference using OpenAI API.

        Args:
            model_id: Model identifier (e.g., "gpt-3.5-turbo", "gpt-4")
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters

        Returns:
            Dictionary with:
                - output: Generated text
                - confidence: Estimated confidence (0.0-1.0)
                - cost_usd: Estimated cost
                - metadata: Additional info (model, tokens, etc.)
        """
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")

        client = self._get_client()

        start_time = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract response
            choice = response.choices[0]
            output = choice.message.content
            finish_reason = choice.finish_reason

            # Calculate cost (approximate)
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            cost_usd = self._estimate_cost(model_id, input_tokens, output_tokens)

            # Estimate confidence from finish_reason and logprobs if available
            confidence = 0.9 if finish_reason == "stop" else 0.7

            return {
                "output": output,
                "confidence": confidence,
                "cost_usd": cost_usd,
                "metadata": {
                    "model": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": usage.total_tokens,
                    "finish_reason": finish_reason,
                    "latency_ms": latency_ms
                }
            }

        except Exception as e:
            logger.exception(f"OpenAI inference failed for {model_id}")
            raise RuntimeError(f"OpenAI API error: {e}")

    def _estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Pricing as of Jan 2025 (approximate)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        # Default to gpt-4o pricing if unknown
        rates = pricing.get(model_id, pricing["gpt-4o"])

        input_cost = (input_tokens / 1000.0) * rates["input"]
        output_cost = (output_tokens / 1000.0) * rates["output"]

        return input_cost + output_cost


class AnthropicProvider:
    """
    Integration with Anthropic API.

    Supports: Claude 3 Opus, Claude 3.5 Sonnet, Claude 3 Haiku, etc.

    Example:
        provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))

        result = provider.inference(
            model_id="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Analyze this data..."}],
            max_tokens=1024
        )
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize Anthropic provider."""
        self.config = config or ProviderConfig()

        # Try to get API key from config or environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.config.api_key:
            logger.warning("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")

        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    def inference(self,
                  model_id: str,
                  messages: List[Dict[str, str]],
                  max_tokens: int = 1024,
                  temperature: float = 1.0,
                  **kwargs) -> Dict[str, Any]:
        """
        Run inference using Anthropic API.

        Args:
            model_id: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            messages: Messages in Anthropic format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional Anthropic API parameters

        Returns:
            Dictionary with:
                - output: Generated text
                - confidence: Estimated confidence (0.0-1.0)
                - cost_usd: Estimated cost
                - metadata: Additional info (model, tokens, etc.)
        """
        if not self.config.api_key:
            raise ValueError("Anthropic API key not configured")

        client = self._get_client()

        start_time = time.perf_counter()

        try:
            response = client.messages.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract response
            output = response.content[0].text if response.content else ""
            stop_reason = response.stop_reason

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            cost_usd = self._estimate_cost(model_id, input_tokens, output_tokens)

            # Estimate confidence
            confidence = 0.9 if stop_reason == "end_turn" else 0.7

            return {
                "output": output,
                "confidence": confidence,
                "cost_usd": cost_usd,
                "metadata": {
                    "model": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "stop_reason": stop_reason,
                    "latency_ms": latency_ms
                }
            }

        except Exception as e:
            logger.exception(f"Anthropic inference failed for {model_id}")
            raise RuntimeError(f"Anthropic API error: {e}")

    def _estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Pricing as of Jan 2025 (approximate)
        pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }

        # Default to Sonnet pricing if unknown
        rates = pricing.get(model_id, pricing["claude-3-5-sonnet-20241022"])

        input_cost = (input_tokens / 1000.0) * rates["input"]
        output_cost = (output_tokens / 1000.0) * rates["output"]

        return input_cost + output_cost


def create_openai_inference_fn(provider: OpenAIProvider, model_id: str, **default_params):
    """
    Create an inference function for ModelRegistry.

    This allows you to register OpenAI models in the registry.

    Example:
        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

        inference_fn = create_openai_inference_fn(
            provider=provider,
            model_id="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )

        registry.register(ModelSpec(
            name="gpt-3.5-turbo-classifier",
            provider=ModelProvider.OPENAI,
            model_id="gpt-3.5-turbo",
            avg_latency_ms=800.0,
            cost_per_1k_tokens=0.002,
            capabilities=[ModelCapability.CHAT, ModelCapability.CLASSIFICATION],
            inference_fn=inference_fn
        ))
    """

    def inference_fn(input_data: Any, context: Dict) -> Dict:
        """Wrapper function for OpenAI inference."""
        # Convert input to messages format
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, dict) and "messages" in input_data:
            messages = input_data["messages"]
        elif isinstance(input_data, dict):
            # Try to format as a prompt
            messages = [{"role": "user", "content": str(input_data)}]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        # Merge context params with defaults
        params = {**default_params}
        if "temperature" in context:
            params["temperature"] = context["temperature"]
        if "max_tokens" in context:
            params["max_tokens"] = context["max_tokens"]

        return provider.inference(
            model_id=model_id,
            messages=messages,
            **params
        )

    return inference_fn


def create_anthropic_inference_fn(provider: AnthropicProvider, model_id: str, **default_params):
    """
    Create an inference function for ModelRegistry.

    This allows you to register Anthropic models in the registry.

    Example:
        provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))

        inference_fn = create_anthropic_inference_fn(
            provider=provider,
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=1.0
        )

        registry.register(ModelSpec(
            name="claude-sonnet-analyst",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            avg_latency_ms=1200.0,
            cost_per_1k_tokens=0.009,
            capabilities=[ModelCapability.CHAT, ModelCapability.REASONING],
            inference_fn=inference_fn
        ))
    """

    def inference_fn(input_data: Any, context: Dict) -> Dict:
        """Wrapper function for Anthropic inference."""
        # Convert input to messages format
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, dict) and "messages" in input_data:
            messages = input_data["messages"]
        elif isinstance(input_data, dict):
            messages = [{"role": "user", "content": str(input_data)}]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        # Merge context params with defaults
        params = {**default_params}
        if "temperature" in context:
            params["temperature"] = context["temperature"]
        if "max_tokens" in context:
            params["max_tokens"] = context["max_tokens"]

        return provider.inference(
            model_id=model_id,
            messages=messages,
            **params
        )

    return inference_fn
