"""
Scenario generators for FALCON-AI simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Callable, List
import math
import random


@dataclass
class EventSample:
    data: Any
    label: bool
    metadata: Dict[str, Any]


@dataclass
class Scenario:
    name: str
    length: int
    seed: int
    generator: Callable[[int, random.Random], Iterator[EventSample]]

    def generate(self) -> Iterator[EventSample]:
        rng = random.Random(self.seed)
        return self.generator(self.length, rng)


class ScenarioRegistry:
    """Registry for built-in scenarios."""

    _scenarios: Dict[str, Callable[[int, random.Random], Iterator[EventSample]]] = {}

    @classmethod
    def register(cls, name: str):
        def _wrapper(func: Callable[[int, random.Random], Iterator[EventSample]]):
            cls._scenarios[name] = func
            return func

        return _wrapper

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._scenarios.keys())

    @classmethod
    def get(cls, name: str, length: int, seed: int) -> Scenario:
        if name not in cls._scenarios:
            raise ValueError(f"Unknown scenario: {name}")
        return Scenario(name=name, length=length, seed=seed, generator=cls._scenarios[name])


@ScenarioRegistry.register("spike")
def _spike_scenario(length: int, rng: random.Random) -> Iterator[EventSample]:
    """Mostly low values with occasional spikes."""
    for idx in range(length):
        spike = rng.random() < 0.1
        value = rng.uniform(0.8, 1.0) if spike else rng.uniform(0.0, 0.5)
        yield EventSample(
            data=value,
            label=spike,
            metadata={"index": idx, "scenario": "spike"},
        )


@ScenarioRegistry.register("drift")
def _drift_scenario(length: int, rng: random.Random) -> Iterator[EventSample]:
    """Gradual baseline drift with occasional jumps."""
    baseline = 0.3
    for idx in range(length):
        baseline += rng.uniform(-0.01, 0.02)
        baseline = max(0.0, min(1.0, baseline))
        jump = rng.random() < 0.05
        value = baseline + (rng.uniform(0.4, 0.6) if jump else rng.uniform(-0.05, 0.05))
        value = max(0.0, min(1.0, value))
        yield EventSample(
            data=value,
            label=jump,
            metadata={"index": idx, "scenario": "drift", "baseline": baseline},
        )


@ScenarioRegistry.register("attack")
def _attack_scenario(length: int, rng: random.Random) -> Iterator[EventSample]:
    """Vector inputs with rare anomalous attacks."""
    for idx in range(length):
        attack = rng.random() < 0.12
        if attack:
            packet_size = rng.gauss(9000, 800)
            packet_rate = rng.gauss(4500, 400)
            port = rng.choice([22, 23, 3389, 445])
        else:
            packet_size = rng.gauss(500, 80)
            packet_rate = rng.gauss(120, 25)
            port = rng.choice([80, 443, 53])

        features = [packet_size / 10000, packet_rate / 5000, port / 65535]
        yield EventSample(
            data=features,
            label=attack,
            metadata={"index": idx, "scenario": "attack", "port": port},
        )


@ScenarioRegistry.register("pulse")
def _pulse_scenario(length: int, rng: random.Random) -> Iterator[EventSample]:
    """Sine wave with periodic pulses and noise."""
    for idx in range(length):
        t = idx / max(length, 1)
        base = 0.5 + 0.2 * math.sin(2 * math.pi * t * 3)
        pulse = rng.random() < 0.08
        value = base + (0.4 if pulse else 0.0) + rng.uniform(-0.05, 0.05)
        value = max(0.0, min(1.0, value))
        yield EventSample(
            data=value,
            label=pulse,
            metadata={"index": idx, "scenario": "pulse"},
        )


@ScenarioRegistry.register("noise")
def _noise_scenario(length: int, rng: random.Random) -> Iterator[EventSample]:
    """Noisy stream with small anomalies."""
    for idx in range(length):
        anomaly = rng.random() < 0.07
        value = rng.uniform(0.0, 1.0)
        if anomaly:
            value = min(1.0, value + rng.uniform(0.3, 0.5))
        yield EventSample(
            data=value,
            label=anomaly,
            metadata={"index": idx, "scenario": "noise"},
        )
