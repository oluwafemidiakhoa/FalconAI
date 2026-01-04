"""
Swarm showcase utilities for FALCON-AI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import FalconSpec, ComponentSpec, build_falcon, normalize_config
from .distributed import FalconSwarm
from .sim.scenarios import ScenarioRegistry, EventSample
from .sim.evaluation import evaluate_decision, SimulationMetrics


@dataclass
class SwarmShowcaseResult:
    swarm_metrics: Dict[str, Any]
    solo_metrics: Dict[str, Any]
    delta: Dict[str, Any]


def build_swarm(app_config: Dict[str, Any], num_agents: int = 5) -> FalconSwarm:
    cfg = normalize_config(app_config)
    memory_spec = cfg.falcon.memory or ComponentSpec("experience", {"max_size": 1000})

    def _factory():
        decision_params = cfg.falcon.decision.params if cfg.falcon.decision.type == "memory_aware" else {}
        decision_spec = ComponentSpec("memory_aware", decision_params)
        falcon_spec = FalconSpec(
            perception=cfg.falcon.perception,
            decision=decision_spec,
            correction=cfg.falcon.correction,
            energy=cfg.falcon.energy,
            memory=memory_spec,
            monitoring=cfg.falcon.monitoring,
        )
        return build_falcon(falcon_spec)

    return FalconSwarm(num_agents=num_agents, agent_factory=_factory, consensus_method="weighted")


def run_swarm_showcase(
    scenario_name: str = "spike",
    length: int = 500,
    seed: int = 42,
    num_agents: int = 5,
    app_config: Optional[Dict[str, Any]] = None,
) -> SwarmShowcaseResult:
    cfg = normalize_config(app_config or {})
    scenario = ScenarioRegistry.get(scenario_name, length, seed)
    samples = list(scenario.generate())

    swarm = build_swarm(app_config or {}, num_agents=num_agents)

    memory_spec = cfg.falcon.memory or ComponentSpec("experience", {"max_size": 1000})
    solo_spec = FalconSpec(
        perception=cfg.falcon.perception,
        decision=cfg.falcon.decision,
        correction=cfg.falcon.correction,
        energy=cfg.falcon.energy,
        memory=memory_spec,
        monitoring=cfg.falcon.monitoring,
    )
    solo = build_falcon(solo_spec)

    swarm_metrics = _run_samples(swarm, samples)
    solo_metrics = _run_samples(solo, samples)

    delta = {
        "success_rate": swarm_metrics.success_rate - solo_metrics.success_rate,
        "avg_reward": swarm_metrics.avg_reward - solo_metrics.avg_reward,
        "trigger_rate": swarm_metrics.trigger_rate - solo_metrics.trigger_rate,
    }

    return SwarmShowcaseResult(
        swarm_metrics=swarm_metrics.to_dict(),
        solo_metrics=solo_metrics.to_dict(),
        delta=delta,
    )


def _run_samples(actor: Any, samples: List[EventSample]) -> SimulationMetrics:
    metrics = SimulationMetrics()
    rewards: List[float] = []
    confidences: List[float] = []

    for sample in samples:
        metrics.total_events += 1
        if sample.label:
            metrics.actionable_events += 1

        decision = actor.process(sample.data, sample.metadata)
        if decision is not None:
            metrics.decisions_made += 1
            metrics.actions_taken += 1
            confidences.append(decision.confidence)

        outcome = evaluate_decision(decision, sample.label)
        rewards.append(outcome.reward)

        if outcome.outcome_type.value == "success":
            metrics.correct_actions += 1
        else:
            if sample.label:
                metrics.false_negatives += 1
            else:
                metrics.false_positives += 1

        if isinstance(actor, FalconSwarm):
            if decision is not None:
                actor.observe(decision, outcome)
        else:
            if decision is not None:
                actor.observe(decision, outcome, sample.metadata)

    metrics.avg_reward = sum(rewards) / max(len(rewards), 1)
    metrics.avg_confidence = sum(confidences) / max(len(confidences), 1)
    metrics.trigger_rate = metrics.actions_taken / max(metrics.total_events, 1)
    metrics.success_rate = metrics.correct_actions / max(metrics.total_events, 1)

    if isinstance(actor, FalconSwarm):
        energy_stats = actor.agents[0].energy_manager.get_budget_status()
        metrics.energy_used = int(energy_stats.get("used_operations", 0))
        metrics.energy_remaining_fraction = float(energy_stats.get("remaining_fraction", 0.0))
        if actor.agents[0].memory:
            metrics.memory_size = actor.agents[0].memory.size()
    else:
        energy_stats = actor.energy_manager.get_budget_status()
        metrics.energy_used = int(energy_stats.get("used_operations", 0))
        metrics.energy_remaining_fraction = float(energy_stats.get("remaining_fraction", 0.0))
        if actor.memory:
            metrics.memory_size = actor.memory.size()

    return metrics
