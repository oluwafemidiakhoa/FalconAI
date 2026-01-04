"""
Simulation evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time

from ..core import FalconAI
from ..decision.base import Decision, ActionType
from ..correction.base import Outcome, OutcomeType
from ..distributed.swarm import FalconSwarm


@dataclass
class SimulationMetrics:
    total_events: int = 0
    actionable_events: int = 0
    decisions_made: int = 0
    actions_taken: int = 0
    correct_actions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    avg_reward: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    trigger_rate: float = 0.0
    success_rate: float = 0.0
    energy_used: int = 0
    energy_remaining_fraction: float = 0.0
    memory_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "actionable_events": self.actionable_events,
            "decisions_made": self.decisions_made,
            "actions_taken": self.actions_taken,
            "correct_actions": self.correct_actions,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "avg_reward": self.avg_reward,
            "avg_confidence": self.avg_confidence,
            "avg_latency_ms": self.avg_latency_ms,
            "trigger_rate": self.trigger_rate,
            "success_rate": self.success_rate,
            "energy_used": self.energy_used,
            "energy_remaining_fraction": self.energy_remaining_fraction,
            "memory_size": self.memory_size,
        }


@dataclass
class SimulationResult:
    metrics: SimulationMetrics
    events: List[Dict[str, Any]] = field(default_factory=list)


def evaluate_decision(decision: Optional[Decision], label: bool) -> Outcome:
    """Return an Outcome based on decision vs ground-truth label."""
    if decision is None:
        if label:
            return Outcome(outcome_type=OutcomeType.FAILURE, reward=-0.7, metadata={"miss": True})
        return Outcome(outcome_type=OutcomeType.SUCCESS, reward=0.2, metadata={"correct_pass": True})

    action = decision.action
    act_on_event = {ActionType.ALERT, ActionType.INTERVENE, ActionType.ESCALATE}
    observe_event = {ActionType.OBSERVE, ActionType.IGNORE}

    if label:
        if action in act_on_event:
            reward = 1.0 * max(decision.confidence, 0.1)
            return Outcome(outcome_type=OutcomeType.SUCCESS, reward=reward)
        reward = -0.7
        return Outcome(outcome_type=OutcomeType.FAILURE, reward=reward)

    if action in observe_event:
        reward = 0.3
        return Outcome(outcome_type=OutcomeType.SUCCESS, reward=reward)

    reward = -0.5
    return Outcome(outcome_type=OutcomeType.FAILURE, reward=reward)


def run_simulation(
    actor: Union[FalconAI, FalconSwarm],
    scenario: Any,
    record_events: bool = False,
    auto_learn: bool = True,
) -> SimulationResult:
    """Run a scenario through a FalconAI or FalconSwarm and score outcomes."""
    metrics = SimulationMetrics()
    events: List[Dict[str, Any]] = []
    rewards: List[float] = []
    confidences: List[float] = []
    latencies: List[float] = []

    for sample in scenario.generate():
        metrics.total_events += 1
        if sample.label:
            metrics.actionable_events += 1

        context = dict(sample.metadata)
        start = time.perf_counter()
        decision = actor.process(sample.data, context)  # type: ignore[arg-type]
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if decision is not None:
            metrics.decisions_made += 1
            metrics.actions_taken += 1
            confidences.append(decision.confidence)

        outcome = evaluate_decision(decision, sample.label)

        if outcome.outcome_type == OutcomeType.SUCCESS:
            if sample.label or decision is None or decision.action in {ActionType.OBSERVE, ActionType.IGNORE}:
                metrics.correct_actions += 1
        else:
            if sample.label:
                metrics.false_negatives += 1
            else:
                metrics.false_positives += 1

        rewards.append(outcome.reward)

        if isinstance(actor, FalconSwarm):
            if decision is not None:
                actor.observe(decision, outcome)
        else:
            if decision is not None:
                actor.observe(decision, outcome, context)

        if auto_learn:
            _update_models(actor, sample.data, decision, sample.label, context)

        if record_events:
            events.append(
                {
                    "data": sample.data,
                    "label": sample.label,
                    "decision": _decision_to_dict(decision),
                    "outcome": {
                        "type": outcome.outcome_type.value,
                        "reward": outcome.reward,
                    },
                    "latency_ms": latency,
                }
            )

    metrics.avg_reward = sum(rewards) / max(len(rewards), 1)
    metrics.avg_confidence = sum(confidences) / max(len(confidences), 1)
    metrics.avg_latency_ms = sum(latencies) / max(len(latencies), 1)
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

    return SimulationResult(metrics=metrics, events=events)


def _decision_to_dict(decision: Optional[Decision]) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    return {
        "action": decision.action.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "metadata": decision.metadata or {},
    }


def _update_models(actor: Union[FalconAI, FalconSwarm], data: Any, decision: Optional[Decision], label: bool, context: Dict[str, Any]):
    if isinstance(actor, FalconSwarm):
        for agent in actor.agents:
            _update_models(agent, data, decision, label, context)
        return

    if hasattr(actor.perception, "update"):
        try:
            actor.perception.update(data, label)
        except Exception:
            pass

    if decision is not None and hasattr(actor.decision, "update"):
        try:
            actor.decision.update(data, decision.action, context)
        except Exception:
            pass
