"""
Configuration loading and component factory for FALCON-AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .core import FalconAI
from .perception import (
    ThresholdPerception,
    ChangeDetectionPerception,
    AnomalyPerception,
    CompositePerception,
)
from .ml import NeuralPerception, OnlineNeuralPerception
from .decision import (
    HeuristicDecision,
    ThresholdDecision,
    RuleBasedDecision,
    HybridDecision,
)
from .decision.base import ActionType
from .correction import OutcomeBasedCorrection, BayesianCorrection, ReinforcementCorrection
from .energy import (
    ComputeBudget,
    SimpleEnergyManager,
    AdaptiveEnergyManager,
    MultiTierEnergyManager,
)
from .memory import ExperienceMemory, InstinctMemory


@dataclass
class ComponentSpec:
    """Simple component specification from config."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FalconSpec:
    perception: ComponentSpec
    decision: ComponentSpec
    correction: ComponentSpec
    energy: ComponentSpec
    memory: Optional[ComponentSpec] = None
    monitoring: bool = True


@dataclass
class ScenarioSpec:
    name: str = "spike"
    length: int = 500
    seed: int = 42


@dataclass
class OutputSpec:
    metrics_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    report_dir: Optional[str] = None


@dataclass
class AppConfig:
    falcon: FalconSpec
    scenario: ScenarioSpec = field(default_factory=ScenarioSpec)
    output: OutputSpec = field(default_factory=OutputSpec)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = config_path.read_text()
    if config_path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml.")
        return yaml.safe_load(raw) or {}

    if config_path.suffix == ".json":
        return json.loads(raw)

    # Fallback: try JSON, then YAML
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if yaml is None:
            raise RuntimeError("Unknown config format and PyYAML not installed.")
        return yaml.safe_load(raw) or {}


def _as_component(spec: Optional[Dict[str, Any]]) -> Optional[ComponentSpec]:
    if spec is None:
        return None
    if isinstance(spec, ComponentSpec):
        return spec
    return ComponentSpec(type=spec.get("type", ""), params=spec.get("params", {}) or {})


def normalize_config(config: Dict[str, Any]) -> AppConfig:
    falcon_cfg = config.get("falcon", {})

    falcon_spec = FalconSpec(
        perception=_as_component(falcon_cfg.get("perception"))
        or ComponentSpec("threshold", {"threshold": 0.7}),
        decision=_as_component(falcon_cfg.get("decision"))
        or ComponentSpec("heuristic", {}),
        correction=_as_component(falcon_cfg.get("correction"))
        or ComponentSpec("outcome_based", {}),
        energy=_as_component(falcon_cfg.get("energy"))
        or ComponentSpec("simple", {"max_operations": 5000}),
        memory=_as_component(falcon_cfg.get("memory")),
        monitoring=bool(falcon_cfg.get("monitoring", True)),
    )

    scenario_cfg = config.get("scenario", {}) or {}
    scenario_spec = ScenarioSpec(
        name=str(scenario_cfg.get("name", "spike")),
        length=int(scenario_cfg.get("length", 500)),
        seed=int(scenario_cfg.get("seed", 42)),
    )

    output_cfg = config.get("output", {}) or {}
    output_spec = OutputSpec(
        metrics_path=output_cfg.get("metrics_path"),
        checkpoint_path=output_cfg.get("checkpoint_path"),
        report_dir=output_cfg.get("report_dir"),
    )

    return AppConfig(falcon=falcon_spec, scenario=scenario_spec, output=output_spec)


def _build_budget(params: Dict[str, Any]) -> ComputeBudget:
    return ComputeBudget(
        max_operations=int(params.get("max_operations", 10000)),
        max_latency_ms=float(params.get("max_latency_ms", 1000.0)),
        max_energy_units=float(params.get("max_energy_units", 500.0)),
    )


def build_perception(spec: ComponentSpec):
    p_type = spec.type.lower()
    params = spec.params

    if p_type == "threshold":
        return ThresholdPerception(**params)
    if p_type in {"change_detection", "change"}:
        return ChangeDetectionPerception(**params)
    if p_type in {"anomaly", "anomaly_detection"}:
        return AnomalyPerception(**params)
    if p_type in {"neural", "neural_perception"}:
        return NeuralPerception(**params)
    if p_type in {"online_neural", "online_neural_perception"}:
        return OnlineNeuralPerception(**params)
    if p_type == "composite":
        engines = []
        for engine_spec in params.get("engines", []):
            engine = build_perception(_as_component(engine_spec) or ComponentSpec("threshold", {}))
            engines.append(engine)
        return CompositePerception(engines)

    raise ValueError(f"Unknown perception type: {spec.type}")


def build_decision(spec: ComponentSpec):
    d_type = spec.type.lower()
    params = spec.params

    if d_type == "heuristic":
        return HeuristicDecision(**params)
    if d_type == "threshold":
        return ThresholdDecision(**params)
    if d_type in {"rule_based", "rules"}:
        core = RuleBasedDecision()
        for rule in params.get("rules", []):
            condition = _build_rule_condition(rule)
            action_value = rule.get("action", "observe")
            if isinstance(action_value, ActionType):
                action = action_value
            else:
                action = ActionType(str(action_value))
            reasoning = rule.get("reasoning", "Rule matched")
            core.add_rule(condition, action, reasoning)
        return core
    if d_type == "hybrid":
        return HybridDecision(**params)
    if d_type in {"ml", "ml_decision"}:
        from .ml import MLDecisionCore

        return MLDecisionCore(**params)
    if d_type in {"ensemble"}:
        from .ml import EnsembleDecision

        cores = [build_decision(_as_component(c) or ComponentSpec("heuristic", {})) for c in params.get("cores", [])]
        if not cores:
            cores = [HeuristicDecision(), ThresholdDecision()]
        return EnsembleDecision(cores, voting=params.get("voting", "soft"))
    if d_type in {"memory_aware", "context_aware"}:
        from .decision.policies import MemoryAwareDecision

        return MemoryAwareDecision(**params)

    raise ValueError(f"Unknown decision type: {spec.type}")


def _build_rule_condition(rule: Dict[str, Any]) -> Callable[[Any], bool]:
    event_type = rule.get("event_type")
    salience_gt = rule.get("salience_gt")
    salience_lt = rule.get("salience_lt")

    def _condition(event: Any) -> bool:
        try:
            if event_type and getattr(event, "event_type", None) is not None:
                if event.event_type.value != event_type:
                    return False
            if salience_gt is not None and getattr(event, "salience_score", 0.0) <= float(salience_gt):
                return False
            if salience_lt is not None and getattr(event, "salience_score", 0.0) >= float(salience_lt):
                return False
            return True
        except Exception:
            return False

    return _condition


def build_correction(spec: ComponentSpec):
    c_type = spec.type.lower()
    params = spec.params

    if c_type in {"outcome_based", "outcome"}:
        return OutcomeBasedCorrection(**params)
    if c_type in {"bayesian", "bayes"}:
        return BayesianCorrection(**params)
    if c_type in {"reinforcement", "rl"}:
        return ReinforcementCorrection(**params)

    raise ValueError(f"Unknown correction type: {spec.type}")


def build_energy(spec: ComponentSpec):
    e_type = spec.type.lower()
    params = spec.params

    if e_type == "simple":
        return SimpleEnergyManager(**params)
    if e_type == "adaptive":
        budget = _build_budget(params.get("budget", params))
        return AdaptiveEnergyManager(budget, adaptation_rate=float(params.get("adaptation_rate", 0.1)))
    if e_type in {"multi_tier", "multitier"}:
        budget = _build_budget(params.get("budget", params))
        return MultiTierEnergyManager(budget)

    raise ValueError(f"Unknown energy type: {spec.type}")


def build_memory(spec: Optional[ComponentSpec]):
    if spec is None:
        return None
    m_type = spec.type.lower()
    params = spec.params

    if m_type in {"experience", "experience_memory"}:
        return ExperienceMemory(**params)
    if m_type in {"instinct", "instinct_memory"}:
        return InstinctMemory(**params)
    if m_type in {"none", "null", "disabled"}:
        return None

    raise ValueError(f"Unknown memory type: {spec.type}")


def build_falcon(spec: FalconSpec) -> FalconAI:
    perception = build_perception(spec.perception)
    decision = build_decision(spec.decision)
    correction = build_correction(spec.correction)
    energy_manager = build_energy(spec.energy)
    memory = build_memory(spec.memory)

    return FalconAI(
        perception=perception,
        decision=decision,
        correction=correction,
        energy_manager=energy_manager,
        memory=memory,
        enable_monitoring=spec.monitoring,
    )
