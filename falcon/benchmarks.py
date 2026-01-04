"""
Benchmark runner and report generator for FALCON-AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from .config import ComponentSpec, FalconSpec, normalize_config, build_falcon
from .sim.scenarios import ScenarioRegistry
from .sim.evaluation import run_simulation


@dataclass
class BenchmarkResult:
    scenario: str
    decision: str
    energy: str
    metrics: Dict[str, Any]
    duration_s: float


@dataclass
class BenchmarkReport:
    created_at: str
    scenarios: List[str]
    decisions: List[str]
    energies: List[str]
    repeats: int
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "scenarios": self.scenarios,
            "decisions": self.decisions,
            "energies": self.energies,
            "repeats": self.repeats,
            "results": [
                {
                    "scenario": r.scenario,
                    "decision": r.decision,
                    "energy": r.energy,
                    "duration_s": r.duration_s,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }


def run_benchmarks(
    scenarios: List[str],
    decisions: List[str],
    energies: List[str],
    repeats: int = 3,
    seed: int = 42,
    base_config: Optional[Dict[str, Any]] = None,
) -> BenchmarkReport:
    app_cfg = normalize_config(base_config or {})

    report = BenchmarkReport(
        created_at=datetime.utcnow().isoformat() + "Z",
        scenarios=scenarios,
        decisions=decisions,
        energies=energies,
        repeats=repeats,
    )

    for scenario_name in scenarios:
        for decision_type in decisions:
            for energy_type in energies:
                aggregated = []
                durations = []

                for rep in range(repeats):
                    decision_params = app_cfg.falcon.decision.params if decision_type == app_cfg.falcon.decision.type else {}
                    energy_params = app_cfg.falcon.energy.params if energy_type == app_cfg.falcon.energy.type else {}

                    falcon_spec = FalconSpec(
                        perception=app_cfg.falcon.perception,
                        decision=ComponentSpec(decision_type, decision_params),
                        correction=app_cfg.falcon.correction,
                        energy=ComponentSpec(energy_type, energy_params),
                        memory=app_cfg.falcon.memory,
                        monitoring=app_cfg.falcon.monitoring,
                    )
                    falcon = build_falcon(falcon_spec)
                    scenario = ScenarioRegistry.get(
                        scenario_name, app_cfg.scenario.length, seed + rep
                    )
                    start = time.perf_counter()
                    result = run_simulation(falcon, scenario)
                    durations.append(time.perf_counter() - start)
                    aggregated.append(result.metrics.to_dict())

                averaged = _average_metrics(aggregated)
                report.results.append(
                    BenchmarkResult(
                        scenario=scenario_name,
                        decision=decision_type,
                        energy=energy_type,
                        metrics=averaged,
                        duration_s=sum(durations) / max(len(durations), 1),
                    )
                )

    return report


def save_benchmark_report(report: BenchmarkReport, output_dir: str) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{timestamp}.json"
    md_path = out_dir / f"benchmark_{timestamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    md_path.write_text(render_markdown_report(report))

    return {"json": json_path, "markdown": md_path}


def render_markdown_report(report: BenchmarkReport) -> str:
    lines = [
        "# FALCON-AI Benchmark Report",
        "",
        f"Generated: {report.created_at}",
        "",
        "## Matrix",
        f"Scenarios: {', '.join(report.scenarios)}",
        f"Decision cores: {', '.join(report.decisions)}",
        f"Energy managers: {', '.join(report.energies)}",
        f"Repeats: {report.repeats}",
        "",
        "## Results",
        "| Scenario | Decision | Energy | Success Rate | Avg Reward | Trigger Rate | Energy Used |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in report.results:
        metrics = result.metrics
        lines.append(
            "| {scenario} | {decision} | {energy} | {success:.1%} | {reward:.2f} | {trigger:.1%} | {energy_used} |".format(
                scenario=result.scenario,
                decision=result.decision,
                energy=result.energy,
                success=metrics.get("success_rate", 0.0),
                reward=metrics.get("avg_reward", 0.0),
                trigger=metrics.get("trigger_rate", 0.0),
                energy_used=int(metrics.get("energy_used", 0)),
            )
        )

    return "\n".join(lines) + "\n"


def _average_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {}

    totals: Dict[str, float] = {}
    for run in runs:
        for key, value in run.items():
            try:
                totals[key] = totals.get(key, 0.0) + float(value)
            except Exception:
                continue

    count = len(runs)
    return {key: value / count for key, value in totals.items()}
