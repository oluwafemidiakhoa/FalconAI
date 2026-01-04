"""
Live dashboard server for FALCON-AI.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import normalize_config, build_falcon, FalconSpec, ComponentSpec
from ..distributed import FalconSwarm
from ..showcase import build_swarm
from ..sim.scenarios import ScenarioRegistry
from ..sim.evaluation import evaluate_decision, SimulationMetrics


class EventBus:
    def __init__(self):
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def publish(self, event: Dict[str, Any]):
        self.queue.put(event)

    async def stream(self):
        while True:
            try:
                event = await asyncio.get_running_loop().run_in_executor(None, self.queue.get, True, 1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"


class DashboardSimulator:
    def __init__(
        self,
        config: Dict[str, Any],
        scenario: str,
        length: int,
        seed: int,
        mode: str,
        interval_ms: int,
    ):
        self.config = config
        self.scenario_name = scenario
        self.length = length
        self.seed = seed
        self.mode = mode
        self.interval_ms = interval_ms

        self.event_bus = EventBus()
        self.metrics = SimulationMetrics()
        self.solo_metrics = SimulationMetrics()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_id = 0

        self._build_actors()

    def _build_actors(self):
        self.app_cfg = normalize_config(self.config)
        memory_spec = self.app_cfg.falcon.memory or ComponentSpec("experience", {"max_size": 1000})
        if self.mode == "swarm":
            self.actor = build_swarm(self.config, num_agents=5)
            solo_spec = FalconSpec(
                perception=self.app_cfg.falcon.perception,
                decision=self.app_cfg.falcon.decision,
                correction=self.app_cfg.falcon.correction,
                energy=self.app_cfg.falcon.energy,
                memory=memory_spec,
                monitoring=self.app_cfg.falcon.monitoring,
            )
            self.solo_actor = build_falcon(solo_spec)
        else:
            self.actor = build_falcon(FalconSpec(
                perception=self.app_cfg.falcon.perception,
                decision=self.app_cfg.falcon.decision,
                correction=self.app_cfg.falcon.correction,
                energy=self.app_cfg.falcon.energy,
                memory=memory_spec,
                monitoring=self.app_cfg.falcon.monitoring,
            ))
            self.solo_actor = None

        self._reset_metrics()
        self._reset_iterator()

    def _reset_metrics(self):
        self.metrics = SimulationMetrics()
        self.solo_metrics = SimulationMetrics()

    def _reset_budgets(self):
        if isinstance(self.actor, FalconSwarm):
            for agent in self.actor.agents:
                agent.energy_manager.reset_budget()
        else:
            self.actor.energy_manager.reset_budget()

        if self.solo_actor is not None:
            self.solo_actor.energy_manager.reset_budget()

    def _reset_iterator(self):
        self._iterator = iter(ScenarioRegistry.get(self.scenario_name, self.length, self.seed).generate())

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def update(self, payload: Dict[str, Any]):
        self.scenario_name = payload.get("scenario", self.scenario_name)
        self.length = int(payload.get("length", self.length))
        self.seed = int(payload.get("seed", self.seed))
        self.mode = payload.get("mode", self.mode)
        self.interval_ms = int(payload.get("interval_ms", self.interval_ms))

        running = payload.get("running", self._running)
        self.stop()
        self._build_actors()
        if running:
            self.start()

    def _run_loop(self):
        while self._running:
            try:
                sample = next(self._iterator)
            except StopIteration:
                self.seed += 1
                self._reset_iterator()
                self._reset_budgets()
                continue

            self._event_id += 1
            self.metrics.total_events += 1
            if sample.label:
                self.metrics.actionable_events += 1

            start = time.perf_counter()
            decision = self.actor.process(sample.data, sample.metadata)
            latency = (time.perf_counter() - start) * 1000

            if decision is not None:
                self.metrics.decisions_made += 1
                self.metrics.actions_taken += 1

            outcome = evaluate_decision(decision, sample.label)
            if outcome.outcome_type.value == "success":
                self.metrics.correct_actions += 1
            else:
                if sample.label:
                    self.metrics.false_negatives += 1
                else:
                    self.metrics.false_positives += 1

            if isinstance(self.actor, FalconSwarm):
                if decision is not None:
                    self.actor.observe(decision, outcome)
            else:
                if decision is not None:
                    self.actor.observe(decision, outcome, sample.metadata)

            # Solo comparison for swarm mode
            solo_payload = None
            if self.solo_actor is not None:
                self.solo_metrics.total_events += 1
                solo_decision = self.solo_actor.process(sample.data, sample.metadata)
                if solo_decision is not None:
                    self.solo_metrics.decisions_made += 1
                    self.solo_metrics.actions_taken += 1
                solo_outcome = evaluate_decision(solo_decision, sample.label)
                if solo_outcome.outcome_type.value == "success":
                    self.solo_metrics.correct_actions += 1
                else:
                    if sample.label:
                        self.solo_metrics.false_negatives += 1
                    else:
                        self.solo_metrics.false_positives += 1
                if solo_decision is not None:
                    self.solo_actor.observe(solo_decision, solo_outcome, sample.metadata)

                solo_payload = {
                    "success_rate": self._rate(self.solo_metrics.correct_actions, self.solo_metrics.total_events),
                    "avg_reward": None,
                }

            event_payload = self._build_event_payload(sample, decision, outcome, latency, solo_payload)
            self.event_bus.publish(event_payload)
            time.sleep(self.interval_ms / 1000.0)

    def _build_event_payload(self, sample, decision, outcome, latency, solo_payload):
        metrics_snapshot = {
            "total_events": self.metrics.total_events,
            "success_rate": self._rate(self.metrics.correct_actions, self.metrics.total_events),
            "trigger_rate": self._rate(self.metrics.actions_taken, self.metrics.total_events),
            "false_positives": self.metrics.false_positives,
            "false_negatives": self.metrics.false_negatives,
        }

        energy_status = self._energy_status()
        memory_status = self._memory_status()
        consensus = None
        shared_pool = None

        if isinstance(self.actor, FalconSwarm):
            shared_pool = self.actor.shared_pool.get_stats()
            if decision is not None:
                consensus = decision.metadata or {}

        return {
            "id": self._event_id,
            "timestamp": time.time(),
            "scenario": self.scenario_name,
            "label": sample.label,
            "data": sample.data,
            "interval_ms": self.interval_ms,
            "decision": _decision_payload(decision),
            "outcome": {
                "type": outcome.outcome_type.value,
                "reward": outcome.reward,
            },
            "latency_ms": latency,
            "mode": self.mode,
            "metrics": metrics_snapshot,
            "energy": energy_status,
            "memory": memory_status,
            "consensus": consensus,
            "shared_pool": shared_pool,
            "solo": solo_payload,
        }

    def _energy_status(self) -> Dict[str, Any]:
        if isinstance(self.actor, FalconSwarm):
            return self.actor.agents[0].energy_manager.get_budget_status()
        return self.actor.energy_manager.get_budget_status()

    def _memory_status(self) -> Dict[str, Any]:
        if isinstance(self.actor, FalconSwarm):
            memory = self.actor.agents[0].memory
        else:
            memory = self.actor.memory

        if not memory:
            return {"size": 0, "type": "none"}
        return {"size": memory.size(), "type": memory.memory_type.value}

    @staticmethod
    def _rate(numerator: int, denom: int) -> float:
        return numerator / max(denom, 1)


def create_app(
    config: Optional[Dict[str, Any]] = None,
    scenario: str = "spike",
    length: int = 500,
    seed: int = 42,
    mode: str = "swarm",
    interval_ms: int = 500,
    report_path: Optional[str] = None,
    swarm_report_path: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="FALCON-AI Dashboard")

    simulator = DashboardSimulator(
        config=config or {},
        scenario=scenario,
        length=length,
        seed=seed,
        mode=mode,
        interval_ms=interval_ms,
    )
    simulator.start()

    app.state.simulator = simulator
    benchmark_path = _resolve_report_path(report_path, "benchmark_*.json")
    swarm_path = _resolve_report_path(swarm_report_path, "swarm_showcase*.json")

    app.state.benchmark_report = _load_report(benchmark_path)
    app.state.swarm_report = _load_report(swarm_path)

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/stream")
    async def stream():
        return StreamingResponse(simulator.event_bus.stream(), media_type="text/event-stream")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    async def status():
        payload = {
            "mode": simulator.mode,
            "scenario": simulator.scenario_name,
            "interval_ms": simulator.interval_ms,
            "metrics": asdict(simulator.metrics),
            "memory": simulator._memory_status(),
            "energy": simulator._energy_status(),
        }
        return JSONResponse(payload)

    @app.get("/api/report")
    async def report():
        return JSONResponse(app.state.benchmark_report or {})

    @app.get("/api/swarm-report")
    async def swarm_report():
        return JSONResponse(app.state.swarm_report or {})

    @app.post("/api/control")
    async def control(request: Request):
        payload = await request.json()
        simulator.update(payload)
        return JSONResponse({"status": "ok", "config": payload})

    @app.on_event("shutdown")
    async def shutdown_event():
        simulator.stop()

    return app


def _decision_payload(decision):
    if decision is None:
        return None
    return {
        "action": decision.action.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "metadata": decision.metadata or {},
    }


def _resolve_report_path(path: Optional[str], pattern: str) -> Optional[Path]:
    if path:
        return Path(path)

    report_dir = Path("reports")
    if not report_dir.exists():
        return None

    matches = list(report_dir.glob(pattern))
    if not matches:
        return None

    return max(matches, key=lambda item: item.stat().st_mtime)


def _load_report(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return None
