"""
Command-line interface for FALCON-AI.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_config, normalize_config, build_falcon
from .sim.scenarios import ScenarioRegistry
from .sim.evaluation import run_simulation
from .benchmarks import run_benchmarks, save_benchmark_report
from .persistence import save_falcon
from .showcase import run_swarm_showcase


def main() -> None:
    parser = argparse.ArgumentParser(prog="falcon-ai", description="FALCON-AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a scenario with a config")
    run_parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    run_parser.add_argument("--scenario", type=str, help="Scenario name")
    run_parser.add_argument("--length", type=int, help="Scenario length")
    run_parser.add_argument("--seed", type=int, help="Scenario seed")
    run_parser.add_argument("--metrics", type=str, help="Path to write metrics JSON")
    run_parser.add_argument("--checkpoint", type=str, help="Path prefix to save checkpoint")

    serve_parser = subparsers.add_parser("serve", help="Run live dashboard server")
    serve_parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    serve_parser.add_argument("--scenario", type=str, default="spike", help="Scenario name")
    serve_parser.add_argument("--length", type=int, default=500, help="Scenario length")
    serve_parser.add_argument("--seed", type=int, default=42, help="Scenario seed")
    serve_parser.add_argument("--mode", type=str, default="swarm", choices=["solo", "swarm"], help="Simulation mode")
    serve_parser.add_argument("--interval", type=int, default=500, help="Tick interval in ms")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--report", type=str, help="Benchmark report JSON path")
    serve_parser.add_argument("--swarm-report", type=str, help="Swarm report JSON path")
    serve_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark matrix")
    bench_parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    bench_parser.add_argument("--scenarios", type=str, default="spike,drift,attack,pulse", help="Comma-separated scenarios")
    bench_parser.add_argument("--decisions", type=str, default="heuristic,threshold,hybrid,memory_aware", help="Comma-separated decision cores")
    bench_parser.add_argument("--energies", type=str, default="simple,adaptive,multi_tier", help="Comma-separated energy managers")
    bench_parser.add_argument("--repeats", type=int, default=3, help="Repeats per combination")
    bench_parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    bench_parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")

    swarm_parser = subparsers.add_parser("swarm-demo", help="Run swarm showcase")
    swarm_parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    swarm_parser.add_argument("--scenario", type=str, default="spike", help="Scenario name")
    swarm_parser.add_argument("--length", type=int, default=500, help="Scenario length")
    swarm_parser.add_argument("--seed", type=int, default=42, help="Scenario seed")
    swarm_parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    swarm_parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")

    wow_parser = subparsers.add_parser("wow", help="Run benchmark + swarm demo + dashboard")
    wow_parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    wow_parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    wow_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    wow_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    wow_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    args = parser.parse_args()

    if args.command == "run":
        _run_command(args)
    elif args.command == "serve":
        _serve_command(args)
    elif args.command == "benchmark":
        _benchmark_command(args)
    elif args.command == "swarm-demo":
        _swarm_command(args)
    elif args.command == "wow":
        _wow_command(args)


def _load_config_dict(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return load_config(path)


def _run_command(args: argparse.Namespace) -> None:
    config_dict = _load_config_dict(args.config)
    app_cfg = normalize_config(config_dict)

    if args.scenario:
        app_cfg.scenario.name = args.scenario
    if args.length:
        app_cfg.scenario.length = args.length
    if args.seed:
        app_cfg.scenario.seed = args.seed

    falcon = build_falcon(app_cfg.falcon)
    scenario = ScenarioRegistry.get(app_cfg.scenario.name, app_cfg.scenario.length, app_cfg.scenario.seed)
    result = run_simulation(falcon, scenario)

    metrics_payload = {
        "simulation": result.metrics.to_dict(),
        "system": falcon.get_status(),
        "scenario": {
            "name": app_cfg.scenario.name,
            "length": app_cfg.scenario.length,
            "seed": app_cfg.scenario.seed,
        },
    }

    print(json.dumps(metrics_payload, indent=2, default=_json_default))

    if args.metrics or app_cfg.output.metrics_path:
        path = Path(args.metrics or app_cfg.output.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics_payload, indent=2, default=_json_default))
        print(f"[OK] Metrics saved to {path}")

    if args.checkpoint or app_cfg.output.checkpoint_path:
        prefix = args.checkpoint or app_cfg.output.checkpoint_path
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        save_falcon(falcon, str(prefix_path))


def _serve_command(args: argparse.Namespace) -> None:
    from .api.dashboard import create_app

    config_dict = _load_config_dict(args.config)

    dashboard_url = f"http://{args.host}:{args.port}"
    print(f"[OK] Starting dashboard at {dashboard_url}")

    # Auto-open browser unless disabled
    if not args.no_browser:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            print(f"[OK] Opening browser to {dashboard_url}")
            webbrowser.open(dashboard_url)

        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    else:
        print("[INFO] Auto-browser launch disabled. Open manually: " + dashboard_url)

    app = create_app(
        config=config_dict,
        scenario=args.scenario,
        length=args.length,
        seed=args.seed,
        mode=args.mode,
        interval_ms=args.interval,
        report_path=args.report,
        swarm_report_path=args.swarm_report,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the dashboard server") from exc

    print("[OK] Server running (Ctrl+C to stop)...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _benchmark_command(args: argparse.Namespace) -> None:
    config_dict = _load_config_dict(args.config)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    decisions = [d.strip() for d in args.decisions.split(",") if d.strip()]
    energies = [e.strip() for e in args.energies.split(",") if e.strip()]

    report = run_benchmarks(
        scenarios=scenarios,
        decisions=decisions,
        energies=energies,
        repeats=args.repeats,
        seed=args.seed,
        base_config=config_dict,
    )

    outputs = save_benchmark_report(report, args.output_dir)
    print(f"[OK] Benchmark JSON: {outputs['json']}")
    print(f"[OK] Benchmark Markdown: {outputs['markdown']}")


def _swarm_command(args: argparse.Namespace) -> None:
    config_dict = _load_config_dict(args.config)
    result = run_swarm_showcase(
        scenario_name=args.scenario,
        length=args.length,
        seed=args.seed,
        num_agents=args.agents,
        app_config=config_dict,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "swarm_showcase.json"
    md_path = out_dir / "swarm_showcase.md"

    json_path.write_text(json.dumps({
        "swarm_metrics": result.swarm_metrics,
        "solo_metrics": result.solo_metrics,
        "delta": result.delta,
    }, indent=2, default=_json_default))

    md_path.write_text(_render_swarm_md(result))

    print(f"[OK] Swarm JSON: {json_path}")
    print(f"[OK] Swarm Markdown: {md_path}")


def _wow_command(args: argparse.Namespace) -> None:
    config_dict = _load_config_dict(args.config)

    print("[1/3] Running comprehensive benchmarks...")
    report = run_benchmarks(
        scenarios=["spike", "drift", "attack", "pulse"],
        decisions=["heuristic", "threshold", "hybrid", "memory_aware"],
        energies=["simple", "adaptive", "multi_tier"],
        repeats=2,
        seed=42,
        base_config=config_dict,
    )
    outputs = save_benchmark_report(report, args.output_dir)
    print(f"[OK] Benchmark JSON: {outputs['json']}")

    print("[2/3] Running swarm vs solo showcase...")
    swarm_result = run_swarm_showcase(
        scenario_name="spike",
        length=500,
        seed=42,
        num_agents=5,
        app_config=config_dict,
    )

    swarm_json_path = Path(args.output_dir) / "swarm_showcase.json"
    swarm_json_path.write_text(json.dumps({
        "swarm_metrics": swarm_result.swarm_metrics,
        "solo_metrics": swarm_result.solo_metrics,
        "delta": swarm_result.delta,
    }, indent=2, default=_json_default))
    print(f"[OK] Swarm JSON: {swarm_json_path}")

    print("[3/3] Launching live dashboard...")
    dashboard_url = f"http://{args.host}:{args.port}"
    print(f"[OK] Dashboard URL: {dashboard_url}")

    # Auto-open browser unless disabled
    if not args.no_browser:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            print(f"[OK] Opening browser to {dashboard_url}")
            webbrowser.open(dashboard_url)

        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    else:
        print("[INFO] Auto-browser launch disabled. Open manually: " + dashboard_url)

    from .api.dashboard import create_app

    app = create_app(
        config=config_dict,
        scenario="spike",
        length=500,
        seed=42,
        mode="swarm",
        interval_ms=500,
        report_path=str(outputs["json"]),
        swarm_report_path=str(swarm_json_path),
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the dashboard server") from exc

    print("[OK] Starting server (Ctrl+C to stop)...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _render_swarm_md(result: Any) -> str:
    lines = [
        "# FALCON-AI Swarm Showcase",
        "",
        "## Summary",
        "",
        "| Metric | Swarm | Solo | Delta |",
        "| --- | --- | --- | --- |",
        "| Success Rate | {swarm_success:.1%} | {solo_success:.1%} | {delta_success:.1%} |".format(
            swarm_success=result.swarm_metrics.get("success_rate", 0.0),
            solo_success=result.solo_metrics.get("success_rate", 0.0),
            delta_success=result.delta.get("success_rate", 0.0),
        ),
        "| Avg Reward | {swarm_reward:.2f} | {solo_reward:.2f} | {delta_reward:.2f} |".format(
            swarm_reward=result.swarm_metrics.get("avg_reward", 0.0),
            solo_reward=result.solo_metrics.get("avg_reward", 0.0),
            delta_reward=result.delta.get("avg_reward", 0.0),
        ),
        "| Trigger Rate | {swarm_trigger:.1%} | {solo_trigger:.1%} | {delta_trigger:.1%} |".format(
            swarm_trigger=result.swarm_metrics.get("trigger_rate", 0.0),
            solo_trigger=result.solo_metrics.get("trigger_rate", 0.0),
            delta_trigger=result.delta.get("trigger_rate", 0.0),
        ),
        "",
        "## Notes",
        "- Swarm uses consensus voting with memory-aware agents.",
        "- Solo uses the base config decision core.",
    ]

    return "\n".join(lines) + "\n"


def _json_default(obj: Any):
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


if __name__ == "__main__":
    main()
