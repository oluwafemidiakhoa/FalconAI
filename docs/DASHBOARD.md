# Live Dashboard

The live dashboard streams events, decisions, energy usage, memory state, and swarm consensus in real time.

## Run

```
falcon-ai serve --scenario spike --mode swarm
```

Open http://127.0.0.1:8000

If `reports/benchmark_*.json` or `reports/swarm_showcase*.json` exist, the dashboard auto-loads the newest files.

## Controls

- Scenario selector
- Mode switch: solo or swarm
- Interval slider to adjust the tick speed
- Pause / resume
- Showcase toggle to auto-cycle scenarios
- Export button to download a JSON snapshot
- Presentation toggle to hide controls and auto-cycle scenarios

Tip: press `P` to toggle presentation mode or `Esc` to exit.

## API Endpoints

- `GET /api/stream` - Server-Sent Events stream
- `GET /api/status` - Current mode and metrics
- `POST /api/control` - Update scenario, mode, interval, or running state
- `GET /api/report` - Benchmark report JSON (if supplied)
- `GET /api/swarm-report` - Swarm showcase JSON (if supplied)
