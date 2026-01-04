# FALCON-AI CLI

The CLI drives repeatable runs, benchmarks, swarm showcases, and the live dashboard.

## Commands

```
falcon-ai run --config configs/demo.yaml
falcon-ai serve --scenario spike --mode swarm
falcon-ai benchmark --output-dir reports
falcon-ai swarm-demo --scenario drift
falcon-ai wow
```

## Config Schema (YAML)

```yaml
falcon:
  perception:
    type: threshold
    params:
      threshold: 0.6
  decision:
    type: memory_aware
    params:
      memory_weight: 0.35
  correction:
    type: outcome_based
    params:
      learning_rate: 0.1
  energy:
    type: simple
    params:
      max_operations: 5000
  memory:
    type: experience
    params:
      max_size: 1000
  monitoring: true

scenario:
  name: spike
  length: 500
  seed: 42

output:
  metrics_path: reports/metrics.json
  checkpoint_path: checkpoints/falcon_run
```

Sample config: `configs/wow.yaml`

## Config Schema (JSON)

```json
{
  "falcon": {
    "perception": {"type": "threshold", "params": {"threshold": 0.6}},
    "decision": {"type": "memory_aware", "params": {"memory_weight": 0.35}},
    "correction": {"type": "outcome_based", "params": {"learning_rate": 0.1}},
    "energy": {"type": "simple", "params": {"max_operations": 5000}},
    "memory": {"type": "experience", "params": {"max_size": 1000}},
    "monitoring": true
  },
  "scenario": {"name": "spike", "length": 500, "seed": 42},
  "output": {"metrics_path": "reports/metrics.json", "checkpoint_path": "checkpoints/falcon_run"}
}
```

## Component Types

- perception: `threshold`, `change_detection`, `anomaly`, `composite`, `neural`, `online_neural`
- decision: `heuristic`, `threshold`, `rule_based`, `hybrid`, `ml`, `ensemble`, `memory_aware`
- correction: `outcome_based`, `bayesian`, `reinforcement`
- energy: `simple`, `adaptive`, `multi_tier`
- memory: `experience`, `instinct`, `none`

## Notes

- `falcon-ai run` prints a JSON summary of simulation + system metrics.
- `falcon-ai run --checkpoint` writes a `.falcon` checkpoint plus JSON metadata.
- `falcon-ai benchmark` writes JSON + Markdown reports under `reports/`.
