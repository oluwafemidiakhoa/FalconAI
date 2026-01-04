# Benchmarks

Benchmarks compare decision cores and energy managers across repeatable scenarios.

## Run

```
falcon-ai benchmark --scenarios spike,drift,attack,pulse --decisions heuristic,threshold,hybrid,memory_aware --energies simple,adaptive,multi_tier
```

## Output

- JSON report: full metrics per scenario and configuration
- Markdown report: quick table summary

## Metrics

- Success rate: accuracy across all events
- Trigger rate: how often actions are taken
- Avg reward: outcome quality signal
- Energy used: operations consumed for the run
