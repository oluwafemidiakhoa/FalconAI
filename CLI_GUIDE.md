# FALCON-AI Command-Line Interface Guide

Complete guide to using the `falcon-ai` CLI tool.

## Installation

```bash
cd FalconAI
pip install -e .
```

This installs the `falcon-ai` command globally.

## Quick Start

```bash
# Run a basic scenario
falcon-ai run --config configs/basic_config.yaml

# Launch the dashboard
falcon-ai serve --config configs/wow.yaml --port 8000

# Run full demo (benchmarks + dashboard)
falcon-ai wow --config configs/wow.yaml
```

---

## Commands

### `falcon-ai run` - Run a Scenario

Run FALCON-AI on a specific scenario with a configuration file.

**Usage:**
```bash
falcon-ai run --config <config_file> [options]
```

**Options:**
- `--config <path>` - Path to YAML/JSON config file
- `--scenario <name>` - Override scenario name (spike, drift, attack, pulse)
- `--length <int>` - Override scenario length (number of events)
- `--seed <int>` - Override random seed
- `--metrics <path>` - Path to save metrics JSON
- `--checkpoint <path>` - Path prefix to save model checkpoint

**Examples:**
```bash
# Basic run
falcon-ai run --config configs/basic_config.yaml

# Run with custom scenario
falcon-ai run --config configs/ml_config.yaml --scenario attack --length 2000

# Save metrics and checkpoint
falcon-ai run --config configs/network_security.yaml \
    --metrics output/metrics.json \
    --checkpoint output/network_falcon

# Quick test with short scenario
falcon-ai run --config configs/basic_config.yaml --length 100 --seed 42
```

**Output:**
- Prints metrics JSON to stdout
- Optionally saves metrics to file (if `--metrics` specified)
- Optionally saves trained model (if `--checkpoint` specified)

---

### `falcon-ai serve` - Launch Dashboard

Start the live web dashboard server.

**Usage:**
```bash
falcon-ai serve --config <config_file> [options]
```

**Options:**
- `--config <path>` - Path to YAML/JSON config file
- `--scenario <name>` - Scenario to simulate (default: spike)
- `--length <int>` - Scenario length (default: 500)
- `--seed <int>` - Random seed (default: 42)
- `--mode <solo|swarm>` - Simulation mode (default: swarm)
- `--interval <ms>` - Tick interval in milliseconds (default: 500)
- `--host <host>` - Bind host (default: 127.0.0.1)
- `--port <port>` - Bind port (default: 8000)
- `--report <path>` - Path to benchmark report JSON (optional)
- `--swarm-report <path>` - Path to swarm report JSON (optional)

**Examples:**
```bash
# Basic dashboard
falcon-ai serve --config configs/wow.yaml

# Dashboard on custom port
falcon-ai serve --config configs/ml_config.yaml --port 8080

# Dashboard with pre-loaded reports
falcon-ai serve --config configs/network_security.yaml \
    --report reports/benchmark.json \
    --swarm-report reports/swarm.json

# Fast simulation mode
falcon-ai serve --config configs/high_performance.yaml --interval 100

# Solo agent mode
falcon-ai serve --config configs/basic_config.yaml --mode solo
```

**Access:**
- Open browser to `http://127.0.0.1:8000` (or your custom host/port)
- Dashboard updates in real-time
- Use controls to pause, change scenarios, adjust speed

---

### `falcon-ai benchmark` - Run Benchmarks

Run a comprehensive benchmark matrix across scenarios, decision cores, and energy managers.

**Usage:**
```bash
falcon-ai benchmark --config <config_file> [options]
```

**Options:**
- `--config <path>` - Path to YAML/JSON config file (optional)
- `--scenarios <list>` - Comma-separated scenario names (default: spike,drift,attack,pulse)
- `--decisions <list>` - Comma-separated decision cores (default: heuristic,threshold,hybrid,memory_aware)
- `--energies <list>` - Comma-separated energy managers (default: simple,adaptive,multi_tier)
- `--repeats <int>` - Repeats per combination (default: 3)
- `--seed <int>` - Base random seed (default: 42)
- `--output-dir <path>` - Output directory for reports (default: reports)

**Examples:**
```bash
# Full benchmark matrix
falcon-ai benchmark --config configs/wow.yaml

# Quick benchmark (fewer combinations)
falcon-ai benchmark --scenarios spike,attack --decisions heuristic,threshold --repeats 1

# Custom scenario list
falcon-ai benchmark --scenarios spike,drift,pulse \
    --decisions memory_aware,hybrid \
    --energies adaptive,multi_tier \
    --repeats 5

# Save to custom directory
falcon-ai benchmark --output-dir my_reports/
```

**Output:**
- `reports/benchmark_<timestamp>.json` - Full benchmark results
- `reports/benchmark_<timestamp>.md` - Markdown report table
- Prints file paths when complete

**Report Format:**
- Success rate per configuration
- Average reward per configuration
- Trigger rate (selectivity)
- Energy usage
- Duration metrics

---

### `falcon-ai swarm-demo` - Swarm Showcase

Run a swarm vs solo comparison to demonstrate multi-agent advantages.

**Usage:**
```bash
falcon-ai swarm-demo --config <config_file> [options]
```

**Options:**
- `--config <path>` - Path to YAML/JSON config file (optional)
- `--scenario <name>` - Scenario to run (default: spike)
- `--length <int>` - Scenario length (default: 500)
- `--seed <int>` - Random seed (default: 42)
- `--agents <int>` - Number of swarm agents (default: 5)
- `--output-dir <path>` - Output directory (default: reports)

**Examples:**
```bash
# Basic swarm demo
falcon-ai swarm-demo --config configs/network_security.yaml

# Large swarm
falcon-ai swarm-demo --config configs/ml_config.yaml --agents 10

# Long scenario
falcon-ai swarm-demo --scenario attack --length 2000 --agents 7

# Save to custom directory
falcon-ai swarm-demo --output-dir swarm_results/
```

**Output:**
- `reports/swarm_showcase.json` - Detailed metrics
- `reports/swarm_showcase.md` - Comparison table
- Shows swarm vs solo performance delta

**Metrics:**
- Success rate improvement
- Average reward improvement
- Trigger rate comparison
- Shared experience pool stats

---

### `falcon-ai wow` - Full Demo

Run the complete "wow" demonstration: benchmarks + swarm showcase + live dashboard.

**Usage:**
```bash
falcon-ai wow --config <config_file> [options]
```

**Options:**
- `--config <path>` - Path to YAML/JSON config file (optional)
- `--output-dir <path>` - Output directory for reports (default: reports)
- `--host <host>` - Dashboard bind host (default: 127.0.0.1)
- `--port <port>` - Dashboard bind port (default: 8000)

**Examples:**
```bash
# Run full demo
falcon-ai wow --config configs/wow.yaml

# Custom port
falcon-ai wow --config configs/ml_config.yaml --port 8080

# Save reports to custom directory
falcon-ai wow --output-dir presentation_reports/
```

**What It Does:**
1. Runs benchmark matrix (spike, drift, attack, pulse scenarios)
2. Runs swarm vs solo comparison
3. Saves all reports to output directory
4. Launches dashboard with reports pre-loaded
5. Dashboard displays benchmark results in snapshot panel

**Perfect For:**
- Demonstrations
- Presentations
- "Wow the world" showcases
- Quick comprehensive overview

---

## Configuration Files

All commands accept YAML or JSON configuration files.

### Config File Structure

```yaml
falcon:
  perception:
    type: <type>
    params: { }
  decision:
    type: <type>
    params: { }
  correction:
    type: <type>
    params: { }
  energy:
    type: <type>
    params: { }
  memory:
    type: <type>
    params: { }
  monitoring: true

scenario:
  name: <scenario>
  length: <length>
  seed: <seed>

output:
  metrics_path: <path>
  checkpoint_path: <path>
  report_dir: <path>
```

### Available Component Types

**Perception:**
- `threshold` - Simple threshold filtering
- `change_detection` - Detect changes in metrics
- `anomaly` - Statistical anomaly detection
- `neural` - Neural network perception
- `online_neural` - Adaptive neural perception
- `composite` - Combine multiple perception engines

**Decision:**
- `heuristic` - Fast heuristic rules
- `threshold` - Multi-level threshold decisions
- `rule_based` - Custom rule-based decisions
- `hybrid` - Heuristics + learned patterns
- `ml_decision` - Random Forest / Gradient Boosting
- `ensemble` - Combine multiple decision cores
- `memory_aware` - Uses shared swarm memory

**Correction:**
- `outcome_based` - Running average updates
- `bayesian` - Bayesian probability learning
- `reinforcement` - Q-learning

**Energy:**
- `simple` - Basic budget tracking
- `adaptive` - Learns optimal inference modes
- `multi_tier` - Urgency-based mode selection

**Memory:**
- `experience` - Learns from outcomes
- `instinct` - Pretrained patterns

**Scenarios:**
- `spike` - Sudden metric spikes
- `drift` - Gradual drift over time
- `attack` - Network attack patterns
- `pulse` - Periodic pulses

### Example Configs

See [`configs/`](configs/) directory:
- `basic_config.yaml` - Simple getting started config
- `ml_config.yaml` - ML-powered configuration
- `network_security.yaml` - Network security monitoring
- `fraud_detection.yaml` - Fraud detection setup
- `high_performance.yaml` - High-throughput config
- `composite_perception.yaml` - Multi-faceted analysis

Full documentation: [`configs/README.md`](configs/README.md)

---

## Typical Workflows

### 1. Development and Testing

```bash
# Test different configurations
falcon-ai run --config configs/basic_config.yaml --length 100
falcon-ai run --config configs/ml_config.yaml --length 100
falcon-ai run --config configs/composite_perception.yaml --length 100

# Visualize results
falcon-ai serve --config configs/ml_config.yaml
```

### 2. Benchmarking

```bash
# Run comprehensive benchmarks
falcon-ai benchmark --config configs/wow.yaml --repeats 5

# Compare swarm vs solo
falcon-ai swarm-demo --config configs/network_security.yaml --agents 5
```

### 3. Production Deployment

```bash
# Train and save model
falcon-ai run --config configs/network_security.yaml \
    --scenario attack \
    --length 5000 \
    --checkpoint production/network_falcon

# Model is saved to:
# - production/network_falcon.falcon (model state)
# - production/network_falcon_metadata.json (metadata)
```

### 4. Presentations and Demos

```bash
# Full "wow" demo with dashboard
falcon-ai wow --config configs/wow.yaml --port 8000

# Then open http://127.0.0.1:8000 in browser
```

### 5. Custom Scenario Analysis

```bash
# Run specific scenario
falcon-ai run --config configs/network_security.yaml \
    --scenario attack \
    --length 2000 \
    --seed 789 \
    --metrics results/attack_analysis.json

# Analyze with dashboard
falcon-ai serve --config configs/network_security.yaml \
    --scenario attack \
    --length 2000
```

---

## Tips and Best Practices

### Performance Optimization

**For speed:**
- Use `heuristic` decision core
- Use `simple` energy manager
- Disable monitoring: `monitoring: false`
- Use smaller memory: `max_size: 100`

**For accuracy:**
- Use `ml_decision` or `memory_aware` decision cores
- Use `bayesian` or `reinforcement` correction
- Increase memory size
- Use `composite` perception

### Memory Management

**Small memory** (100-500):
- Fast, low overhead
- Good for high-throughput scenarios
- May forget older patterns

**Large memory** (1000-5000):
- Better long-term learning
- Good for complex scenarios
- Higher memory usage

### Scenario Selection

**spike** - Good for:
- Alert systems
- Threshold tuning
- Quick tests

**drift** - Good for:
- Change detection
- Adaptive systems
- Long-term monitoring

**attack** - Good for:
- Security systems
- Anomaly detection
- Network monitoring

**pulse** - Good for:
- Periodic pattern detection
- Throughput testing
- Regular event streams

---

## Troubleshooting

### Command not found
```bash
# Reinstall package
pip install -e .

# Or use python module syntax
python -m falcon.cli run --config configs/basic_config.yaml
```

### Import errors
```bash
# Install dependencies
pip install -r requirements.txt
```

### Dashboard won't start
```bash
# Check if uvicorn is installed
pip install uvicorn fastapi

# Check port availability
falcon-ai serve --port 8080  # Try different port
```

### Config file errors
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# Use JSON instead
falcon-ai run --config configs/my_config.json
```

---

## Advanced Usage

### Programmatic Access

You can also use FALCON-AI programmatically:

```python
from falcon.config import load_config, normalize_config, build_falcon
from falcon.sim.scenarios import ScenarioRegistry
from falcon.sim.evaluation import run_simulation

# Load config
config_dict = load_config("configs/ml_config.yaml")
app_cfg = normalize_config(config_dict)

# Build FALCON
falcon = build_falcon(app_cfg.falcon)

# Run scenario
scenario = ScenarioRegistry.get(
    app_cfg.scenario.name,
    app_cfg.scenario.length,
    app_cfg.scenario.seed
)
result = run_simulation(falcon, scenario)

# Access metrics
print(result.metrics.to_dict())
```

### Batch Processing

```bash
# Run multiple configs
for config in configs/*.yaml; do
    echo "Running $config..."
    falcon-ai run --config "$config" --metrics "results/$(basename $config .yaml).json"
done
```

### Continuous Monitoring

```bash
# Keep dashboard running
while true; do
    falcon-ai serve --config configs/network_security.yaml --port 8000
    sleep 5
done
```

---

## Next Steps

1. **Try the examples:** Run all configs in `configs/` directory
2. **Launch dashboard:** `falcon-ai serve --config configs/wow.yaml`
3. **Run benchmarks:** `falcon-ai benchmark --config configs/wow.yaml`
4. **Create custom config:** Copy and modify an existing config
5. **Read docs:** Check out [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Support

For issues, questions, or contributions:
- GitHub: [FALCON-AI](https://github.com/oluwafemidiakhoa/FalconAI)
- Documentation: See repository docs
- Examples: Check `examples/` directory

---

**FALCON-AI: Hunt for relevance. Decide fast. Learn continuously.** 🦅
