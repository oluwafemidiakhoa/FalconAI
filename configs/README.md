# FALCON-AI Configuration Files

This directory contains example configuration files for various use cases.

## Available Configurations

### 1. `basic_config.yaml` - Getting Started
**Best for:** Learning FALCON-AI, simple scenarios, quick testing

**Features:**
- Threshold-based perception (simple and fast)
- Heuristic decision making
- Outcome-based learning
- Minimal resource usage

**Usage:**
```bash
falcon-ai run --config configs/basic_config.yaml
```

---

### 2. `ml_config.yaml` - Machine Learning
**Best for:** Complex patterns, high accuracy, adaptive learning

**Features:**
- Online neural network perception
- Random Forest decision making
- Bayesian correction
- Adaptive energy management

**Usage:**
```bash
falcon-ai run --config configs/ml_config.yaml
```

---

### 3. `network_security.yaml` - Network Security Monitoring
**Best for:** DDoS detection, intrusion detection, traffic analysis

**Features:**
- Anomaly detection perception
- Memory-aware decisions (uses shared swarm knowledge)
- Reinforcement learning correction
- Multi-tier energy management

**Usage:**
```bash
falcon-ai run --config configs/network_security.yaml
```

---

### 4. `fraud_detection.yaml` - Fraud Detection
**Best for:** Financial fraud, payment anomalies, unusual patterns

**Features:**
- Change detection perception
- Multi-level threshold decisions
- Outcome-based learning
- Large memory capacity

**Usage:**
```bash
falcon-ai run --config configs/fraud_detection.yaml
```

---

### 5. `high_performance.yaml` - High Throughput
**Best for:** High-volume streams, low-latency, resource-constrained

**Features:**
- Minimal overhead threshold perception
- Fast heuristic decisions
- Monitoring disabled for max speed
- Large operation budget

**Usage:**
```bash
falcon-ai run --config configs/high_performance.yaml
```

---

### 6. `composite_perception.yaml` - Multi-Faceted Analysis
**Best for:** Complex scenarios requiring comprehensive coverage

**Features:**
- Composite perception (threshold + anomaly + change detection)
- Hybrid decision making
- Bayesian learning
- Adaptive energy

**Usage:**
```bash
falcon-ai run --config configs/composite_perception.yaml
```

---

### 7. `wow.yaml` - Production Showcase
**Best for:** Demonstrations, benchmarking, presentations

**Features:**
- Memory-aware decision making
- Swarm-optimized configuration
- Balanced performance and accuracy

**Usage:**
```bash
falcon-ai wow --config configs/wow.yaml
```

---

## CLI Commands

### Run a Single Scenario
```bash
falcon-ai run --config configs/basic_config.yaml
```

### Run with Custom Parameters
```bash
falcon-ai run --config configs/ml_config.yaml --scenario attack --length 2000 --seed 999
```

### Save Metrics and Checkpoint
```bash
falcon-ai run --config configs/network_security.yaml --metrics output/metrics.json --checkpoint output/model
```

### Launch Live Dashboard
```bash
falcon-ai serve --config configs/wow.yaml --port 8000
```

### Run Benchmarks
```bash
falcon-ai benchmark --config configs/basic_config.yaml --scenarios spike,attack,drift --output-dir reports/
```

### Run Swarm Showcase
```bash
falcon-ai swarm-demo --config configs/network_security.yaml --agents 5 --output-dir reports/
```

### WOW Command (Full Demo)
```bash
falcon-ai wow --config configs/wow.yaml --port 8000
```

This runs:
1. Complete benchmark matrix
2. Swarm vs solo comparison
3. Live dashboard with results

---

## Configuration Structure

All configs follow this structure:

```yaml
falcon:
  perception:
    type: <perception_type>
    params:
      # perception-specific parameters

  decision:
    type: <decision_type>
    params:
      # decision-specific parameters

  correction:
    type: <correction_type>
    params:
      # correction-specific parameters

  energy:
    type: <energy_type>
    params:
      # energy-specific parameters

  memory:
    type: <memory_type>
    params:
      # memory-specific parameters

  monitoring: true/false

scenario:
  name: <scenario_name>
  length: <number_of_events>
  seed: <random_seed>

output:
  metrics_path: <path_to_metrics.json>
  checkpoint_path: <path_to_checkpoint>
  report_dir: <report_directory>
```

## Component Types

### Perception
- `threshold` - Simple threshold filtering
- `change_detection` - Detect significant changes
- `anomaly` - Statistical anomaly detection
- `neural` - Neural network perception
- `online_neural` - Adaptive neural perception
- `composite` - Combine multiple engines

### Decision
- `heuristic` - Fast heuristic rules
- `threshold` - Multi-level thresholds
- `rule_based` - Custom rules
- `hybrid` - Heuristics + learned patterns
- `ml_decision` - Random Forest / Gradient Boosting
- `ensemble` - Combine multiple cores
- `memory_aware` - Uses shared swarm memory

### Correction
- `outcome_based` - Running average updates
- `bayesian` - Probabilistic learning
- `reinforcement` - Q-learning

### Energy
- `simple` - Basic budget tracking
- `adaptive` - Learns optimal modes
- `multi_tier` - Urgency-based selection

### Memory
- `experience` - Learned from outcomes
- `instinct` - Pretrained patterns

### Scenarios
- `spike` - Sudden spikes in metrics
- `drift` - Gradual drift over time
- `attack` - Network attack patterns
- `pulse` - Periodic pulses

---

## Creating Custom Configs

1. Copy an existing config:
```bash
cp configs/basic_config.yaml configs/my_config.yaml
```

2. Edit parameters to match your use case

3. Test your config:
```bash
falcon-ai run --config configs/my_config.yaml --length 100
```

4. Benchmark different configurations:
```bash
falcon-ai benchmark --config configs/my_config.yaml
```

---

## Performance Tips

### For High Accuracy
- Use `ml_config.yaml` or `composite_perception.yaml`
- Increase memory size
- Use bayesian or reinforcement correction

### For Low Latency
- Use `high_performance.yaml`
- Disable monitoring
- Use simple energy manager
- Use heuristic decisions

### For Resource-Constrained
- Use `basic_config.yaml`
- Reduce memory size
- Lower operation budgets
- Disable monitoring

### For Complex Scenarios
- Use `composite_perception.yaml`
- Enable memory-aware decisions
- Use adaptive energy
- Increase correction learning rates

---

## Next Steps

1. Try running each config to see how they perform
2. Launch the dashboard to visualize results: `falcon-ai serve`
3. Run benchmarks to compare configurations
4. Create your own custom config for your use case
5. Deploy to production with model persistence

For more information, see the main documentation in the repository root.
