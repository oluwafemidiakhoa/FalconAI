# 🦅 FALCON-AI Quick Start

## ⚡ One-Liner Demos

```bash
# 🌟 WOW the world! (Full demo with dashboard)
falcon-ai wow --config configs/wow.yaml --port 8002

# 🚀 Launch dashboard only
falcon-ai serve --config configs/wow.yaml --port 8002

# 📊 Run benchmarks
falcon-ai benchmark --config configs/wow.yaml --repeats 5

# 🤖 Test swarm intelligence
falcon-ai swarm-demo --agents 10 --scenario attack

# ⚡ Run a scenario
falcon-ai run --config configs/basic_config.yaml
```

---

## 📁 6 Ready-to-Use Configs

| Config | Best For | Key Features |
|--------|----------|--------------|
| `basic_config.yaml` | Learning & testing | Simple threshold, fast |
| `ml_config.yaml` | High accuracy | Neural nets, Random Forest |
| `network_security.yaml` | Security monitoring | Anomaly detection, Q-learning |
| `fraud_detection.yaml` | Fraud detection | Change detection, large memory |
| `high_performance.yaml` | Max throughput | Minimal overhead, fast heuristics |
| `composite_perception.yaml` | Complex analysis | Multi-engine perception |

---

## 🔧 Common Commands

### Run a Scenario
```bash
# Basic
falcon-ai run --config configs/basic_config.yaml

# With custom parameters
falcon-ai run --config configs/ml_config.yaml \
  --scenario attack \
  --length 2000 \
  --metrics output/metrics.json \
  --checkpoint models/my_falcon
```

### Launch Dashboard
```bash
# Auto-opens browser
falcon-ai serve --config configs/wow.yaml --port 8002

# Without browser
falcon-ai serve --config configs/wow.yaml --port 8002 --no-browser

# Different scenarios
falcon-ai serve --config configs/ml_config.yaml --scenario attack --port 8003
```

### Run Benchmarks
```bash
# Full matrix
falcon-ai benchmark --config configs/wow.yaml

# Custom scenarios
falcon-ai benchmark \
  --scenarios spike,attack,drift \
  --decisions heuristic,ml_decision \
  --repeats 10
```

### Swarm Demo
```bash
# Basic
falcon-ai swarm-demo --agents 5

# Custom
falcon-ai swarm-demo \
  --config configs/network_security.yaml \
  --agents 10 \
  --scenario attack \
  --length 2000
```

---

## 🎯 Typical Workflows

### Testing & Development
```bash
# 1. Quick test
falcon-ai run --config configs/basic_config.yaml --length 100

# 2. Visualize
falcon-ai serve --config configs/basic_config.yaml --port 8002

# 3. Benchmark
falcon-ai benchmark --scenarios spike --repeats 3
```

### Production Deployment
```bash
# 1. Train and save model
falcon-ai run --config configs/network_security.yaml \
  --scenario attack \
  --length 5000 \
  --checkpoint production/network_falcon

# 2. Run live monitoring
falcon-ai serve --config configs/network_security.yaml \
  --mode swarm \
  --port 8080
```

### Presentation/Demo
```bash
# The WOW command does it all!
falcon-ai wow --config configs/wow.yaml --port 8002

# Opens browser automatically with:
# - Benchmark results
# - Swarm comparison
# - Live dashboard
# - Real-time visualization
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use a different port
falcon-ai serve --port 8002  # or 8003, 9000, etc.
```

### Command Not Found
```bash
# Reinstall
pip install -e .

# Or use Python module
python -m falcon.cli wow --config configs/wow.yaml --port 8002
```

### Config Errors
See [FIXES.md](FIXES.md) for correct parameter formats.

---

## 📚 More Info

- **Full CLI Guide:** [CLI_GUIDE.md](CLI_GUIDE.md)
- **Config Details:** [configs/README.md](configs/README.md)
- **All Fixes:** [FIXES.md](FIXES.md)
- **Test Results:** [TESTING_SUMMARY.md](TESTING_SUMMARY.md)
- **Main README:** [README.md](README.md)

---

## 💡 Pro Tips

1. **Always specify port** for serve/wow: `--port 8002`
2. **Use --no-browser** for headless servers
3. **Save checkpoints** with `--checkpoint` flag
4. **Export metrics** with `--metrics` flag
5. **Test with small lengths** first: `--length 100`

---

## 🚀 Get Started Now!

```bash
# Copy and paste this:
falcon-ai wow --config configs/wow.yaml --port 8002
```

**Sit back and watch FALCON-AI in action!** 🦅

The browser will open automatically showing:
- ✅ Real-time event processing
- ✅ Swarm intelligence coordination
- ✅ Performance benchmarks
- ✅ Beautiful visualizations

**Hunt for relevance. Decide fast. Learn continuously.**
