# FALCON-AI CLI Fixes Applied

## Issues Found and Fixed

### Issue 1: Port 8000 Already in Use
**Error:**
```
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)
```

**Solution:**
Use a different port with `--port` flag:
```bash
falcon-ai serve --config configs/wow.yaml --port 8001
falcon-ai wow --config configs/wow.yaml --port 8001
```

---

### Issue 2: OnlineNeuralPerception Parameter Mismatch
**Error:**
```
TypeError: OnlineNeuralPerception.__init__() got an unexpected keyword argument 'hidden_layers'
```

**Root Cause:**
`configs/ml_config.yaml` had incorrect parameters for `OnlineNeuralPerception`.

**Fixed Parameters:**
```yaml
# BEFORE (WRONG):
perception:
  type: online_neural
  params:
    input_dim: 5
    hidden_layers: [64, 32]    # WRONG - not a valid parameter
    learning_rate: 0.001       # WRONG - not a valid parameter

# AFTER (CORRECT):
perception:
  type: online_neural
  params:
    input_dim: 5
    window_size: 100           # CORRECT
    adaptation_rate: 0.1       # CORRECT
```

**Actual OnlineNeuralPerception Parameters:**
- `input_dim` - Dimension of input vectors (default: 10)
- `window_size` - Size of sliding window for statistics (default: 100)
- `adaptation_rate` - How quickly to adapt threshold (default: 0.1)

---

### Issue 3: AnomalyPerception Parameter Mismatch
**Error:**
```
TypeError: AnomalyPerception.__init__() got an unexpected keyword argument 'window_size'
```

**Root Cause:**
`configs/network_security.yaml` had incorrect parameters for `AnomalyPerception`.

**Fixed Parameters:**
```yaml
# BEFORE (WRONG):
perception:
  type: anomaly
  params:
    window_size: 20      # WRONG - not a valid parameter
    sensitivity: 2.5     # WRONG - should be 'z_threshold'
    percentile: 95       # WRONG - not a valid parameter

# AFTER (CORRECT):
perception:
  type: anomaly
  params:
    z_threshold: 2.5     # CORRECT - z-score threshold
    min_samples: 20      # CORRECT - minimum samples needed
```

**Actual AnomalyPerception Parameters:**
- `z_threshold` - Z-score threshold for anomaly detection (default: 2.5)
- `min_samples` - Minimum samples before triggering (default: 20)
- `value_extractor` - Optional function to extract numeric value

---

### Issue 4: Composite Perception Config
**Fixed:**
```yaml
# configs/composite_perception.yaml
perception:
  type: composite
  params:
    engines:
      - type: threshold
        params:
          threshold: 0.6
      - type: anomaly
        params:
          z_threshold: 2.0      # FIXED
          min_samples: 15       # FIXED
      - type: change_detection
        params:
          change_threshold: 0.3
          window_size: 10
```

---

### Issue 5: ReinforcementCorrection Parameter Error
**Error:**
```
TypeError: ReinforcementCorrection.__init__() got an unexpected keyword argument 'exploration_rate'
```

**Root Cause:**
`configs/network_security.yaml` had incorrect parameter for `ReinforcementCorrection`.

**Fixed Parameters:**
```yaml
# BEFORE (WRONG):
correction:
  type: reinforcement
  params:
    learning_rate: 0.05
    discount_factor: 0.95
    exploration_rate: 0.15    # WRONG - not a valid parameter

# AFTER (CORRECT):
correction:
  type: reinforcement
  params:
    learning_rate: 0.05
    discount_factor: 0.95     # Only these two parameters
```

**Actual ReinforcementCorrection Parameters:**
- `learning_rate` - Learning rate for Q-value updates (default: 0.1)
- `discount_factor` - Discount factor for future rewards (default: 0.9)

---

## ✅ All Configs Now Working

All 6 configuration files have been fixed:

1. **basic_config.yaml** - ✅ Works (no changes needed)
2. **ml_config.yaml** - ✅ Fixed (OnlineNeuralPerception params)
3. **network_security.yaml** - ✅ Fixed (AnomalyPerception params)
4. **fraud_detection.yaml** - ✅ Works (no changes needed)
5. **high_performance.yaml** - ✅ Works (no changes needed)
6. **composite_perception.yaml** - ✅ Fixed (AnomalyPerception params)

---

## 🚀 Working Commands

### All These Now Work:

```bash
# WOW command on port 8001
falcon-ai wow --config configs/wow.yaml --port 8001

# Serve different configs
falcon-ai serve --config configs/basic_config.yaml --port 8001
falcon-ai serve --config configs/ml_config.yaml --port 8001
falcon-ai serve --config configs/network_security.yaml --port 8001
falcon-ai serve --config configs/fraud_detection.yaml --port 8001
falcon-ai serve --config configs/high_performance.yaml --port 8001
falcon-ai serve --config configs/composite_perception.yaml --port 8001

# Run scenarios
falcon-ai run --config configs/basic_config.yaml
falcon-ai run --config configs/network_security.yaml --scenario attack --length 2000

# Benchmarks
falcon-ai benchmark --config configs/wow.yaml --repeats 5

# Swarm demos
falcon-ai swarm-demo --agents 10 --scenario attack
```

---

## 📝 Reference: Correct Component Parameters

### Perception Engines

#### ThresholdPerception
```yaml
type: threshold
params:
  threshold: 0.7
```

#### ChangeDetectionPerception
```yaml
type: change_detection
params:
  change_threshold: 0.3
  window_size: 10
```

#### AnomalyPerception
```yaml
type: anomaly
params:
  z_threshold: 2.5       # Z-score threshold
  min_samples: 20        # Minimum samples
```

#### NeuralPerception (NOT OnlineNeuralPerception!)
```yaml
type: neural
params:
  input_dim: 10
  hidden_layers: [64, 32]    # Only for NeuralPerception
  salience_threshold: 0.7
```

#### OnlineNeuralPerception
```yaml
type: online_neural
params:
  input_dim: 10
  window_size: 100           # NOT hidden_layers
  adaptation_rate: 0.1       # NOT learning_rate
```

---

## 🔍 How to Check Valid Parameters

To see what parameters a component accepts:

```python
# In Python
from falcon.perception import AnomalyPerception
help(AnomalyPerception.__init__)

from falcon.ml import OnlineNeuralPerception
help(OnlineNeuralPerception.__init__)
```

Or read the source code:
- Perception: `falcon/perception/filters.py`
- ML Perception: `falcon/ml/neural_perception.py`
- Decisions: `falcon/decision/policies.py`
- ML Decisions: `falcon/ml/ml_decision.py`

---

## ✅ Summary

**All Issues Resolved:**
- ✅ Port conflict → Use `--port 8001`
- ✅ OnlineNeuralPerception params → Fixed in ml_config.yaml
- ✅ AnomalyPerception params → Fixed in network_security.yaml & composite_perception.yaml
- ✅ All configs validated and working

**Ready to Demo:**
```bash
falcon-ai wow --config configs/wow.yaml --port 8001
```

This will:
1. Run comprehensive benchmarks
2. Run swarm vs solo comparison
3. Launch dashboard on port 8001
4. Auto-open browser

**Enjoy your working FALCON-AI CLI!** 🦅
