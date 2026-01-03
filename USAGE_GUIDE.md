# FALCON-AI Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Layer-by-Layer Guide](#layer-by-layer-guide)
4. [Common Patterns](#common-patterns)
5. [Customization](#customization)
6. [Best Practices](#best-practices)

## Installation

```bash
# Clone or download FALCON-AI
cd fALCON

# Install dependencies
pip install -r requirements.txt

# Install FALCON-AI
pip install -e .
```

## Quick Start

### Minimal Example

```python
from falcon import (
    FalconAI,
    ThresholdPerception,
    HeuristicDecision,
    OutcomeBasedCorrection,
    SimpleEnergyManager,
    ExperienceMemory,
    Outcome,
    OutcomeType
)

# Initialize FALCON
falcon = FalconAI(
    perception=ThresholdPerception(threshold=0.7),
    decision=HeuristicDecision(),
    correction=OutcomeBasedCorrection(learning_rate=0.1),
    energy_manager=SimpleEnergyManager(max_operations=1000),
    memory=ExperienceMemory()
)

# Process data
for data in data_stream:
    decision = falcon.process(data)

    if decision:
        # Execute action
        result = execute_action(decision)

        # Provide feedback
        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS if result else OutcomeType.FAILURE,
            reward=1.0 if result else -0.5
        )
        falcon.observe(decision, outcome)

# Check status
print(falcon.get_status())
```

## Layer-by-Layer Guide

### Layer 1: Perception

#### Threshold Perception
Use when you have a simple threshold for actionable events:

```python
from falcon import ThresholdPerception

# Trigger on values > 0.7
perception = ThresholdPerception(
    threshold=0.7,
    value_extractor=lambda x: float(x)  # Optional: extract value from data
)
```

#### Change Detection
Use when you want to detect significant changes:

```python
from falcon import ChangeDetectionPerception

perception = ChangeDetectionPerception(
    change_threshold=0.3,  # 30% change triggers
    window_size=10,         # Compare to last 10 values
    value_extractor=lambda x: x['metric']
)
```

#### Anomaly Detection
Use for statistical anomaly detection:

```python
from falcon import AnomalyPerception

perception = AnomalyPerception(
    z_threshold=2.5,    # 2.5 standard deviations
    min_samples=20,     # Need 20 samples before detecting
    value_extractor=lambda x: x['value']
)
```

#### Composite Perception
Combine multiple perception engines:

```python
from falcon import CompositePerception, ThresholdPerception, AnomalyPerception

perception = CompositePerception([
    ThresholdPerception(threshold=0.8),
    AnomalyPerception(z_threshold=3.0)
])
# Triggers if ANY engine detects an event
```

### Layer 2: Decision

#### Heuristic Decision
Fast decision making based on event type:

```python
from falcon import HeuristicDecision

decision = HeuristicDecision(confidence_boost=0.1)
```

#### Threshold Decision
Multi-level threshold-based decisions:

```python
from falcon import ThresholdDecision

decision = ThresholdDecision(
    alert_threshold=0.5,      # Alert at 0.5 salience
    intervene_threshold=0.7,  # Intervene at 0.7
    escalate_threshold=0.9    # Escalate at 0.9
)
```

#### Rule-Based Decision
Custom decision rules:

```python
from falcon import RuleBasedDecision, ActionType

decision = RuleBasedDecision()

# Add custom rules
decision.add_rule(
    condition=lambda e: e.salience_score > 0.95,
    action=ActionType.ESCALATE,
    reasoning="Critically high salience"
)

decision.add_rule(
    condition=lambda e: e.event_type.value == 'anomaly',
    action=ActionType.ALERT,
    reasoning="Anomaly detected"
)
```

### Layer 3: Correction

#### Outcome-Based Correction
Simple running average updates:

```python
from falcon import OutcomeBasedCorrection

correction = OutcomeBasedCorrection(
    learning_rate=0.1,        # How fast to learn
    abort_threshold=-0.5      # When to abort strategy
)
```

#### Bayesian Correction
Probabilistic updates:

```python
from falcon import BayesianCorrection

correction = BayesianCorrection(
    prior_alpha=1.0,  # Prior successes
    prior_beta=1.0    # Prior failures
)

# Get success probability for an action
mean, variance = correction.get_action_confidence('alert')
```

#### Reinforcement Correction
Q-learning based correction:

```python
from falcon import ReinforcementCorrection

correction = ReinforcementCorrection(
    learning_rate=0.1,
    discount_factor=0.9
)
```

### Layer 4: Energy Management

#### Simple Energy Manager
Budget-based mode selection:

```python
from falcon import SimpleEnergyManager

energy_manager = SimpleEnergyManager(
    max_operations=10000,
    max_latency_ms=1000.0,
    max_energy_units=500.0
)
```

#### Adaptive Energy Manager
Learns optimal modes:

```python
from falcon import AdaptiveEnergyManager, ComputeBudget

budget = ComputeBudget(
    max_operations=10000,
    max_latency_ms=1000.0,
    max_energy_units=500.0
)

energy_manager = AdaptiveEnergyManager(
    budget=budget,
    adaptation_rate=0.1
)

# Update mode performance
energy_manager.update_mode_performance(mode, performance_score)
```

#### Multi-Tier Energy Manager
Urgency-based selection:

```python
from falcon import MultiTierEnergyManager

energy_manager = MultiTierEnergyManager(budget)

# Use with urgency context
decision = falcon.process(data, context={'urgency': 'critical'})
```

### Layer 5: Memory

#### Experience Memory
Learns from outcomes:

```python
from falcon import ExperienceMemory, ExperienceEntry

memory = ExperienceMemory(max_size=1000)

# Store experience
entry = ExperienceEntry(
    situation='high_load',
    action='scale_up',
    outcome='success',
    reward=1.0
)
memory.store('high_load', entry)

# Query
success_rate = memory.get_success_rate('high_load')
best_action = memory.get_best_action('high_load')
```

#### Instinct Memory
Pretrained patterns:

```python
from falcon import InstinctMemory

memory = InstinctMemory()

# Add custom patterns
memory.store('critical_pattern', {
    'response': 'escalate',
    'confidence': 0.9
})
```

## Common Patterns

### Pattern 1: Stream Monitoring

```python
falcon = FalconAI(
    perception=ChangeDetectionPerception(change_threshold=0.3),
    decision=ThresholdDecision(),
    correction=BayesianCorrection(),
    energy_manager=SimpleEnergyManager(),
    memory=ExperienceMemory()
)

for metric_value in metric_stream:
    decision = falcon.process(metric_value)

    if decision and decision.action == ActionType.ALERT:
        send_alert(metric_value, decision)
        outcome = check_if_needed(metric_value)
        falcon.observe(decision, outcome)
```

### Pattern 2: Anomaly Detection

```python
falcon = FalconAI(
    perception=AnomalyPerception(z_threshold=2.5),
    decision=RuleBasedDecision(),
    correction=OutcomeBasedCorrection(),
    energy_manager=AdaptiveEnergyManager(budget),
    memory=ExperienceMemory()
)
```

### Pattern 3: Resource-Constrained

```python
# Minimize resource usage
falcon = FalconAI(
    perception=ThresholdPerception(threshold=0.9),  # Very selective
    decision=HeuristicDecision(),                   # Fast decisions
    correction=OutcomeBasedCorrection(),            # Lightweight learning
    energy_manager=SimpleEnergyManager(
        max_operations=1000  # Tight budget
    ),
    memory=None  # No memory overhead
)
```

## Customization

### Custom Perception Engine

```python
from falcon.perception.base import PerceptionEngine, Event, EventType

class MyPerception(PerceptionEngine):
    def perceive(self, data):
        self.events_processed += 1

        # Your custom logic
        if self.is_interesting(data):
            self.events_triggered += 1
            return Event(
                data=data,
                event_type=EventType.SALIENT,
                salience_score=self.compute_salience(data)
            )

        return None
```

### Custom Decision Core

```python
from falcon.decision.base import DecisionCore, Decision, ActionType

class MyDecision(DecisionCore):
    def decide(self, event, context=None):
        self.decisions_made += 1

        # Your custom logic
        action = self.determine_action(event)
        confidence = self.compute_confidence(event)

        self.total_confidence += confidence

        return Decision(
            action=action,
            confidence=confidence,
            reasoning="My custom reasoning"
        )
```

## Best Practices

### 1. Start Conservative
```python
# Begin with high thresholds
perception = ThresholdPerception(threshold=0.8)  # Start high
correction = OutcomeBasedCorrection(learning_rate=0.05)  # Learn slowly
```

### 2. Monitor Performance
```python
# Regularly check system status
status = falcon.get_status()

if status['perception']['trigger_rate'] > 0.5:
    print("Warning: Triggering too often, increase threshold")

if status['energy']['remaining_fraction'] < 0.1:
    print("Warning: Low energy budget")
```

### 3. Provide Quality Feedback
```python
# Good: Specific, quantitative feedback
outcome = Outcome(
    outcome_type=OutcomeType.SUCCESS,
    reward=0.85,  # Partial success
    metadata={'metric': actual_value}
)

# Bad: Binary, no context
outcome = Outcome(OutcomeType.SUCCESS, reward=1.0)
```

### 4. Use Context
```python
# Leverage context for better decisions
decision = falcon.process(data, context={
    'urgency': 'high',
    'time_of_day': 'peak',
    'current_load': 0.9
})
```

### 5. Reset Periodically
```python
# Reset budget for new time periods
falcon.energy_manager.reset_budget()

# Or full system reset
falcon.reset()
```

## Running Examples

```bash
# Simple demo
python examples/simple_demo.py

# Stream monitoring
python examples/stream_monitoring.py

# Anomaly detection
python examples/anomaly_detection.py
```
