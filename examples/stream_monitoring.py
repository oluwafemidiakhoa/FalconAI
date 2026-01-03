"""
Stream Monitoring Example

Demonstrates FALCON-AI monitoring a data stream for anomalies.
Uses change detection to identify when metrics shift significantly.
"""

import random
import time
import math
from falcon import (
    FalconAI,
    ChangeDetectionPerception,
    ThresholdDecision,
    BayesianCorrection,
    AdaptiveEnergyManager,
    ComputeBudget,
    ExperienceMemory,
    Outcome,
    OutcomeType
)


class MetricStream:
    """Simulates a metric stream with normal operation and anomalies"""

    def __init__(self):
        self.baseline = 50.0
        self.time = 0
        self.anomaly_active = False

    def next_value(self):
        """Generate next metric value"""
        self.time += 1

        # Inject anomalies periodically
        if self.time % 30 == 0:
            self.anomaly_active = True
        elif self.time % 30 == 10:
            self.anomaly_active = False

        # Generate value
        if self.anomaly_active:
            # Anomaly: shifted mean with more variance
            value = random.gauss(80.0, 15.0)
        else:
            # Normal: stable around baseline
            value = random.gauss(self.baseline, 5.0)

        # Add some periodic component
        value += 10 * math.sin(self.time * 0.1)

        return value


def handle_alert(value: float, decision) -> Outcome:
    """
    Simulate handling an alert.

    In a real system, this might:
    - Send notifications
    - Trigger auto-remediation
    - Escalate to on-call team
    """
    # For demo purposes, we succeed if we caught a real anomaly
    is_actual_anomaly = value > 70  # Simple threshold for "real" anomaly

    if is_actual_anomaly:
        return Outcome(
            outcome_type=OutcomeType.SUCCESS,
            reward=1.0,
            metadata={'value': value, 'true_positive': True}
        )
    else:
        # False alarm
        return Outcome(
            outcome_type=OutcomeType.FAILURE,
            reward=-0.5,
            metadata={'value': value, 'false_positive': True}
        )


def main():
    print("🦅 FALCON-AI Stream Monitoring Demo")
    print("=" * 60)
    print("Monitoring a metric stream for anomalies...\n")

    # Initialize FALCON with change detection
    budget = ComputeBudget(
        max_operations=10000,
        max_latency_ms=5000.0,
        max_energy_units=1000.0
    )

    falcon = FalconAI(
        perception=ChangeDetectionPerception(
            change_threshold=0.3,
            window_size=10
        ),
        decision=ThresholdDecision(
            alert_threshold=0.5,
            intervene_threshold=0.7,
            escalate_threshold=0.9
        ),
        correction=BayesianCorrection(),
        energy_manager=AdaptiveEnergyManager(budget),
        memory=ExperienceMemory()
    )

    print("System configured:")
    print("  - Change Detection Perception (30% change threshold)")
    print("  - Threshold-Based Decisions")
    print("  - Bayesian Correction Loop")
    print("  - Adaptive Energy Management\n")

    stream = MetricStream()
    alerts_triggered = 0
    true_positives = 0
    false_positives = 0

    print("Time | Value  | Status | Action")
    print("-" * 60)

    # Monitor stream
    for i in range(100):
        value = stream.next_value()

        # Process through FALCON
        decision = falcon.process(value)

        status = "NORMAL"
        action = "-"

        if decision is not None:
            alerts_triggered += 1
            status = "ALERT"
            action = decision.action.value

            # Handle the alert
            outcome = handle_alert(value, decision)
            falcon.observe(decision, outcome)

            if outcome.is_successful():
                true_positives += 1
            else:
                false_positives += 1

        # Print monitoring output
        if i % 5 == 0 or decision is not None:
            print(f"{i:4d} | {value:6.1f} | {status:6s} | {action}")

        time.sleep(0.05)

    # Summary
    print("\n" + "=" * 60)
    print("Monitoring Complete!")
    print("=" * 60)

    status = falcon.get_status()

    print(f"\nDetection Performance:")
    print(f"  Alerts triggered: {alerts_triggered}")
    print(f"  True positives: {true_positives}")
    print(f"  False positives: {false_positives}")
    if alerts_triggered > 0:
        precision = true_positives / alerts_triggered
        print(f"  Precision: {precision:.1%}")

    print(f"\nSystem Statistics:")
    print(f"  Events processed: {status['perception']['events_processed']}")
    print(f"  Trigger rate: {status['perception']['trigger_rate']:.1%}")
    print(f"  Average confidence: {status['decision']['average_confidence']:.2f}")

    print(f"\nBayesian Learning:")
    correction_signals = status['correction']['correction_signals']
    if 'success_rates' in correction_signals:
        print("  Success rates by action:")
        for action, rate in correction_signals['success_rates'].items():
            print(f"    {action}: {rate:.1%}")

    print(f"\nEnergy Usage:")
    print(f"  Budget remaining: {status['energy']['remaining_fraction']:.1%}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
