"""
Anomaly Detection Example

Demonstrates FALCON-AI using statistical anomaly detection
to identify outliers in a data stream.
"""

import random
import numpy as np
from falcon import (
    FalconAI,
    AnomalyPerception,
    RuleBasedDecision,
    ReinforcementCorrection,
    MultiTierEnergyManager,
    ComputeBudget,
    InstinctMemory,
    ExperienceMemory,
    Outcome,
    OutcomeType,
    ActionType
)


def generate_data_with_anomalies(n_samples: int = 200):
    """
    Generate a dataset with injected anomalies.

    Returns tuples of (value, is_anomaly)
    """
    data = []

    for i in range(n_samples):
        if random.random() < 0.05:  # 5% anomalies
            # Inject anomaly
            value = random.gauss(100, 20)  # Far from normal
            is_anomaly = True
        else:
            # Normal data
            value = random.gauss(50, 10)
            is_anomaly = False

        data.append((value, is_anomaly))

    return data


def main():
    print("🦅 FALCON-AI Anomaly Detection Demo")
    print("=" * 60)
    print("Detecting statistical anomalies in a data stream...\n")

    # Set up decision rules
    decision_engine = RuleBasedDecision()

    # Add rules based on event characteristics
    decision_engine.add_rule(
        condition=lambda e: hasattr(e, 'salience_score') and e.salience_score > 0.9,
        action=ActionType.ESCALATE,
        reasoning="Very high salience - escalate immediately"
    )

    decision_engine.add_rule(
        condition=lambda e: hasattr(e, 'salience_score') and e.salience_score > 0.7,
        action=ActionType.ALERT,
        reasoning="High salience - raise alert"
    )

    decision_engine.add_rule(
        condition=lambda e: hasattr(e, 'salience_score') and e.salience_score > 0.5,
        action=ActionType.OBSERVE,
        reasoning="Moderate salience - continue observing"
    )

    # Initialize FALCON
    budget = ComputeBudget(
        max_operations=20000,
        max_latency_ms=10000.0,
        max_energy_units=2000.0
    )

    falcon = FalconAI(
        perception=AnomalyPerception(
            z_threshold=2.0,  # 2 standard deviations
            min_samples=20
        ),
        decision=decision_engine,
        correction=ReinforcementCorrection(learning_rate=0.1),
        energy_manager=MultiTierEnergyManager(budget),
        memory=ExperienceMemory()
    )

    print("System configured:")
    print("  - Statistical Anomaly Detection (z-score threshold=2.0)")
    print("  - Rule-Based Decision Engine")
    print("  - Reinforcement Learning Correction")
    print("  - Multi-Tier Energy Management\n")

    # Generate test data
    data = generate_data_with_anomalies(200)

    detected_anomalies = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("Processing data stream...")
    print("-" * 60)

    for i, (value, is_actual_anomaly) in enumerate(data):
        decision = falcon.process(value, context={'urgency': 'high'})

        if decision is not None:
            detected_anomalies += 1

            # Check if detection was correct
            if is_actual_anomaly:
                true_positives += 1
                outcome = Outcome(OutcomeType.SUCCESS, reward=1.0)
            else:
                false_positives += 1
                outcome = Outcome(OutcomeType.FAILURE, reward=-0.3)

            falcon.observe(decision, outcome)

            # Print detection
            result = "✓ TRUE" if is_actual_anomaly else "✗ FALSE"
            print(f"Sample {i:3d}: value={value:6.1f} | {decision.action.value:10s} | {result} POSITIVE")

        elif is_actual_anomaly:
            # Missed an anomaly
            false_negatives += 1

    # Calculate metrics
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)

    total_actual_anomalies = sum(1 for _, is_anomaly in data if is_anomaly)

    print(f"\nGround Truth:")
    print(f"  Total samples: {len(data)}")
    print(f"  Actual anomalies: {total_actual_anomalies}")
    print(f"  Normal samples: {len(data) - total_actual_anomalies}")

    print(f"\nDetection Performance:")
    print(f"  Detected anomalies: {detected_anomalies}")
    print(f"  True positives: {true_positives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"  Precision: {precision:.1%}")

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"  Recall: {recall:.1%}")

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"  F1 Score: {f1:.3f}")

    # System stats
    status = falcon.get_status()

    print(f"\nSystem Performance:")
    print(f"  Average confidence: {status['decision']['average_confidence']:.2f}")
    print(f"  Average reward: {status['correction']['average_reward']:.2f}")
    print(f"  Energy remaining: {status['energy']['remaining_fraction']:.1%}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
