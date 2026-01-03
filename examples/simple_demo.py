"""
Simple demonstration of FALCON-AI.

This example shows the basic falcon hunting loop:
1. Perceive incoming data
2. Make fast decisions on salient events
3. Observe outcomes and learn
"""

import random
import time
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


def simulate_data_stream(count: int = 50):
    """Simulate a stream of data values"""
    for i in range(count):
        # Generate data with occasional spikes
        if random.random() < 0.1:
            # 10% chance of high value (prey spotted!)
            value = random.uniform(0.8, 1.0)
        else:
            # Normal low values
            value = random.uniform(0.0, 0.5)

        yield value


def simulate_outcome(decision, value: float) -> Outcome:
    """
    Simulate the outcome of taking an action.

    In a real system, this would be the actual result of the action.
    """
    # Higher values generally lead to better outcomes
    success_probability = value * decision.confidence

    if random.random() < success_probability:
        return Outcome(
            outcome_type=OutcomeType.SUCCESS,
            reward=value,
            metadata={'actual_value': value}
        )
    else:
        return Outcome(
            outcome_type=OutcomeType.FAILURE,
            reward=-0.3,
            metadata={'actual_value': value}
        )


def main():
    print("FALCON-AI Simple Demo")
    print("=" * 60)
    print("Simulating a falcon hunting for prey in a data stream...\n")

    # Initialize FALCON-AI
    falcon = FalconAI(
        perception=ThresholdPerception(threshold=0.6),
        decision=HeuristicDecision(),
        correction=OutcomeBasedCorrection(learning_rate=0.1),
        energy_manager=SimpleEnergyManager(max_operations=5000),
        memory=ExperienceMemory()
    )

    print("System initialized with:")
    print("  - Threshold Perception (threshold=0.6)")
    print("  - Heuristic Decision Making")
    print("  - Outcome-Based Correction")
    print("  - Energy Budget: 5000 operations")
    print("\nStarting hunt...\n")

    actions_taken = 0
    successful_actions = 0

    # Process data stream
    for i, value in enumerate(simulate_data_stream(50)):
        # Process through FALCON
        decision = falcon.process(value)

        if decision is not None:
            actions_taken += 1

            print(f"Event #{i}: Value={value:.3f}")
            print(f"  > Decision: {decision.action.value}")
            print(f"  > Confidence: {decision.confidence:.2f}")
            print(f"  > Reasoning: {decision.reasoning}")

            # Execute action and observe outcome
            outcome = simulate_outcome(decision, value)
            falcon.observe(decision, outcome)

            if outcome.is_successful():
                successful_actions += 1
                print(f"  [+] Outcome: SUCCESS (reward={outcome.reward:.2f})")
            else:
                print(f"  [-] Outcome: FAILURE (reward={outcome.reward:.2f})")

            print()

            time.sleep(0.1)  # Slow down for readability

    # Print summary
    print("\n" + "=" * 60)
    print("Hunt Complete!")
    print("=" * 60)

    status = falcon.get_status()

    print(f"\nPerformance Summary:")
    print(f"  Events processed: {status['perception']['events_processed']}")
    print(f"  Events triggered: {status['perception']['events_triggered']}")
    print(f"  Trigger rate: {status['perception']['trigger_rate']:.1%}")
    print(f"  Actions taken: {actions_taken}")
    print(f"  Successful actions: {successful_actions}")
    print(f"  Success rate: {successful_actions/max(actions_taken, 1):.1%}")
    print(f"  Average confidence: {status['decision']['average_confidence']:.2f}")
    print(f"  Average reward: {status['correction']['average_reward']:.2f}")

    print(f"\nEnergy Budget:")
    print(f"  Used: {status['energy']['used_operations']}/{status['energy']['max_operations']} operations")
    print(f"  Remaining: {status['energy']['remaining_fraction']:.1%}")

    print(f"\nMemory:")
    print(f"  Experiences stored: {status['memory']['size']}")

    print("\n[OK] Demo complete!")


if __name__ == "__main__":
    main()
