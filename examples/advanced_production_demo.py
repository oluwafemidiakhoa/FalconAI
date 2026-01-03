"""
Advanced Production Demo for FALCON-AI

Demonstrates:
- ML-based perception and decision making
- Model persistence (save/load)
- Multi-agent swarm intelligence
- Production-ready deployment scenario

Scenario: Distributed network monitoring system
Multiple FALCON agents monitor different network segments,
share experiences, and coordinate responses to threats.
"""

import random
import numpy as np
import time
from pathlib import Path

# Core FALCON
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

# Advanced ML components
from falcon.ml import NeuralPerception, MLDecisionCore, OnlineNeuralPerception

# Swarm intelligence
from falcon.distributed import FalconSwarm, SwarmCoordinator, SharedExperiencePool

# Persistence
from falcon.persistence import save_falcon, load_falcon, FalconCheckpoint


class NetworkDataGenerator:
    """Simulates network traffic with various anomaly types"""

    def __init__(self):
        self.time = 0
        self.attack_active = False
        self.attack_type = None

    def generate_packet(self):
        """Generate a simulated network packet"""
        self.time += 1

        # Randomly inject attacks
        if self.time % 100 == 0:
            self.attack_active = True
            self.attack_type = random.choice(['ddos', 'intrusion', 'data_exfiltration'])
        elif self.time % 100 == 30:
            self.attack_active = False

        if self.attack_active:
            # Attack traffic
            if self.attack_type == 'ddos':
                packet_size = random.gauss(1500, 200)
                packet_rate = random.gauss(10000, 2000)
                src_port = random.randint(1, 1000)
            elif self.attack_type == 'intrusion':
                packet_size = random.gauss(500, 100)
                packet_rate = random.gauss(100, 20)
                src_port = random.choice([22, 23, 3389])  # SSH, Telnet, RDP
            else:  # data_exfiltration
                packet_size = random.gauss(8000, 1000)
                packet_rate = random.gauss(5000, 500)
                src_port = random.randint(1024, 65535)
        else:
            # Normal traffic
            packet_size = random.gauss(500, 100)
            packet_rate = random.gauss(100, 30)
            src_port = random.choice([80, 443, 53])  # HTTP, HTTPS, DNS

        return {
            'packet_size': max(packet_size, 0),
            'packet_rate': max(packet_rate, 0),
            'src_port': int(src_port),
            'timestamp': self.time,
            'is_attack': self.attack_active,
            'attack_type': self.attack_type if self.attack_active else None
        }


def demo_ml_perception():
    """Demo 1: ML-based perception"""
    print("\n" + "=" * 70)
    print("DEMO 1: ML-Based Perception & Decision Making")
    print("=" * 70)

    # Create FALCON with ML components
    falcon_ml = FalconAI(
        perception=OnlineNeuralPerception(input_dim=3),
        decision=MLDecisionCore(model_type='random_forest'),
        correction=OutcomeBasedCorrection(learning_rate=0.15),
        energy_manager=SimpleEnergyManager(max_operations=5000),
        memory=ExperienceMemory()
    )

    print("\nSystem configured with:")
    print("  [*] Online Neural Perception (adaptive learning)")
    print("  [*] ML Decision Core (Random Forest)")
    print("  [*] Energy-Aware Processing")
    print("\nProcessing network traffic stream...\n")

    netgen = NetworkDataGenerator()
    attacks_detected = 0
    true_positives = 0
    false_positives = 0

    for i in range(150):
        packet = netgen.generate_packet()

        # Extract features for ML
        features = [
            packet['packet_size'] / 1000.0,
            packet['packet_rate'] / 1000.0,
            packet['src_port'] / 65535.0
        ]

        decision = falcon_ml.process(features)

        if decision:
            attacks_detected += 1
            is_real_attack = packet['is_attack']

            if is_real_attack:
                true_positives += 1
                outcome = Outcome(OutcomeType.SUCCESS, reward=1.0)
                result_icon = "[+]"
            else:
                false_positives += 1
                outcome = Outcome(OutcomeType.FAILURE, reward=-0.5)
                result_icon = "[-]"

            falcon_ml.observe(decision, outcome)

            # Train ML model with feedback
            if hasattr(falcon_ml.decision, 'update'):
                falcon_ml.decision.update(features, decision.action)

            if i % 10 == 0:  # Print sample detections
                print(f"  {result_icon} Time {packet['timestamp']:3d}: "
                      f"{decision.action.value:12s} | "
                      f"Attack: {packet['attack_type'] or 'None':20s} | "
                      f"Confidence: {decision.confidence:.2f}")

    print(f"\n  Detection Performance:")
    print(f"    Total detections: {attacks_detected}")
    print(f"    True positives: {true_positives}")
    print(f"    False positives: {false_positives}")
    if attacks_detected > 0:
        print(f"    Precision: {true_positives/attacks_detected:.1%}")

    # Save the trained model
    save_path = Path("models/network_monitor_ml")
    save_path.parent.mkdir(exist_ok=True)

    save_falcon(
        falcon_ml,
        str(save_path),
        checkpoint_info=FalconCheckpoint(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes="ML-trained network monitor with online learning"
        )
    )

    print(f"\n  [OK] Model saved to {save_path}.falcon")

    return falcon_ml


def demo_swarm_intelligence():
    """Demo 2: Multi-Agent Swarm"""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Agent Swarm Intelligence")
    print("=" * 70)

    print("\nDeploying swarm of 5 FALCON agents to monitor network...")
    print("Each agent specializes in different attack signatures\n")

    # Factory to create agents
    def create_agent():
        return FalconAI(
            perception=ThresholdPerception(
                threshold=random.uniform(0.5, 0.8)  # Varying sensitivity
            ),
            decision=HeuristicDecision(),
            correction=OutcomeBasedCorrection(learning_rate=0.1),
            energy_manager=SimpleEnergyManager(max_operations=2000),
            memory=ExperienceMemory()
        )

    # Create swarm
    swarm = FalconSwarm(
        num_agents=5,
        agent_factory=create_agent,
        consensus_method='weighted'
    )

    netgen = NetworkDataGenerator()
    swarm_detections = 0
    swarm_true_positives = 0

    print("Processing distributed traffic (each agent sees different packets)...\n")

    for i in range(100):
        packet = netgen.generate_packet()
        features = packet['packet_rate'] / 1000.0

        # Swarm consensus decision
        decision = swarm.process(features, use_consensus=True)

        if decision:
            swarm_detections += 1
            is_real_attack = packet['is_attack']

            if is_real_attack:
                swarm_true_positives += 1
                outcome = Outcome(OutcomeType.SUCCESS, reward=1.0)
                status = "SUCCESS"
            else:
                outcome = Outcome(OutcomeType.FAILURE, reward=-0.3)
                status = "FALSE ALARM"

            # All agents learn from outcome
            swarm.observe(decision, outcome)

            if i % 10 == 0:
                num_votes = decision.metadata.get('num_agents', 0)
                print(f"  [Swarm] Time {i:3d}: {decision.action.value:10s} | "
                      f"Votes: {num_votes}/5 | "
                      f"Confidence: {decision.confidence:.2f} | "
                      f"{status}")

    # Show swarm statistics
    stats = swarm.get_swarm_status()
    load_dist = swarm.get_load_distribution()

    print(f"\n  Swarm Performance:")
    print(f"    Total detections: {swarm_detections}")
    print(f"    True positives: {swarm_true_positives}")
    if swarm_detections > 0:
        print(f"    Precision: {swarm_true_positives/swarm_detections:.1%}")
    print(f"    Average confidence: {stats['average_confidence']:.2f}")
    print(f"    Shared experiences: {stats['shared_pool']['total_experiences']}")

    print(f"\n  Load Distribution:")
    for agent_load in load_dist['agent_loads']:
        print(f"    Agent {agent_load['agent_id']}: "
              f"{agent_load['events_processed']} events processed")

    return swarm


def demo_model_persistence():
    """Demo 3: Model Persistence & Production Deployment"""
    print("\n" + "=" * 70)
    print("DEMO 3: Model Persistence & Production Deployment")
    print("=" * 70)

    # Load previously saved model
    print("\n  [*] Loading pretrained model from disk...")

    try:
        falcon_loaded = load_falcon("models/network_monitor_ml")

        print("  [OK] Model loaded successfully!")
        print("\n  Testing loaded model on new data...\n")

        netgen = NetworkDataGenerator()
        test_detections = 0

        for i in range(30):
            packet = netgen.generate_packet()
            features = [
                packet['packet_size'] / 1000.0,
                packet['packet_rate'] / 1000.0,
                packet['src_port'] / 65535.0
            ]

            decision = falcon_loaded.process(features)

            if decision:
                test_detections += 1
                print(f"    Time {i:3d}: {decision.action.value} "
                      f"(confidence: {decision.confidence:.2f})")

        print(f"\n  [OK] Loaded model made {test_detections} detections")

    except FileNotFoundError:
        print("  [!] No saved model found - run Demo 1 first")


def main():
    """Run all advanced demos"""
    print("\n" + "=" * 70)
    print("FALCON-AI Advanced Production Demo")
    print("Scenario: Distributed Network Security Monitoring")
    print("=" * 70)

    print("\nThis demo showcases:")
    print("  1. Machine Learning-based perception and decision making")
    print("  2. Multi-agent swarm intelligence with shared knowledge")
    print("  3. Model persistence for production deployment")

    input("\nPress Enter to start Demo 1 (ML Perception)...")
    demo_ml_perception()

    input("\nPress Enter to start Demo 2 (Swarm Intelligence)...")
    demo_swarm_intelligence()

    input("\nPress Enter to start Demo 3 (Model Persistence)...")
    demo_model_persistence()

    print("\n" + "=" * 70)
    print("All Demos Complete!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("  [*] ML models learn online from outcomes")
    print("  [*] Multiple agents coordinate for better accuracy")
    print("  [*] Models can be saved/loaded for production")
    print("  [*] System adapts continuously to new threats")

    print("\nProduction-Ready Features:")
    print("  [+] Online learning (no retraining needed)")
    print("  [+] Distributed processing (horizontal scaling)")
    print("  [+] Shared knowledge (swarm intelligence)")
    print("  [+] Model versioning (checkpoints)")
    print("  [+] Energy-aware (resource optimization)")

    print("\n[OK] FALCON-AI is ready for production deployment!")


if __name__ == "__main__":
    main()
