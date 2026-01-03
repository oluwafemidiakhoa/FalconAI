"""
FALCON-AI Core Orchestrator

Integrates all 5 layers into a cohesive system.
"""

from typing import Any, Optional, Dict
import time

from .perception.base import PerceptionEngine, Event
from .decision.base import DecisionCore, Decision
from .correction.base import CorrectionLoop, Outcome, OutcomeType
from .energy.base import EnergyManager
from .memory.base import Memory
from .memory.experience import ExperienceEntry
from .utils.metrics import SystemMonitor


class FalconAI:
    """
    Main FALCON-AI system orchestrator.

    Integrates:
    - Layer 1: Selective Perception
    - Layer 2: Fast Decision
    - Layer 3: Mid-Flight Correction
    - Layer 4: Energy-Aware Intelligence
    - Layer 5: Instinct + Experience Memory
    """

    def __init__(self,
                 perception: PerceptionEngine,
                 decision: DecisionCore,
                 correction: CorrectionLoop,
                 energy_manager: EnergyManager,
                 memory: Optional[Memory] = None,
                 enable_monitoring: bool = True):
        """
        Initialize FALCON-AI system.

        Args:
            perception: Perception engine (Layer 1)
            decision: Decision core (Layer 2)
            correction: Correction loop (Layer 3)
            energy_manager: Energy manager (Layer 4)
            memory: Memory system (Layer 5)
            enable_monitoring: Whether to enable performance monitoring
        """
        self.perception = perception
        self.decision = decision
        self.correction = correction
        self.energy_manager = energy_manager
        self.memory = memory

        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = SystemMonitor()
        else:
            self.monitor = None

    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Optional[Decision]:
        """
        Process incoming data through the FALCON pipeline.

        Args:
            data: Raw input data
            context: Optional context information

        Returns:
            Decision object if action is needed, None otherwise
        """
        start_time = time.time()

        # Layer 1: Selective Perception
        event = self.perception.perceive(data)

        if event is None or not event.is_actionable():
            # No salient event detected - return None (falcon stays perched)
            return None

        # Layer 4: Check energy budget and choose inference mode
        inference_mode = self.energy_manager.choose_inference_mode(context)

        # Layer 5: Query memory for similar situations
        memory_context = context or {}
        if self.memory:
            # Add memory-based context
            past_experience = self.memory.retrieve(str(event.event_type.value))
            if past_experience:
                memory_context['past_experience'] = past_experience

        # Layer 2: Make fast decision
        decision = self.decision.decide(event, context=memory_context)

        # Record energy usage
        latency_ms = (time.time() - start_time) * 1000
        self.energy_manager.record_usage(
            inference_mode,
            {'operations': 100, 'latency_ms': latency_ms, 'energy': 5.0}
        )

        # Monitor if enabled
        if self.monitor:
            # We don't know success yet, so we'll update later
            self.monitor.record_event(latency_ms, decision.confidence, True)

        return decision

    def observe(self, decision: Decision, outcome: Outcome, context: Optional[Dict[str, Any]] = None):
        """
        Observe the outcome of a decision and learn from it.

        This implements the mid-flight correction loop.

        Args:
            decision: The decision that was executed
            outcome: The observed outcome
            context: Optional context information
        """
        # Layer 3: Correction loop observes outcome
        self.correction.observe(decision, outcome, context)

        # Layer 5: Store experience in memory
        if self.memory:
            experience = ExperienceEntry(
                situation=decision.action.value,
                action=decision.action.value,
                outcome=outcome.outcome_type.value,
                reward=outcome.reward,
                metadata=context or {}
            )
            self.memory.store(decision.action.value, experience)

        # Check if we should abort current strategy
        if self.correction.should_abort(context):
            # In a real system, this might trigger a strategy change
            pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dictionary with status information from all layers
        """
        status = {
            'perception': {
                'events_processed': self.perception.events_processed,
                'events_triggered': self.perception.events_triggered,
                'trigger_rate': self.perception.get_trigger_rate()
            },
            'decision': {
                'decisions_made': self.decision.decisions_made,
                'average_confidence': self.decision.get_average_confidence()
            },
            'correction': {
                'corrections_made': self.correction.corrections_made,
                'average_reward': self.correction.get_average_reward(),
                'correction_signals': self.correction.get_correction_signal()
            },
            'energy': self.energy_manager.get_budget_status(),
        }

        if self.memory:
            status['memory'] = {
                'size': self.memory.size(),
                'type': self.memory.memory_type.value
            }

        if self.monitor:
            status['performance'] = self.monitor.get_stats()

        return status

    def reset(self):
        """Reset all components to initial state"""
        self.perception.reset_stats()
        self.decision.reset_stats()
        self.correction.reset_stats()
        self.energy_manager.reset_budget()

        if self.monitor:
            self.monitor = SystemMonitor()
