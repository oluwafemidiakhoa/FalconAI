"""
FALCON-AI: A Selective, Fast, Self-Adapting Intelligence System

A decision-first intelligence engine inspired by falcon hunting behavior.
"""

from .core import FalconAI

# Layer 1: Perception
from .perception import (
    PerceptionEngine,
    Event,
    EventType,
    ThresholdPerception,
    ChangeDetectionPerception,
    AnomalyPerception,
    CompositePerception
)

# Layer 2: Decision
from .decision import (
    DecisionCore,
    Decision,
    ActionType,
    HeuristicDecision,
    ThresholdDecision,
    RuleBasedDecision,
    HybridDecision,
    MemoryAwareDecision,
)

# Layer 3: Correction
from .correction import (
    CorrectionLoop,
    Outcome,
    OutcomeType,
    OutcomeBasedCorrection,
    BayesianCorrection,
    ReinforcementCorrection
)

# Layer 4: Energy
from .energy import (
    EnergyManager,
    ComputeBudget,
    InferenceMode,
    SimpleEnergyManager,
    AdaptiveEnergyManager,
    MultiTierEnergyManager
)

# Layer 5: Memory
from .memory import (
    Memory,
    MemoryType,
    InstinctMemory,
    ExperienceMemory,
    ExperienceEntry
)

# Utils
from .utils import PerformanceMetrics, SystemMonitor

__version__ = '0.1.0'

__all__ = [
    # Core
    'FalconAI',

    # Perception
    'PerceptionEngine',
    'Event',
    'EventType',
    'ThresholdPerception',
    'ChangeDetectionPerception',
    'AnomalyPerception',
    'CompositePerception',

    # Decision
    'DecisionCore',
    'Decision',
    'ActionType',
    'HeuristicDecision',
    'ThresholdDecision',
    'RuleBasedDecision',
    'HybridDecision',
    'MemoryAwareDecision',

    # Correction
    'CorrectionLoop',
    'Outcome',
    'OutcomeType',
    'OutcomeBasedCorrection',
    'BayesianCorrection',
    'ReinforcementCorrection',

    # Energy
    'EnergyManager',
    'ComputeBudget',
    'InferenceMode',
    'SimpleEnergyManager',
    'AdaptiveEnergyManager',
    'MultiTierEnergyManager',

    # Memory
    'Memory',
    'MemoryType',
    'InstinctMemory',
    'ExperienceMemory',
    'ExperienceEntry',

    # Utils
    'PerformanceMetrics',
    'SystemMonitor'
]
