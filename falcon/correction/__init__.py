"""
Mid-Flight Correction Loop - Layer 3

Observes outcomes and adjusts after acting.
Learns while deployed, safely.
"""

from .base import CorrectionLoop, Outcome, OutcomeType
from .learners import (
    OutcomeBasedCorrection,
    BayesianCorrection,
    ReinforcementCorrection
)

__all__ = [
    'CorrectionLoop',
    'Outcome',
    'OutcomeType',
    'OutcomeBasedCorrection',
    'BayesianCorrection',
    'ReinforcementCorrection'
]
