"""
Base classes for the Mid-Flight Correction Loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict
from enum import Enum


class OutcomeType(Enum):
    """Types of outcomes from decisions"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class Outcome:
    """
    Represents the result of executing a decision.
    """
    outcome_type: OutcomeType
    reward: float  # Numeric reward signal (-1.0 to 1.0)
    metadata: Optional[Dict[str, Any]] = None

    def is_successful(self) -> bool:
        """Check if outcome was successful"""
        return self.outcome_type == OutcomeType.SUCCESS

    def is_failure(self) -> bool:
        """Check if outcome was a failure"""
        return self.outcome_type == OutcomeType.FAILURE


class CorrectionLoop(ABC):
    """
    Abstract base class for correction loops.

    The correction loop's job is to:
    1. Observe outcomes of decisions
    2. Update internal models based on results
    3. Provide feedback to improve future decisions
    4. Learn safely during deployment
    """

    def __init__(self):
        self.corrections_made = 0
        self.total_reward = 0.0

    @abstractmethod
    def observe(self, decision: Any, outcome: Outcome, context: Optional[Dict[str, Any]] = None):
        """
        Observe the outcome of a decision and learn from it.

        Args:
            decision: The decision that was executed
            outcome: The observed outcome
            context: Additional context
        """
        pass

    @abstractmethod
    def should_abort(self, current_state: Any) -> bool:
        """
        Determine if current action should be aborted.

        Args:
            current_state: Current state of the system

        Returns:
            True if action should be aborted
        """
        pass

    @abstractmethod
    def get_correction_signal(self) -> Dict[str, Any]:
        """
        Get correction signals for other components.

        Returns:
            Dictionary of correction parameters
        """
        pass

    def get_average_reward(self) -> float:
        """Calculate average reward"""
        if self.corrections_made == 0:
            return 0.0
        return self.total_reward / self.corrections_made

    def reset_stats(self):
        """Reset internal statistics"""
        self.corrections_made = 0
        self.total_reward = 0.0
