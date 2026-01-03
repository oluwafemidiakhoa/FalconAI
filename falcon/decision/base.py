"""
Base classes for the Fast Decision Core.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict
from enum import Enum


class ActionType(Enum):
    """Types of actions the system can take"""
    OBSERVE = "observe"      # Continue monitoring
    ALERT = "alert"          # Raise an alert
    INTERVENE = "intervene"  # Take corrective action
    ESCALATE = "escalate"    # Escalate to higher authority
    IGNORE = "ignore"        # Explicitly ignore


@dataclass
class Decision:
    """
    Represents a decision made by the system.

    Decisions are provisional - they're hypotheses that can be corrected.
    """
    action: ActionType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if decision has high confidence"""
        return self.confidence >= threshold

    def should_execute(self, min_confidence: float = 0.5) -> bool:
        """Check if decision should be executed"""
        return self.confidence >= min_confidence


class DecisionCore(ABC):
    """
    Abstract base class for decision cores.

    The decision core's job is to:
    1. Make fast decisions based on events
    2. Provide confidence scores
    3. Optimize for speed over perfection
    4. Treat decisions as provisional hypotheses
    """

    def __init__(self):
        self.decisions_made = 0
        self.total_confidence = 0.0

    @abstractmethod
    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        """
        Make a decision based on an event.

        Args:
            event: The event to respond to
            context: Additional context for decision making

        Returns:
            Decision object with action and confidence
        """
        pass

    def get_average_confidence(self) -> float:
        """Calculate average confidence of decisions made"""
        if self.decisions_made == 0:
            return 0.0
        return self.total_confidence / self.decisions_made

    def reset_stats(self):
        """Reset internal statistics"""
        self.decisions_made = 0
        self.total_confidence = 0.0
