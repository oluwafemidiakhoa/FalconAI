"""
Base classes for the Selective Perception Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict
from enum import Enum


class EventType(Enum):
    """Types of detected events"""
    NORMAL = "normal"
    SALIENT = "salient"
    ANOMALY = "anomaly"
    CRITICAL = "critical"


@dataclass
class Event:
    """Represents a detected event"""
    data: Any
    event_type: EventType
    salience_score: float  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None

    def is_actionable(self) -> bool:
        """Check if this event should trigger action"""
        return self.event_type != EventType.NORMAL


class PerceptionEngine(ABC):
    """
    Abstract base class for perception engines.

    The perception engine's job is to:
    1. Filter incoming data
    2. Detect salient events
    3. Ignore low-signal noise
    4. Only activate downstream processing when needed
    """

    def __init__(self):
        self.events_processed = 0
        self.events_triggered = 0

    @abstractmethod
    def perceive(self, data: Any) -> Optional[Event]:
        """
        Process incoming data and return an Event if salient.

        Args:
            data: Raw input data

        Returns:
            Event object if data is salient, None otherwise
        """
        pass

    def get_trigger_rate(self) -> float:
        """Calculate what percentage of inputs trigger events"""
        if self.events_processed == 0:
            return 0.0
        return self.events_triggered / self.events_processed

    def reset_stats(self):
        """Reset internal statistics"""
        self.events_processed = 0
        self.events_triggered = 0
