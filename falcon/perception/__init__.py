"""
Selective Perception Engine - Layer 1

Detects salient events and filters low-signal data.
Only triggers deeper analysis when necessary.
"""

from .base import PerceptionEngine, Event, EventType
from .filters import (
    ThresholdPerception,
    ChangeDetectionPerception,
    AnomalyPerception,
    CompositePerception
)

__all__ = [
    'PerceptionEngine',
    'Event',
    'EventType',
    'ThresholdPerception',
    'ChangeDetectionPerception',
    'AnomalyPerception',
    'CompositePerception'
]
