"""
Concrete implementations of perception engines.
"""

import numpy as np
from typing import Any, Optional, List, Callable
from collections import deque

from .base import PerceptionEngine, Event, EventType


class ThresholdPerception(PerceptionEngine):
    """
    Simple threshold-based perception.
    Triggers when input value exceeds a threshold.
    """

    def __init__(self, threshold: float = 0.7, value_extractor: Optional[Callable] = None):
        """
        Args:
            threshold: Values above this trigger an event
            value_extractor: Function to extract numeric value from data
        """
        super().__init__()
        self.threshold = threshold
        self.value_extractor = value_extractor if value_extractor is not None else float

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        try:
            value = self.value_extractor(data)
        except (TypeError, ValueError):
            return None

        if value > self.threshold:
            self.events_triggered += 1

            # Determine event type based on how far above threshold
            if value > self.threshold * 1.5:
                event_type = EventType.CRITICAL
            elif value > self.threshold * 1.2:
                event_type = EventType.SALIENT
            else:
                event_type = EventType.SALIENT

            return Event(
                data=data,
                event_type=event_type,
                salience_score=min(value / self.threshold, 1.0),
                metadata={'value': value, 'threshold': self.threshold}
            )

        return None


class ChangeDetectionPerception(PerceptionEngine):
    """
    Detects significant changes in input over time.
    Triggers when change exceeds threshold.
    """

    def __init__(self, change_threshold: float = 0.3, window_size: int = 10,
                 value_extractor: Optional[Callable] = None):
        """
        Args:
            change_threshold: Fractional change required to trigger
            window_size: Number of recent values to track
            value_extractor: Function to extract numeric value from data
        """
        super().__init__()
        self.change_threshold = change_threshold
        self.window_size = window_size
        self.value_extractor = value_extractor if value_extractor is not None else float
        self.history = deque(maxlen=window_size)

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        try:
            value = self.value_extractor(data)
        except (TypeError, ValueError):
            return None

        if len(self.history) > 0:
            baseline = np.mean(self.history)
            change = abs(value - baseline) / (baseline + 1e-8)

            if change > self.change_threshold:
                self.events_triggered += 1

                event_type = EventType.CRITICAL if change > self.change_threshold * 2 else EventType.SALIENT

                self.history.append(value)

                return Event(
                    data=data,
                    event_type=event_type,
                    salience_score=min(change / self.change_threshold, 1.0),
                    metadata={
                        'value': value,
                        'baseline': baseline,
                        'change': change
                    }
                )

        self.history.append(value)
        return None


class AnomalyPerception(PerceptionEngine):
    """
    Statistical anomaly detection using z-score.
    Triggers when input is statistically anomalous.
    """

    def __init__(self, z_threshold: float = 2.5, min_samples: int = 20,
                 value_extractor: Optional[Callable] = None):
        """
        Args:
            z_threshold: Z-score threshold for anomaly detection
            min_samples: Minimum samples before anomaly detection activates
            value_extractor: Function to extract numeric value from data
        """
        super().__init__()
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.value_extractor = value_extractor if value_extractor is not None else float
        self.values = []

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        try:
            value = self.value_extractor(data)
        except (TypeError, ValueError):
            return None

        if len(self.values) >= self.min_samples:
            mean = np.mean(self.values)
            std = np.std(self.values)

            if std > 0:
                z_score = abs(value - mean) / std

                if z_score > self.z_threshold:
                    self.events_triggered += 1

                    event_type = EventType.ANOMALY
                    if z_score > self.z_threshold * 2:
                        event_type = EventType.CRITICAL

                    self.values.append(value)

                    return Event(
                        data=data,
                        event_type=event_type,
                        salience_score=min(z_score / self.z_threshold, 1.0),
                        metadata={
                            'value': value,
                            'mean': mean,
                            'std': std,
                            'z_score': z_score
                        }
                    )

        self.values.append(value)
        return None


class CompositePerception(PerceptionEngine):
    """
    Combines multiple perception engines.
    Triggers if any component engine triggers.
    """

    def __init__(self, engines: List[PerceptionEngine]):
        """
        Args:
            engines: List of perception engines to combine
        """
        super().__init__()
        self.engines = engines

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        events = []
        for engine in self.engines:
            event = engine.perceive(data)
            if event is not None:
                events.append(event)

        if events:
            self.events_triggered += 1
            # Return the most salient event
            return max(events, key=lambda e: e.salience_score)

        return None
