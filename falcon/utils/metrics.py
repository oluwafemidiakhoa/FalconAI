"""
Performance metrics and monitoring utilities.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class PerformanceMetrics:
    """
    Tracks performance metrics for FALCON-AI.
    """
    total_events: int = 0
    events_triggered: int = 0
    decisions_made: int = 0
    corrections_applied: int = 0
    average_latency_ms: float = 0.0
    average_confidence: float = 0.0
    success_rate: float = 0.0

    def trigger_rate(self) -> float:
        """Calculate what fraction of events trigger actions"""
        if self.total_events == 0:
            return 0.0
        return self.events_triggered / self.total_events

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_events': self.total_events,
            'events_triggered': self.events_triggered,
            'decisions_made': self.decisions_made,
            'corrections_applied': self.corrections_applied,
            'trigger_rate': self.trigger_rate(),
            'average_latency_ms': self.average_latency_ms,
            'average_confidence': self.average_confidence,
            'success_rate': self.success_rate
        }


class SystemMonitor:
    """
    Monitors system performance over time.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent events to track
        """
        self.window_size = window_size
        self.event_times: List[float] = []
        self.latencies: List[float] = []
        self.confidences: List[float] = []
        self.successes: List[bool] = []
        self.start_time = time.time()

    def record_event(self, latency_ms: float, confidence: float, success: bool):
        """
        Record an event.

        Args:
            latency_ms: Processing latency in milliseconds
            confidence: Decision confidence
            success: Whether the decision was successful
        """
        self.event_times.append(time.time())
        self.latencies.append(latency_ms)
        self.confidences.append(confidence)
        self.successes.append(success)

        # Keep only recent window
        if len(self.latencies) > self.window_size:
            self.event_times.pop(0)
            self.latencies.pop(0)
            self.confidences.pop(0)
            self.successes.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        if not self.latencies:
            return {
                'events_processed': 0,
                'avg_latency_ms': 0.0,
                'avg_confidence': 0.0,
                'success_rate': 0.0,
                'events_per_second': 0.0,
                'uptime_seconds': time.time() - self.start_time
            }

        events_per_second = 0.0
        if len(self.event_times) > 1:
            time_span = self.event_times[-1] - self.event_times[0]
            if time_span > 0:
                events_per_second = len(self.event_times) / time_span

        return {
            'events_processed': len(self.latencies),
            'avg_latency_ms': sum(self.latencies) / len(self.latencies),
            'avg_confidence': sum(self.confidences) / len(self.confidences),
            'success_rate': sum(self.successes) / len(self.successes),
            'events_per_second': events_per_second,
            'uptime_seconds': time.time() - self.start_time
        }
