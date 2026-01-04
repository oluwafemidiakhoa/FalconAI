"""
Base classes for Energy-Aware Intelligence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class InferenceMode(Enum):
    """Different inference modes with varying resource costs"""
    MINIMAL = "minimal"      # Lowest cost, fastest
    LIGHT = "light"          # Low cost, fast
    STANDARD = "standard"    # Balanced
    DEEP = "deep"            # High cost, thorough
    CLOUD = "cloud"          # Offload to cloud


@dataclass
class ComputeBudget:
    """
    Represents available computational resources.
    """
    max_operations: int  # Maximum operations allowed
    max_latency_ms: float  # Maximum latency in milliseconds
    max_energy_units: float  # Abstract energy units

    used_operations: int = 0
    used_latency_ms: float = 0.0
    used_energy_units: float = 0.0

    def has_budget(self, operations: int = 0, latency_ms: float = 0.0,
                   energy_units: float = 0.0) -> bool:
        """Check if budget is available for an operation"""
        return (
            self.used_operations + operations <= self.max_operations and
            self.used_latency_ms + latency_ms <= self.max_latency_ms and
            self.used_energy_units + energy_units <= self.max_energy_units
        )

    def consume(self, operations: int = 0, latency_ms: float = 0.0,
                energy_units: float = 0.0):
        """Consume budget"""
        self.used_operations = min(self.used_operations + operations, self.max_operations)
        self.used_latency_ms = min(self.used_latency_ms + latency_ms, self.max_latency_ms)
        self.used_energy_units = min(self.used_energy_units + energy_units, self.max_energy_units)

    def remaining_fraction(self) -> float:
        """Get fraction of budget remaining (0.0 to 1.0)"""
        op_remaining = 1.0 - (self.used_operations / max(self.max_operations, 1))
        latency_remaining = 1.0 - (self.used_latency_ms / max(self.max_latency_ms, 1))
        energy_remaining = 1.0 - (self.used_energy_units / max(self.max_energy_units, 1))
        remaining = min(op_remaining, latency_remaining, energy_remaining)
        return max(0.0, min(remaining, 1.0))

    def reset(self):
        """Reset budget usage"""
        self.used_operations = 0
        self.used_latency_ms = 0.0
        self.used_energy_units = 0.0


class EnergyManager(ABC):
    """
    Abstract base class for energy managers.

    The energy manager's job is to:
    1. Track resource usage (compute, latency, energy)
    2. Choose appropriate inference modes
    3. Prevent resource exhaustion
    4. Optimize for efficiency
    """

    def __init__(self, budget: ComputeBudget):
        self.budget = budget

    @abstractmethod
    def choose_inference_mode(self, context: Optional[Dict[str, Any]] = None) -> InferenceMode:
        """
        Choose the appropriate inference mode based on available budget.

        Args:
            context: Context for decision (urgency, complexity, etc.)

        Returns:
            Inference mode to use
        """
        pass

    @abstractmethod
    def record_usage(self, mode: InferenceMode, actual_cost: Dict[str, float]):
        """
        Record actual resource usage.

        Args:
            mode: The inference mode that was used
            actual_cost: Actual costs incurred
        """
        pass

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            'remaining_fraction': self.budget.remaining_fraction(),
            'used_operations': self.budget.used_operations,
            'max_operations': self.budget.max_operations,
            'used_latency_ms': self.budget.used_latency_ms,
            'max_latency_ms': self.budget.max_latency_ms,
            'used_energy': self.budget.used_energy_units,
            'max_energy': self.budget.max_energy_units
        }

    def reset_budget(self):
        """Reset budget for new period"""
        self.budget.reset()
