"""
Concrete implementations of energy managers.
"""

from typing import Optional, Dict, Any
from collections import defaultdict

from .base import EnergyManager, ComputeBudget, InferenceMode


# Cost table for different inference modes
MODE_COSTS = {
    InferenceMode.MINIMAL: {'operations': 10, 'latency_ms': 1.0, 'energy': 0.5},
    InferenceMode.LIGHT: {'operations': 50, 'latency_ms': 5.0, 'energy': 2.0},
    InferenceMode.STANDARD: {'operations': 200, 'latency_ms': 20.0, 'energy': 10.0},
    InferenceMode.DEEP: {'operations': 1000, 'latency_ms': 100.0, 'energy': 50.0},
    InferenceMode.CLOUD: {'operations': 500, 'latency_ms': 200.0, 'energy': 5.0}
}


class SimpleEnergyManager(EnergyManager):
    """
    Simple energy manager that chooses mode based on remaining budget.
    """

    def __init__(self, budget: Optional[ComputeBudget] = None,
                 max_operations: int = 10000,
                 max_latency_ms: float = 1000.0,
                 max_energy_units: float = 500.0):
        """
        Args:
            budget: ComputeBudget object (or create from individual params)
            max_operations: Maximum operations if creating new budget
            max_latency_ms: Maximum latency if creating new budget
            max_energy_units: Maximum energy if creating new budget
        """
        if budget is None:
            budget = ComputeBudget(
                max_operations=max_operations,
                max_latency_ms=max_latency_ms,
                max_energy_units=max_energy_units
            )
        super().__init__(budget)

    def choose_inference_mode(self, context: Optional[Dict[str, Any]] = None) -> InferenceMode:
        remaining = self.budget.remaining_fraction()

        # Choose mode based on remaining budget
        if remaining > 0.7:
            return InferenceMode.STANDARD
        elif remaining > 0.4:
            return InferenceMode.LIGHT
        elif remaining > 0.1:
            return InferenceMode.MINIMAL
        else:
            # Budget critical - use minimal
            return InferenceMode.MINIMAL

    def record_usage(self, mode: InferenceMode, actual_cost: Optional[Dict[str, float]] = None):
        if actual_cost is None:
            actual_cost = MODE_COSTS.get(mode, MODE_COSTS[InferenceMode.STANDARD])

        self.budget.consume(
            operations=int(actual_cost.get('operations', 0)),
            latency_ms=actual_cost.get('latency_ms', 0.0),
            energy_units=actual_cost.get('energy', 0.0)
        )


class AdaptiveEnergyManager(EnergyManager):
    """
    Adaptive energy manager that learns optimal modes for different contexts.
    """

    def __init__(self, budget: ComputeBudget, adaptation_rate: float = 0.1):
        """
        Args:
            budget: ComputeBudget object
            adaptation_rate: How quickly to adapt mode selection
        """
        super().__init__(budget)
        self.adaptation_rate = adaptation_rate
        self.mode_performance: Dict[InferenceMode, float] = defaultdict(lambda: 0.5)
        self.mode_usage_count: Dict[InferenceMode, int] = defaultdict(int)

    def choose_inference_mode(self, context: Optional[Dict[str, Any]] = None) -> InferenceMode:
        remaining = self.budget.remaining_fraction()

        # Filter modes by budget availability
        affordable_modes = []
        for mode in InferenceMode:
            cost = MODE_COSTS.get(mode, MODE_COSTS[InferenceMode.STANDARD])
            if self.budget.has_budget(
                operations=int(cost['operations']),
                latency_ms=cost['latency_ms'],
                energy_units=cost['energy']
            ):
                affordable_modes.append(mode)

        if not affordable_modes:
            return InferenceMode.MINIMAL

        # Among affordable modes, choose based on learned performance
        if context and context.get('use_learned_preference', True):
            best_mode = max(affordable_modes, key=lambda m: self.mode_performance[m])
            return best_mode
        else:
            # Fallback to budget-based selection
            if remaining > 0.7:
                return InferenceMode.STANDARD if InferenceMode.STANDARD in affordable_modes else affordable_modes[0]
            elif remaining > 0.4:
                return InferenceMode.LIGHT if InferenceMode.LIGHT in affordable_modes else affordable_modes[0]
            else:
                return InferenceMode.MINIMAL

    def record_usage(self, mode: InferenceMode, actual_cost: Optional[Dict[str, float]] = None):
        if actual_cost is None:
            actual_cost = MODE_COSTS.get(mode, MODE_COSTS[InferenceMode.STANDARD])

        self.budget.consume(
            operations=int(actual_cost.get('operations', 0)),
            latency_ms=actual_cost.get('latency_ms', 0.0),
            energy_units=actual_cost.get('energy', 0.0)
        )

        self.mode_usage_count[mode] += 1

    def update_mode_performance(self, mode: InferenceMode, performance: float):
        """
        Update learned performance for a mode.

        Args:
            mode: The inference mode
            performance: Performance score (0.0 to 1.0)
        """
        old_perf = self.mode_performance[mode]
        self.mode_performance[mode] = old_perf + self.adaptation_rate * (performance - old_perf)


class MultiTierEnergyManager(EnergyManager):
    """
    Multi-tier energy manager with urgency-based mode selection.
    """

    def __init__(self, budget: ComputeBudget):
        super().__init__(budget)
        self.urgency_thresholds = {
            'critical': InferenceMode.DEEP,
            'high': InferenceMode.STANDARD,
            'medium': InferenceMode.LIGHT,
            'low': InferenceMode.MINIMAL
        }

    def choose_inference_mode(self, context: Optional[Dict[str, Any]] = None) -> InferenceMode:
        # Check urgency in context
        urgency = 'medium'
        if context:
            urgency = context.get('urgency', 'medium')

        # Get suggested mode based on urgency
        suggested_mode = self.urgency_thresholds.get(urgency, InferenceMode.STANDARD)

        # Check if we can afford it
        cost = MODE_COSTS.get(suggested_mode, MODE_COSTS[InferenceMode.STANDARD])
        if self.budget.has_budget(
            operations=int(cost['operations']),
            latency_ms=cost['latency_ms'],
            energy_units=cost['energy']
        ):
            return suggested_mode

        # Fallback to cheaper modes
        mode_hierarchy = [
            InferenceMode.MINIMAL,
            InferenceMode.LIGHT,
            InferenceMode.STANDARD,
            InferenceMode.DEEP,
            InferenceMode.CLOUD
        ]

        for mode in mode_hierarchy:
            cost = MODE_COSTS.get(mode, MODE_COSTS[InferenceMode.MINIMAL])
            if self.budget.has_budget(
                operations=int(cost['operations']),
                latency_ms=cost['latency_ms'],
                energy_units=cost['energy']
            ):
                return mode

        return InferenceMode.MINIMAL

    def record_usage(self, mode: InferenceMode, actual_cost: Optional[Dict[str, float]] = None):
        if actual_cost is None:
            actual_cost = MODE_COSTS.get(mode, MODE_COSTS[InferenceMode.STANDARD])

        self.budget.consume(
            operations=int(actual_cost.get('operations', 0)),
            latency_ms=actual_cost.get('latency_ms', 0.0),
            energy_units=actual_cost.get('energy', 0.0)
        )
