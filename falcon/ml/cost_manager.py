"""
Cost Manager for FALCON Runtime API.

This module tracks inference costs and enforces budget limits.
Prevents runaway spending on expensive model inference.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class CostEntry:
    """Single cost entry record."""
    cost: float
    timestamp: str
    trace_id: str
    model: str
    metadata: Dict


@dataclass
class DailyCostReport:
    """Cost report for a single day."""
    date: str
    total_spend: float
    num_inferences: int
    avg_cost: float
    max_cost: float
    min_cost: float


class CostManager:
    """
    Track and enforce cost budgets for inference operations.

    Features:
    - Daily/monthly budget enforcement
    - Cost attribution by model
    - Spending reports and analytics
    - Alert thresholds

    Example:
        cost_mgr = CostManager(daily_limit=100.0)

        # Check budget before inference
        if not cost_mgr.check_budget(estimated_cost=0.002):
            return {"error": "Budget exceeded"}

        # Record actual cost after inference
        cost_mgr.record_cost(
            cost_usd=0.002,
            trace_id="abc123",
            model="gpt-3.5-turbo",
            metadata={"tokens": 150}
        )

        # Get spending report
        report = cost_mgr.get_report(days=7)
    """

    def __init__(self, daily_limit: float = 100.0, monthly_limit: float = 2000.0):
        """
        Initialize cost manager.

        Args:
            daily_limit: Maximum spend per day in USD
            monthly_limit: Maximum spend per month in USD
        """
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.costs: Dict[str, List[CostEntry]] = defaultdict(list)  # date -> [entries]

    def record_cost(
        self,
        cost_usd: float,
        trace_id: str = "",
        model: str = "unknown",
        metadata: Optional[Dict] = None
    ):
        """
        Record a cost entry.

        Args:
            cost_usd: Cost in USD
            trace_id: Request trace ID
            model: Model name used
            metadata: Additional context
        """
        today = datetime.now().date().isoformat()

        entry = CostEntry(
            cost=cost_usd,
            timestamp=datetime.now().isoformat(),
            trace_id=trace_id,
            model=model,
            metadata=metadata or {}
        )

        self.costs[today].append(entry)

    def get_daily_spend(self, date: Optional[str] = None) -> float:
        """
        Get total spending for a specific date.

        Args:
            date: Date in ISO format (YYYY-MM-DD), defaults to today

        Returns:
            Total spend in USD
        """
        if date is None:
            date = datetime.now().date().isoformat()

        return sum(entry.cost for entry in self.costs.get(date, []))

    def get_monthly_spend(self, year: Optional[int] = None, month: Optional[int] = None) -> float:
        """
        Get total spending for a month.

        Args:
            year: Year (defaults to current)
            month: Month 1-12 (defaults to current)

        Returns:
            Total spend in USD
        """
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        total = 0.0
        for date_str, entries in self.costs.items():
            date = datetime.fromisoformat(date_str).date()
            if date.year == year and date.month == month:
                total += sum(entry.cost for entry in entries)

        return total

    def check_budget(self, estimated_cost: float) -> bool:
        """
        Check if we can afford an inference within budget.

        Args:
            estimated_cost: Estimated cost in USD

        Returns:
            True if within budget, False otherwise
        """
        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        # Check daily limit
        if (daily_spend + estimated_cost) > self.daily_limit:
            return False

        # Check monthly limit
        if (monthly_spend + estimated_cost) > self.monthly_limit:
            return False

        return True

    def get_budget_status(self) -> Dict:
        """
        Get current budget status.

        Returns:
            Dictionary with budget information
        """
        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        return {
            "daily": {
                "spend": round(daily_spend, 4),
                "limit": self.daily_limit,
                "remaining": round(self.daily_limit - daily_spend, 4),
                "utilization": round(daily_spend / self.daily_limit, 4) if self.daily_limit > 0 else 0.0
            },
            "monthly": {
                "spend": round(monthly_spend, 4),
                "limit": self.monthly_limit,
                "remaining": round(self.monthly_limit - monthly_spend, 4),
                "utilization": round(monthly_spend / self.monthly_limit, 4) if self.monthly_limit > 0 else 0.0
            }
        }

    def get_report(self, days: int = 7) -> Dict:
        """
        Get cost report for the last N days.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with daily breakdowns
        """
        report = {}

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            entries = self.costs.get(date, [])

            if entries:
                costs = [e.cost for e in entries]
                total_spend = sum(costs)
                avg_cost = total_spend / len(costs)
                max_cost = max(costs)
                min_cost = min(costs)
            else:
                total_spend = 0.0
                avg_cost = 0.0
                max_cost = 0.0
                min_cost = 0.0

            report[date] = {
                "total_spend": round(total_spend, 4),
                "num_inferences": len(entries),
                "avg_cost": round(avg_cost, 6),
                "max_cost": round(max_cost, 6),
                "min_cost": round(min_cost, 6)
            }

        return report

    def get_cost_by_model(self, days: int = 7) -> Dict:
        """
        Get cost breakdown by model.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary mapping model name to total cost
        """
        model_costs = defaultdict(float)

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            entries = self.costs.get(date, [])

            for entry in entries:
                model_costs[entry.model] += entry.cost

        return {
            model: round(cost, 4)
            for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
        }

    def should_alert(self, threshold: float = 0.8) -> Dict:
        """
        Check if spending is approaching limits.

        Args:
            threshold: Alert threshold (0.0-1.0)

        Returns:
            Dictionary with alert status
        """
        status = self.get_budget_status()

        return {
            "alert": (
                status["daily"]["utilization"] >= threshold or
                status["monthly"]["utilization"] >= threshold
            ),
            "daily_alert": status["daily"]["utilization"] >= threshold,
            "monthly_alert": status["monthly"]["utilization"] >= threshold,
            "daily_utilization": status["daily"]["utilization"],
            "monthly_utilization": status["monthly"]["utilization"]
        }
