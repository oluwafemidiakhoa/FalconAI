"""
Experience Memory - learned from outcomes.
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .base import Memory, MemoryType


@dataclass
class ExperienceEntry:
    """Represents a single experience"""
    situation: str  # Description of the situation
    action: str  # Action taken
    outcome: str  # Outcome observed
    reward: float  # Reward received
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def was_successful(self) -> bool:
        """Check if experience was successful"""
        return self.reward > 0


class ExperienceMemory(Memory):
    """
    Experience memory stores outcomes of past decisions.

    This is episodic memory that learns what works and what doesn't.
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum number of experiences to store
        """
        super().__init__(MemoryType.EXPERIENCE)
        self.max_size = max_size
        self.experiences: deque = deque(maxlen=max_size)
        self.situation_outcomes: Dict[str, List[float]] = {}

    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store an experience.

        Args:
            key: Situation identifier
            value: Should be an ExperienceEntry or dict with experience data
            metadata: Additional metadata
        """
        if isinstance(value, ExperienceEntry):
            entry = value
        elif isinstance(value, dict):
            entry = ExperienceEntry(
                situation=key,
                action=value.get('action', 'unknown'),
                outcome=value.get('outcome', 'unknown'),
                reward=value.get('reward', 0.0),
                metadata=metadata or {}
            )
        else:
            # Treat value as outcome
            entry = ExperienceEntry(
                situation=key,
                action='unknown',
                outcome=str(value),
                reward=0.0,
                metadata=metadata or {}
            )

        self.experiences.append(entry)

        # Update situation outcomes for quick lookup
        if entry.situation not in self.situation_outcomes:
            self.situation_outcomes[entry.situation] = []
        self.situation_outcomes[entry.situation].append(entry.reward)

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve the most recent experience for a situation.

        Args:
            key: Situation identifier

        Returns:
            Most recent ExperienceEntry or None
        """
        for entry in reversed(self.experiences):
            if entry.situation == key:
                return entry
        return None

    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Any]:
        """
        Search experiences based on criteria.

        Args:
            query: Search criteria (e.g., {'action': 'alert', 'min_reward': 0.5})
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        results = []

        for entry in reversed(self.experiences):
            match = True

            # Check each query criterion
            if 'situation' in query and entry.situation != query['situation']:
                match = False
            if 'action' in query and entry.action != query['action']:
                match = False
            if 'outcome' in query and entry.outcome != query['outcome']:
                match = False
            if 'min_reward' in query and entry.reward < query['min_reward']:
                match = False
            if 'max_reward' in query and entry.reward > query['max_reward']:
                match = False

            if match:
                results.append(entry)
                if len(results) >= limit:
                    break

        return results

    def size(self) -> int:
        """Return number of experiences"""
        return len(self.experiences)

    def get_success_rate(self, situation: str) -> float:
        """
        Calculate success rate for a situation.

        Args:
            situation: Situation identifier

        Returns:
            Success rate (0.0 to 1.0)
        """
        if situation not in self.situation_outcomes:
            return 0.5  # Unknown situation

        outcomes = self.situation_outcomes[situation]
        if not outcomes:
            return 0.5

        successful = sum(1 for r in outcomes if r > 0)
        return successful / len(outcomes)

    def get_average_reward(self, situation: str) -> float:
        """
        Get average reward for a situation.

        Args:
            situation: Situation identifier

        Returns:
            Average reward
        """
        if situation not in self.situation_outcomes:
            return 0.0

        outcomes = self.situation_outcomes[situation]
        if not outcomes:
            return 0.0

        return sum(outcomes) / len(outcomes)

    def get_best_action(self, situation: str) -> Optional[str]:
        """
        Get the action with best average outcome for a situation.

        Args:
            situation: Situation identifier

        Returns:
            Best action or None
        """
        action_rewards: Dict[str, List[float]] = {}

        for entry in self.experiences:
            if entry.situation == situation:
                if entry.action not in action_rewards:
                    action_rewards[entry.action] = []
                action_rewards[entry.action].append(entry.reward)

        if not action_rewards:
            return None

        # Find action with highest average reward
        best_action = None
        best_avg = float('-inf')

        for action, rewards in action_rewards.items():
            avg = sum(rewards) / len(rewards)
            if avg > best_avg:
                best_avg = avg
                best_action = action

        return best_action

    def get_recent_experiences(self, count: int = 10) -> List[ExperienceEntry]:
        """
        Get most recent experiences.

        Args:
            count: Number of experiences to retrieve

        Returns:
            List of recent experiences
        """
        return list(self.experiences)[-count:]
