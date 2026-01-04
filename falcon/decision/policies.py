"""
Concrete implementations of decision cores.
"""

from typing import Any, Optional, Dict, Callable, List, Tuple
import random

from .base import DecisionCore, Decision, ActionType
from ..perception.base import Event, EventType


class HeuristicDecision(DecisionCore):
    """
    Simple heuristic-based decision making.
    Uses event type and salience to determine action.
    """

    def __init__(self, confidence_boost: float = 0.1):
        """
        Args:
            confidence_boost: Amount to boost confidence for known patterns
        """
        super().__init__()
        self.confidence_boost = confidence_boost

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        # Handle Event objects from perception layer
        if isinstance(event, Event):
            event_type = event.event_type
            salience = event.salience_score
        else:
            # Fallback for raw data
            event_type = EventType.NORMAL
            salience = 0.5

        # Fast heuristic decision tree
        if event_type == EventType.CRITICAL:
            action = ActionType.ESCALATE
            confidence = min(0.9 * salience, 1.0)
            reasoning = "Critical event detected - escalating"

        elif event_type == EventType.ANOMALY:
            action = ActionType.ALERT
            confidence = 0.7 * salience
            reasoning = "Anomaly detected - raising alert"

        elif event_type == EventType.SALIENT:
            action = ActionType.INTERVENE if salience > 0.7 else ActionType.ALERT
            confidence = 0.6 * salience
            reasoning = f"Salient event - {'intervening' if salience > 0.7 else 'alerting'}"

        else:
            action = ActionType.OBSERVE
            confidence = 0.5
            reasoning = "Normal event - continuing observation"

        self.total_confidence += confidence

        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={'event_type': event_type.value if isinstance(event_type, EventType) else 'unknown'}
        )


class ThresholdDecision(DecisionCore):
    """
    Threshold-based decision making with configurable levels.
    """

    def __init__(self,
                 alert_threshold: float = 0.5,
                 intervene_threshold: float = 0.7,
                 escalate_threshold: float = 0.9):
        """
        Args:
            alert_threshold: Salience threshold for alerts
            intervene_threshold: Salience threshold for intervention
            escalate_threshold: Salience threshold for escalation
        """
        super().__init__()
        self.alert_threshold = alert_threshold
        self.intervene_threshold = intervene_threshold
        self.escalate_threshold = escalate_threshold

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        salience = event.salience_score if isinstance(event, Event) else 0.5

        if salience >= self.escalate_threshold:
            action = ActionType.ESCALATE
            reasoning = f"Salience {salience:.2f} exceeds escalate threshold"
        elif salience >= self.intervene_threshold:
            action = ActionType.INTERVENE
            reasoning = f"Salience {salience:.2f} exceeds intervene threshold"
        elif salience >= self.alert_threshold:
            action = ActionType.ALERT
            reasoning = f"Salience {salience:.2f} exceeds alert threshold"
        else:
            action = ActionType.OBSERVE
            reasoning = f"Salience {salience:.2f} below alert threshold"

        confidence = salience
        self.total_confidence += confidence

        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning
        )


class RuleBasedDecision(DecisionCore):
    """
    Rule-based decision making with custom rules.
    """

    def __init__(self):
        super().__init__()
        self.rules: List[Tuple[Callable, ActionType, str]] = []

    def add_rule(self, condition: Callable[[Any], bool], action: ActionType, reasoning: str):
        """
        Add a decision rule.

        Args:
            condition: Function that returns True if rule applies
            action: Action to take if rule matches
            reasoning: Explanation for the rule
        """
        self.rules.append((condition, action, reasoning))

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        # Evaluate rules in order
        for condition, action, reasoning in self.rules:
            try:
                if condition(event):
                    confidence = 0.8  # High confidence for explicit rules
                    self.total_confidence += confidence

                    return Decision(
                        action=action,
                        confidence=confidence,
                        reasoning=reasoning
                    )
            except Exception:
                continue

        # Default action if no rules match
        confidence = 0.3
        self.total_confidence += confidence

        return Decision(
            action=ActionType.OBSERVE,
            confidence=confidence,
            reasoning="No rules matched - defaulting to observe"
        )


class HybridDecision(DecisionCore):
    """
    Hybrid decision making combining heuristics and learned patterns.
    """

    def __init__(self, use_learned_patterns: bool = True):
        """
        Args:
            use_learned_patterns: Whether to incorporate learned patterns
        """
        super().__init__()
        self.use_learned_patterns = use_learned_patterns
        self.pattern_weights: Dict[str, float] = {}

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        # Fast heuristic baseline
        if isinstance(event, Event):
            salience = event.salience_score
            event_type = event.event_type
        else:
            salience = 0.5
            event_type = EventType.NORMAL

        # Determine base action
        if salience > 0.8:
            base_action = ActionType.ESCALATE
        elif salience > 0.6:
            base_action = ActionType.INTERVENE
        elif salience > 0.4:
            base_action = ActionType.ALERT
        else:
            base_action = ActionType.OBSERVE

        base_confidence = salience

        # Adjust based on learned patterns if available
        if self.use_learned_patterns and context:
            pattern_key = f"{event_type.value if isinstance(event_type, EventType) else 'unknown'}_{base_action.value}"
            if pattern_key in self.pattern_weights:
                weight = self.pattern_weights[pattern_key]
                base_confidence = base_confidence * 0.7 + weight * 0.3

        confidence = min(base_confidence, 1.0)
        self.total_confidence += confidence

        return Decision(
            action=base_action,
            confidence=confidence,
            reasoning=f"Hybrid decision: salience={salience:.2f}, learned_boost={'yes' if self.use_learned_patterns else 'no'}"
        )

    def update_pattern(self, pattern_key: str, success: bool):
        """
        Update learned pattern weights based on outcomes.

        Args:
            pattern_key: Key identifying the pattern
            success: Whether the decision was successful
        """
        if pattern_key not in self.pattern_weights:
            self.pattern_weights[pattern_key] = 0.5

        # Simple update: move towards 1.0 for success, 0.0 for failure
        learning_rate = 0.1
        target = 1.0 if success else 0.0
        self.pattern_weights[pattern_key] += learning_rate * (target - self.pattern_weights[pattern_key])


class MemoryAwareDecision(DecisionCore):
    """
    Decision core that incorporates memory and swarm context signals.
    """

    def __init__(self, memory_weight: float = 0.35, min_samples: int = 3):
        """
        Args:
            memory_weight: How strongly to weight memory signals (0.0 to 1.0)
            min_samples: Minimum samples needed to trust swarm memory
        """
        super().__init__()
        self.memory_weight = memory_weight
        self.min_samples = min_samples

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        if isinstance(event, Event):
            event_type = event.event_type
            salience = event.salience_score
        else:
            event_type = EventType.NORMAL
            salience = 0.5

        # Base heuristic decision
        if event_type == EventType.CRITICAL:
            base_action = ActionType.ESCALATE
            base_confidence = min(0.9 * salience, 1.0)
            reasoning = "Critical event detected - escalating"
        elif event_type == EventType.ANOMALY:
            base_action = ActionType.ALERT
            base_confidence = 0.7 * salience
            reasoning = "Anomaly detected - raising alert"
        elif event_type == EventType.SALIENT:
            base_action = ActionType.INTERVENE if salience > 0.7 else ActionType.ALERT
            base_confidence = 0.6 * salience
            reasoning = "Salient event - memory-aware decision"
        else:
            base_action = ActionType.OBSERVE
            base_confidence = 0.5
            reasoning = "Normal event - continuing observation"

        memory_action = None
        memory_confidence = None
        memory_samples = 0

        def _coerce_action(value: Any) -> Optional[ActionType]:
            if isinstance(value, ActionType):
                return value
            if isinstance(value, str):
                try:
                    return ActionType(value)
                except Exception:
                    return None
            return None

        if context:
            past_experience = context.get("past_experience")
            if past_experience is not None:
                reward = getattr(past_experience, "reward", None)
                action = getattr(past_experience, "action", None)
                if reward is not None and action is not None:
                    memory_samples = 1
                    memory_action = _coerce_action(action)
                    memory_confidence = max(min(0.5 + float(reward), 1.0), 0.0)

            swarm_experiences = context.get("swarm_experiences") or []
            if swarm_experiences:
                action_rewards: Dict[str, List[float]] = {}
                for exp in swarm_experiences:
                    exp_action = getattr(exp, "action", None)
                    exp_reward = getattr(exp, "reward", None)
                    if exp_action is None or exp_reward is None:
                        continue
                    action_rewards.setdefault(str(exp_action), []).append(float(exp_reward))

                if action_rewards:
                    memory_samples = max(memory_samples, max(len(v) for v in action_rewards.values()))
                    best_action = None
                    best_avg = float("-inf")
                    for act, rewards in action_rewards.items():
                        avg_reward = sum(rewards) / len(rewards)
                        if avg_reward > best_avg:
                            best_avg = avg_reward
                            best_action = act

                    if best_action is not None and memory_samples >= self.min_samples:
                        memory_action = _coerce_action(best_action)
                        memory_confidence = max(min(0.5 + best_avg, 1.0), 0.0)

        action = base_action
        confidence = base_confidence

        if memory_action is not None and memory_confidence is not None:
            if memory_action != base_action and memory_confidence > 0.55:
                action = memory_action
                reasoning = "Memory override based on shared experience"
            confidence = (1 - self.memory_weight) * base_confidence + self.memory_weight * memory_confidence

        self.total_confidence += confidence

        return Decision(
            action=action,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            metadata={
                "event_type": event_type.value if isinstance(event_type, EventType) else "unknown",
                "memory_action": memory_action.value if memory_action else None,
                "memory_confidence": memory_confidence,
                "memory_samples": memory_samples,
            },
        )
