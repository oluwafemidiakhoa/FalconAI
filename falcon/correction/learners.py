"""
Concrete implementations of correction loops.
"""

from typing import Any, Optional, Dict, Tuple
from collections import defaultdict, deque
import numpy as np

from .base import CorrectionLoop, Outcome, OutcomeType
from ..decision.base import Decision, ActionType


class OutcomeBasedCorrection(CorrectionLoop):
    """
    Simple outcome-based correction using running averages.
    """

    def __init__(self, learning_rate: float = 0.1, abort_threshold: float = -0.5):
        """
        Args:
            learning_rate: How quickly to adapt to new outcomes
            abort_threshold: Reward threshold below which to abort
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.abort_threshold = abort_threshold
        self.action_rewards: Dict[str, float] = defaultdict(float)
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.recent_rewards = deque(maxlen=10)

    def observe(self, decision: Any, outcome: Outcome, context: Optional[Dict[str, Any]] = None):
        self.corrections_made += 1
        self.total_reward += outcome.reward

        if isinstance(decision, Decision):
            action_key = decision.action.value
        else:
            action_key = "unknown"

        # Update running average for this action type
        self.action_counts[action_key] += 1
        old_avg = self.action_rewards[action_key]
        self.action_rewards[action_key] = old_avg + self.learning_rate * (outcome.reward - old_avg)

        self.recent_rewards.append(outcome.reward)

    def should_abort(self, current_state: Any) -> bool:
        """Abort if recent average reward is too low"""
        if len(self.recent_rewards) < 3:
            return False

        avg_recent = np.mean(self.recent_rewards)
        return avg_recent < self.abort_threshold

    def get_correction_signal(self) -> Dict[str, Any]:
        """Return action performance metrics"""
        return {
            'action_rewards': dict(self.action_rewards),
            'action_counts': dict(self.action_counts),
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'should_abort': self.should_abort(None)
        }


class BayesianCorrection(CorrectionLoop):
    """
    Bayesian correction using beta distributions for success rates.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Args:
            prior_alpha: Prior successes (beta distribution)
            prior_beta: Prior failures (beta distribution)
        """
        super().__init__()
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        # Track alpha and beta for each action type
        self.action_alphas: Dict[str, float] = {}
        self.action_betas: Dict[str, float] = {}

    def observe(self, decision: Any, outcome: Outcome, context: Optional[Dict[str, Any]] = None):
        self.corrections_made += 1
        self.total_reward += outcome.reward

        if isinstance(decision, Decision):
            action_key = decision.action.value
        else:
            action_key = "unknown"

        # Initialize if not present
        if action_key not in self.action_alphas:
            self.action_alphas[action_key] = self.prior_alpha
            self.action_betas[action_key] = self.prior_beta

        # Update beta distribution
        if outcome.is_successful():
            self.action_alphas[action_key] += 1
        elif outcome.is_failure():
            self.action_betas[action_key] += 1
        else:
            # Partial outcome - split the update
            self.action_alphas[action_key] += 0.5
            self.action_betas[action_key] += 0.5

    def should_abort(self, current_state: Any) -> bool:
        """Abort if overall success probability is too low"""
        total_alpha = sum(self.action_alphas.values())
        total_beta = sum(self.action_betas.values())

        if total_alpha + total_beta < 10:
            return False

        success_prob = total_alpha / (total_alpha + total_beta)
        return success_prob < 0.3

    def get_correction_signal(self) -> Dict[str, Any]:
        """Return Bayesian estimates of action success rates"""
        success_rates = {}
        for action_key in self.action_alphas.keys():
            alpha = self.action_alphas[action_key]
            beta = self.action_betas[action_key]
            success_rates[action_key] = alpha / (alpha + beta)

        return {
            'success_rates': success_rates,
            'action_alphas': dict(self.action_alphas),
            'action_betas': dict(self.action_betas)
        }

    def get_action_confidence(self, action_key: str) -> Tuple[float, float]:
        """
        Get mean and variance for action success rate.

        Returns:
            Tuple of (mean, variance)
        """
        alpha = self.action_alphas[action_key]
        beta = self.action_betas[action_key]

        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        return mean, variance


class ReinforcementCorrection(CorrectionLoop):
    """
    Reinforcement learning-based correction with Q-value updates.
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values: Dict[Tuple[str, str], float] = defaultdict(float)
        self.last_state_action: Optional[Tuple[str, str]] = None

    def observe(self, decision: Any, outcome: Outcome, context: Optional[Dict[str, Any]] = None):
        self.corrections_made += 1
        self.total_reward += outcome.reward

        if isinstance(decision, Decision):
            action = decision.action.value
        else:
            action = "unknown"

        # Extract state representation
        state = self._get_state_key(context)
        state_action = (state, action)

        # Q-learning update
        if self.last_state_action is not None:
            # Get max Q-value for current state
            current_state_actions = [key for key in self.q_values.keys() if key[0] == state]
            if current_state_actions:
                max_q = max(self.q_values[sa] for sa in current_state_actions)
            else:
                max_q = 0.0

            # Update Q-value for last state-action
            old_q = self.q_values[self.last_state_action]
            self.q_values[self.last_state_action] = old_q + self.learning_rate * (
                outcome.reward + self.discount_factor * max_q - old_q
            )

        self.last_state_action = state_action

    def should_abort(self, current_state: Any) -> bool:
        """Abort if all Q-values for current state are negative"""
        state_key = self._get_state_key({'current_state': current_state})
        state_actions = [key for key in self.q_values.keys() if key[0] == state_key]

        if not state_actions:
            return False

        max_q = max(self.q_values[sa] for sa in state_actions)
        return max_q < -0.5

    def get_correction_signal(self) -> Dict[str, Any]:
        """Return Q-values and best actions"""
        # Group Q-values by state
        state_best_actions = {}
        for (state, action), q_value in self.q_values.items():
            if state not in state_best_actions or q_value > state_best_actions[state][1]:
                state_best_actions[state] = (action, q_value)

        return {
            'q_values': dict(self.q_values),
            'best_actions': state_best_actions,
            'total_states': len(set(k[0] for k in self.q_values.keys()))
        }

    def _get_state_key(self, context: Optional[Dict[str, Any]]) -> str:
        """Extract state representation from context"""
        if context is None:
            return "default"

        # Simple state representation
        if 'state' in context:
            return str(context['state'])
        elif 'current_state' in context:
            return str(context['current_state'])
        else:
            return "default"
