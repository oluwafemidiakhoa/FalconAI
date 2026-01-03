"""
Machine learning-based decision cores.
"""

import numpy as np
from typing import Any, Optional, Dict, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from ..decision.base import DecisionCore, Decision, ActionType
from ..perception.base import Event


class MLDecisionCore(DecisionCore):
    """
    Machine learning-based decision making using ensemble methods.

    Uses Random Forest or Gradient Boosting to learn optimal actions.
    """

    def __init__(self,
                 model_type: str = 'random_forest',
                 n_estimators: int = 100,
                 min_training_samples: int = 50):
        """
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            n_estimators: Number of trees in ensemble
            min_training_samples: Minimum samples before using ML
        """
        super().__init__()
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.min_training_samples = min_training_samples

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([action.value for action in ActionType])

        self.is_fitted = False
        self.training_features = []
        self.training_labels = []

        # Feature importance tracking
        self.feature_importances = None

    def _extract_features(self, event: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract features from event and context"""
        features = []

        # Features from event
        if isinstance(event, Event):
            features.append(event.salience_score)
            features.append(1.0 if event.event_type.value == 'critical' else 0.0)
            features.append(1.0 if event.event_type.value == 'anomaly' else 0.0)
            features.append(1.0 if event.event_type.value == 'salient' else 0.0)
        else:
            # Fallback for non-Event inputs
            if isinstance(event, (int, float)):
                features.extend([float(event), 0.0, 0.0, 0.0])
            else:
                features.extend([0.5, 0.0, 0.0, 1.0])

        # Features from context
        if context:
            features.append(1.0 if context.get('urgency') == 'critical' else 0.0)
            features.append(1.0 if context.get('urgency') == 'high' else 0.0)
            features.append(float(context.get('confidence', 0.5)))
        else:
            features.extend([0.0, 0.0, 0.5])

        return np.array(features).reshape(1, -1)

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        features = self._extract_features(event, context)

        if not self.is_fitted or len(self.training_features) < self.min_training_samples:
            # Fallback to heuristic decision
            if isinstance(event, Event):
                salience = event.salience_score
            else:
                salience = 0.5

            if salience > 0.8:
                action = ActionType.ESCALATE
            elif salience > 0.6:
                action = ActionType.INTERVENE
            elif salience > 0.4:
                action = ActionType.ALERT
            else:
                action = ActionType.OBSERVE

            confidence = salience
            reasoning = "Heuristic decision (model not yet trained)"

        else:
            # Use ML model
            probabilities = self.model.predict_proba(features)[0]
            action_idx = np.argmax(probabilities)
            confidence = probabilities[action_idx]

            action_value = self.label_encoder.inverse_transform([action_idx])[0]
            action = ActionType(action_value)

            reasoning = f"ML decision ({self.model_type}): confidence={confidence:.2f}"

            # Update feature importances
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_

        self.total_confidence += confidence

        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'model_type': self.model_type,
                'is_ml_decision': self.is_fitted,
                'features': features.tolist()[0]
            }
        )

    def train(self, features: List[np.ndarray], actions: List[ActionType]):
        """
        Train the model on historical data.

        Args:
            features: List of feature arrays
            actions: List of actions taken
        """
        X = np.vstack(features)
        y = self.label_encoder.transform([action.value for action in actions])

        self.model.fit(X, y)
        self.is_fitted = True

    def update(self, event: Any, action: ActionType, context: Optional[Dict[str, Any]] = None):
        """
        Add training example for online learning.

        Args:
            event: The event
            action: The action that was successful
            context: Context
        """
        features = self._extract_features(event, context)
        self.training_features.append(features)
        self.training_labels.append(action)

        # Retrain periodically
        if len(self.training_features) >= self.min_training_samples and \
           len(self.training_features) % 20 == 0:
            self.train(self.training_features, self.training_labels)

            # Keep recent history
            self.training_features = self.training_features[-200:]
            self.training_labels = self.training_labels[-200:]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances from the model"""
        return self.feature_importances


class EnsembleDecision(DecisionCore):
    """
    Ensemble of multiple decision cores.

    Combines predictions from multiple models for robust decisions.
    """

    def __init__(self, decision_cores: List[DecisionCore], voting: str = 'soft'):
        """
        Args:
            decision_cores: List of decision cores to ensemble
            voting: 'hard' for majority vote, 'soft' for weighted average
        """
        super().__init__()
        self.decision_cores = decision_cores
        self.voting = voting

    def decide(self, event: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        self.decisions_made += 1

        # Get decisions from all cores
        decisions = [core.decide(event, context) for core in self.decision_cores]

        if self.voting == 'hard':
            # Majority vote
            action_counts = defaultdict(int)
            for dec in decisions:
                action_counts[dec.action] += 1

            final_action = max(action_counts.items(), key=lambda x: x[1])[0]
            confidence = action_counts[final_action] / len(decisions)
            reasoning = f"Ensemble vote: {action_counts[final_action]}/{len(decisions)} cores agreed"

        else:  # soft voting
            # Weighted average by confidence
            action_scores = defaultdict(float)
            total_confidence = 0.0

            for dec in decisions:
                action_scores[dec.action] += dec.confidence
                total_confidence += dec.confidence

            final_action = max(action_scores.items(), key=lambda x: x[1])[0]
            confidence = action_scores[final_action] / total_confidence if total_confidence > 0 else 0.5
            reasoning = f"Ensemble weighted vote across {len(decisions)} cores"

        self.total_confidence += confidence

        return Decision(
            action=final_action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'ensemble_size': len(self.decision_cores),
                'voting': self.voting,
                'individual_decisions': [
                    {'action': d.action.value, 'confidence': d.confidence}
                    for d in decisions
                ]
            }
        )
