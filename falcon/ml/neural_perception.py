"""
Neural network-based perception engines.
"""

import numpy as np
from typing import Any, Optional, List
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque

from ..perception.base import PerceptionEngine, Event, EventType


class NeuralPerception(PerceptionEngine):
    """
    Neural network-based perception using sklearn MLPClassifier.

    Uses a neural network to classify inputs as salient or not.
    Can be pretrained or learn online.
    """

    def __init__(self,
                 input_dim: int = 10,
                 hidden_layers: tuple = (64, 32),
                 salience_threshold: float = 0.7,
                 pretrained_model: Optional[MLPClassifier] = None):
        """
        Args:
            input_dim: Dimension of input vectors
            hidden_layers: Sizes of hidden layers
            salience_threshold: Threshold for salient event detection
            pretrained_model: Optional pretrained sklearn MLPClassifier
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.salience_threshold = salience_threshold

        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=100,
                random_state=42
            )

        self.scaler = StandardScaler()
        self.is_fitted = False

        # Buffer for initial training
        self.training_buffer_x = []
        self.training_buffer_y = []
        self.min_training_samples = 50

    def _vectorize(self, data: Any) -> np.ndarray:
        """
        Convert input data to feature vector.

        Args:
            data: Input data (can be number, list, dict, etc.)

        Returns:
            Numpy array of features
        """
        if isinstance(data, (int, float)):
            # Scalar input - create features
            features = [data] + [0.0] * (self.input_dim - 1)
        elif isinstance(data, (list, tuple, np.ndarray)):
            features = list(data)[:self.input_dim]
            # Pad if needed
            features += [0.0] * (self.input_dim - len(features))
        elif isinstance(data, dict):
            # Extract numeric values
            features = [v for v in data.values() if isinstance(v, (int, float))][:self.input_dim]
            features += [0.0] * (self.input_dim - len(features))
        else:
            # Fallback
            features = [hash(str(data)) % 1000 / 1000.0] + [0.0] * (self.input_dim - 1)

        return np.array(features[:self.input_dim]).reshape(1, -1)

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        features = self._vectorize(data)

        if not self.is_fitted:
            # Not yet trained - use heuristic
            salience = float(np.mean(np.abs(features)))
        else:
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[0]

            # Probability of class 1 (salient)
            salience = probabilities[1] if len(probabilities) > 1 else probabilities[0]

        if salience > self.salience_threshold:
            self.events_triggered += 1

            # Determine event type based on salience level
            if salience > 0.95:
                event_type = EventType.CRITICAL
            elif salience > 0.85:
                event_type = EventType.SALIENT
            else:
                event_type = EventType.SALIENT

            return Event(
                data=data,
                event_type=event_type,
                salience_score=salience,
                metadata={'neural_confidence': salience}
            )

        return None

    def train(self, X: List[Any], y: List[int]):
        """
        Train the neural network.

        Args:
            X: List of input data
            y: List of labels (0=normal, 1=salient)
        """
        # Vectorize all inputs
        X_vec = np.vstack([self._vectorize(x) for x in X])

        # Fit scaler
        self.scaler.fit(X_vec)
        X_scaled = self.scaler.transform(X_vec)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def update(self, data: Any, is_salient: bool):
        """
        Add training sample for online learning.

        Args:
            data: Input data
            is_salient: Whether this data was salient
        """
        self.training_buffer_x.append(data)
        self.training_buffer_y.append(1 if is_salient else 0)

        # Train when we have enough samples
        if len(self.training_buffer_x) >= self.min_training_samples:
            self.train(self.training_buffer_x, self.training_buffer_y)
            # Keep recent history
            self.training_buffer_x = self.training_buffer_x[-100:]
            self.training_buffer_y = self.training_buffer_y[-100:]


class OnlineNeuralPerception(PerceptionEngine):
    """
    Online learning neural perception.

    Continuously adapts to incoming data stream.
    """

    def __init__(self,
                 input_dim: int = 10,
                 window_size: int = 100,
                 adaptation_rate: float = 0.1):
        """
        Args:
            input_dim: Dimension of input vectors
            window_size: Size of sliding window for statistics
            adaptation_rate: How quickly to adapt threshold
        """
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        self.history = deque(maxlen=window_size)
        self.scaler = StandardScaler()
        self.threshold_mean = 0.0
        self.threshold_std = 1.0
        self.is_initialized = False

    def _vectorize(self, data: Any) -> np.ndarray:
        """Convert input to feature vector"""
        if isinstance(data, (int, float)):
            features = [data] + [0.0] * (self.input_dim - 1)
        elif isinstance(data, (list, tuple, np.ndarray)):
            features = list(data)[:self.input_dim]
            features += [0.0] * (self.input_dim - len(features))
        else:
            features = [hash(str(data)) % 1000 / 1000.0] + [0.0] * (self.input_dim - 1)

        return np.array(features[:self.input_dim])

    def perceive(self, data: Any) -> Optional[Event]:
        self.events_processed += 1

        features = self._vectorize(data)
        self.history.append(features)

        if len(self.history) < 20:
            # Not enough data yet
            return None

        # Update statistics
        history_array = np.array(list(self.history))
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0) + 1e-8

        # Calculate anomaly score
        deviation = np.abs(features - mean) / std
        anomaly_score = float(np.mean(deviation))

        # Adaptive threshold
        if not self.is_initialized:
            self.threshold_mean = anomaly_score
            self.threshold_std = 1.0
            self.is_initialized = True

        # Update threshold adaptively
        self.threshold_mean = (1 - self.adaptation_rate) * self.threshold_mean + \
                              self.adaptation_rate * anomaly_score

        # Trigger if significantly above adaptive threshold
        if anomaly_score > self.threshold_mean + 2 * self.threshold_std:
            self.events_triggered += 1

            salience = min(anomaly_score / (self.threshold_mean + 2 * self.threshold_std), 1.0)

            event_type = EventType.ANOMALY if anomaly_score > self.threshold_mean + 3 * self.threshold_std \
                else EventType.SALIENT

            return Event(
                data=data,
                event_type=event_type,
                salience_score=salience,
                metadata={
                    'anomaly_score': anomaly_score,
                    'threshold': self.threshold_mean + 2 * self.threshold_std
                }
            )

        return None
