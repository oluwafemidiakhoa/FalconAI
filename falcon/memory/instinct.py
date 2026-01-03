"""
Instinct Memory - pretrained general knowledge.
"""

from typing import Any, Optional, Dict, List

from .base import Memory, MemoryType


class InstinctMemory(Memory):
    """
    Instinct memory represents pretrained, general knowledge.

    This is analogous to a foundation model or innate patterns.
    In a real system, this could interface with a pretrained model.
    """

    def __init__(self, pretrained_patterns: Optional[Dict[str, Any]] = None):
        """
        Args:
            pretrained_patterns: Dictionary of pretrained patterns/knowledge
        """
        super().__init__(MemoryType.INSTINCT)
        self.patterns = pretrained_patterns or {}
        self._load_default_patterns()

    def _load_default_patterns(self):
        """Load some default instinctive patterns"""
        if not self.patterns:
            self.patterns = {
                'high_salience': {
                    'response': 'alert',
                    'confidence': 0.8,
                    'description': 'High salience events require alerting'
                },
                'critical_event': {
                    'response': 'escalate',
                    'confidence': 0.9,
                    'description': 'Critical events should be escalated'
                },
                'anomaly': {
                    'response': 'investigate',
                    'confidence': 0.7,
                    'description': 'Anomalies should be investigated'
                },
                'normal_event': {
                    'response': 'observe',
                    'confidence': 0.6,
                    'description': 'Normal events only need observation'
                }
            }

    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a pattern in instinct memory.

        Note: Instinct memory is typically pretrained and doesn't change much.
        """
        self.patterns[key] = {
            'value': value,
            'metadata': metadata or {}
        }

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a pattern by key"""
        return self.patterns.get(key)

    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Any]:
        """
        Search for relevant patterns.

        Args:
            query: Search criteria (e.g., {'event_type': 'anomaly'})
            limit: Maximum results

        Returns:
            List of matching patterns
        """
        results = []

        for key, pattern in self.patterns.items():
            match = True
            for query_key, query_value in query.items():
                if query_key in pattern:
                    if pattern[query_key] != query_value:
                        match = False
                        break

            if match:
                results.append({'key': key, 'pattern': pattern})
                if len(results) >= limit:
                    break

        return results

    def size(self) -> int:
        """Return number of patterns"""
        return len(self.patterns)

    def get_response_for_pattern(self, pattern_name: str) -> Optional[str]:
        """Get recommended response for a pattern"""
        pattern = self.patterns.get(pattern_name)
        if pattern:
            if isinstance(pattern, dict) and 'response' in pattern:
                return pattern['response']
            elif isinstance(pattern, dict) and 'value' in pattern:
                value = pattern['value']
                if isinstance(value, dict) and 'response' in value:
                    return value['response']
        return None
