"""
Base classes for Memory systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from enum import Enum


class MemoryType(Enum):
    """Types of memory"""
    INSTINCT = "instinct"      # Pretrained, general knowledge
    EXPERIENCE = "experience"  # Learned from outcomes
    EPISODIC = "episodic"      # Specific events
    SEMANTIC = "semantic"      # Factual knowledge


class Memory(ABC):
    """
    Abstract base class for memory systems.

    Memory systems store and retrieve information to inform decisions.
    """

    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type

    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store information in memory.

        Args:
            key: Identifier for the memory
            value: The information to store
            metadata: Additional metadata
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve information from memory.

        Args:
            key: Identifier for the memory

        Returns:
            Retrieved value or None if not found
        """
        pass

    @abstractmethod
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Any]:
        """
        Search memory based on query.

        Args:
            query: Search criteria
            limit: Maximum results to return

        Returns:
            List of matching results
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of items in memory"""
        pass
