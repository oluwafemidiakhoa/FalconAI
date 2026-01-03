"""
Distributed multi-agent FALCON swarm.

Multiple FALCON agents working together with shared knowledge.
"""

from .swarm import (
    FalconSwarm,
    SwarmCoordinator,
    SharedExperiencePool,
    ConsensusDecision
)

__all__ = [
    'FalconSwarm',
    'SwarmCoordinator',
    'SharedExperiencePool',
    'ConsensusDecision'
]
