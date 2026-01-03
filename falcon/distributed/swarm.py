"""
Multi-agent swarm intelligence for FALCON-AI.

Enables multiple FALCON agents to work together, share knowledge,
and make coordinated decisions.
"""

import threading
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import time
import uuid

from ..core import FalconAI
from ..decision.base import Decision, ActionType
from ..correction.base import Outcome
from ..memory.experience import ExperienceEntry


@dataclass
class SwarmMessage:
    """Message passed between agents in the swarm"""
    sender_id: str
    message_type: str  # 'experience', 'decision_request', 'consensus_vote'
    payload: Any
    timestamp: float


class SharedExperiencePool:
    """
    Shared memory pool for swarm agents.

    Agents can contribute and query collective experience.
    """

    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: Maximum experiences to store
        """
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.lock = threading.Lock()

        # Index for fast lookup
        self.situation_index: Dict[str, List[ExperienceEntry]] = defaultdict(list)

    def add_experience(self, agent_id: str, experience: ExperienceEntry):
        """
        Add experience from an agent.

        Args:
            agent_id: ID of the contributing agent
            experience: The experience to share
        """
        with self.lock:
            # Add source agent info
            experience.metadata['source_agent'] = agent_id
            experience.metadata['shared_time'] = time.time()

            self.experiences.append(experience)

            # Update index
            self.situation_index[experience.situation].append(experience)

            # Limit index size
            if len(self.situation_index[experience.situation]) > 100:
                self.situation_index[experience.situation] = \
                    self.situation_index[experience.situation][-100:]

    def query_experiences(self,
                         situation: Optional[str] = None,
                         min_reward: Optional[float] = None,
                         limit: int = 10) -> List[ExperienceEntry]:
        """
        Query experiences from the pool.

        Args:
            situation: Filter by situation
            min_reward: Filter by minimum reward
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        with self.lock:
            if situation and situation in self.situation_index:
                candidates = self.situation_index[situation]
            else:
                candidates = list(self.experiences)

            # Filter by reward if specified
            if min_reward is not None:
                candidates = [e for e in candidates if e.reward >= min_reward]

            # Return most recent
            return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_best_action(self, situation: str) -> Optional[str]:
        """
        Get the best action for a situation based on swarm experience.

        Args:
            situation: The situation to query

        Returns:
            Best action or None
        """
        with self.lock:
            if situation not in self.situation_index:
                return None

            experiences = self.situation_index[situation]
            if not experiences:
                return None

            # Group by action and calculate average reward
            action_rewards = defaultdict(list)
            for exp in experiences:
                action_rewards[exp.action].append(exp.reward)

            # Find best action
            best_action = None
            best_avg_reward = float('-inf')

            for action, rewards in action_rewards.items():
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_action = action

            return best_action

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the shared pool"""
        with self.lock:
            return {
                'total_experiences': len(self.experiences),
                'unique_situations': len(self.situation_index),
                'avg_reward': sum(e.reward for e in self.experiences) / max(len(self.experiences), 1)
            }


class ConsensusDecision:
    """
    Consensus mechanism for swarm decision making.

    Multiple agents vote on the best action.
    """

    def __init__(self, voting_method: str = 'weighted'):
        """
        Args:
            voting_method: 'majority', 'weighted', or 'unanimous'
        """
        self.voting_method = voting_method

    def reach_consensus(self, decisions: List[Decision]) -> Decision:
        """
        Reach consensus from multiple agent decisions.

        Args:
            decisions: List of decisions from different agents

        Returns:
            Consensus decision
        """
        if not decisions:
            raise ValueError("No decisions to consensus")

        if self.voting_method == 'majority':
            # Simple majority vote
            action_counts = defaultdict(int)
            for dec in decisions:
                action_counts[dec.action] += 1

            best_action = max(action_counts.items(), key=lambda x: x[1])[0]
            confidence = action_counts[best_action] / len(decisions)
            reasoning = f"Majority consensus: {action_counts[best_action]}/{len(decisions)} votes"

        elif self.voting_method == 'weighted':
            # Weighted by confidence
            action_scores = defaultdict(float)
            total_confidence = sum(d.confidence for d in decisions)

            for dec in decisions:
                action_scores[dec.action] += dec.confidence

            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            confidence = action_scores[best_action] / total_confidence if total_confidence > 0 else 0.5
            reasoning = f"Weighted consensus across {len(decisions)} agents"

        elif self.voting_method == 'unanimous':
            # All agents must agree
            actions = [d.action for d in decisions]
            if len(set(actions)) == 1:
                best_action = actions[0]
                confidence = sum(d.confidence for d in decisions) / len(decisions)
                reasoning = "Unanimous agreement"
            else:
                # Fallback to weighted
                action_scores = defaultdict(float)
                for dec in decisions:
                    action_scores[dec.action] += dec.confidence

                best_action = max(action_scores.items(), key=lambda x: x[1])[0]
                confidence = 0.3  # Low confidence without unanimity
                reasoning = f"No unanimity - fallback to most confident"

        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

        return Decision(
            action=best_action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'consensus_method': self.voting_method,
                'num_agents': len(decisions),
                'individual_votes': [
                    {'action': d.action.value, 'confidence': d.confidence}
                    for d in decisions
                ]
            }
        )


class FalconSwarm:
    """
    Swarm of FALCON agents working together.

    Features:
    - Shared experience pool
    - Coordinated decision making
    - Load balancing
    - Collective learning
    """

    def __init__(self,
                 num_agents: int = 5,
                 agent_factory: Optional[Callable[[], FalconAI]] = None,
                 consensus_method: str = 'weighted'):
        """
        Args:
            num_agents: Number of agents in the swarm
            agent_factory: Factory function to create agents
            consensus_method: How to reach consensus
        """
        self.num_agents = num_agents
        self.consensus_method = consensus_method

        # Create agents
        if agent_factory:
            self.agents = [agent_factory() for _ in range(num_agents)]
        else:
            # Need agent_factory for proper initialization
            raise ValueError("agent_factory is required to create swarm agents")

        self.agent_ids = [str(uuid.uuid4())[:8] for _ in range(num_agents)]

        # Shared components
        self.shared_pool = SharedExperiencePool()
        self.consensus = ConsensusDecision(voting_method=consensus_method)

        # Statistics
        self.total_decisions = 0
        self.consensus_history = []

    def process(self,
                data: Any,
                context: Optional[Dict[str, Any]] = None,
                use_consensus: bool = True) -> Optional[Decision]:
        """
        Process data through the swarm.

        Args:
            data: Input data
            context: Optional context
            use_consensus: Whether to use consensus decision

        Returns:
            Swarm decision (or None if not salient)
        """
        # Each agent perceives and decides
        decisions = []

        for i, agent in enumerate(self.agents):
            # Add shared knowledge to context
            enhanced_context = context.copy() if context else {}
            enhanced_context['agent_id'] = self.agent_ids[i]

            # Query shared experiences
            if hasattr(agent.memory, 'retrieve'):
                shared_exp = self.shared_pool.query_experiences(limit=5)
                enhanced_context['swarm_experiences'] = shared_exp

            # Agent processes data
            decision = agent.process(data, enhanced_context)

            if decision:
                decisions.append(decision)

        if not decisions:
            return None

        # Reach consensus or pick best
        if use_consensus and len(decisions) > 1:
            final_decision = self.consensus.reach_consensus(decisions)
        else:
            # Pick most confident
            final_decision = max(decisions, key=lambda d: d.confidence)

        self.total_decisions += 1
        self.consensus_history.append({
            'num_votes': len(decisions),
            'consensus_confidence': final_decision.confidence,
            'timestamp': time.time()
        })

        return final_decision

    def observe(self,
                decision: Decision,
                outcome: Outcome,
                agent_idx: Optional[int] = None):
        """
        Share outcome with the swarm.

        Args:
            decision: The decision that was made
            outcome: The observed outcome
            agent_idx: Which agent to update (None = all)
        """
        # Create experience
        experience = ExperienceEntry(
            situation=decision.action.value,
            action=decision.action.value,
            outcome=outcome.outcome_type.value,
            reward=outcome.reward
        )

        # Share with pool
        source_id = self.agent_ids[agent_idx] if agent_idx is not None else 'swarm'
        self.shared_pool.add_experience(source_id, experience)

        # Update agent(s)
        agents_to_update = [agent_idx] if agent_idx is not None else range(len(self.agents))

        for idx in agents_to_update:
            self.agents[idx].observe(decision, outcome)

            # Also update agent's memory from shared pool
            if self.agents[idx].memory:
                self.agents[idx].memory.store(
                    decision.action.value,
                    experience
                )

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics"""
        individual_stats = [agent.get_status() for agent in self.agents]

        return {
            'num_agents': self.num_agents,
            'total_decisions': self.total_decisions,
            'shared_pool': self.shared_pool.get_stats(),
            'consensus_method': self.consensus_method,
            'average_confidence': sum(
                s['decision']['average_confidence']
                for s in individual_stats
            ) / len(individual_stats),
            'individual_agents': individual_stats
        }

    def get_load_distribution(self) -> Dict[str, Any]:
        """Analyze load distribution across agents"""
        return {
            'agent_loads': [
                {
                    'agent_id': self.agent_ids[i],
                    'events_processed': agent.perception.events_processed,
                    'decisions_made': agent.decision.decisions_made
                }
                for i, agent in enumerate(self.agents)
            ],
            'total_swarm_load': sum(
                agent.perception.events_processed
                for agent in self.agents
            )
        }


class SwarmCoordinator:
    """
    Coordinates multiple swarms for hierarchical organization.
    """

    def __init__(self):
        self.swarms: Dict[str, FalconSwarm] = {}
        self.routing_rules: Dict[str, Callable] = {}

    def add_swarm(self, name: str, swarm: FalconSwarm):
        """Add a swarm to the coordinator"""
        self.swarms[name] = swarm

    def route_to_swarm(self, data: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine which swarm should handle the data.

        Args:
            data: Input data
            context: Optional context

        Returns:
            Name of the swarm to route to
        """
        # Default: route to first swarm
        if not self.swarms:
            raise ValueError("No swarms available")

        # Apply routing rules if configured
        for rule_name, rule_func in self.routing_rules.items():
            try:
                swarm_name = rule_func(data, context)
                if swarm_name in self.swarms:
                    return swarm_name
            except Exception:
                continue

        # Fallback: round-robin or least loaded
        swarm_loads = {
            name: sum(agent.perception.events_processed for agent in swarm.agents)
            for name, swarm in self.swarms.items()
        }

        return min(swarm_loads.items(), key=lambda x: x[1])[0]

    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Optional[Decision]:
        """
        Process data through appropriate swarm.

        Args:
            data: Input data
            context: Optional context

        Returns:
            Decision from routed swarm
        """
        swarm_name = self.route_to_swarm(data, context)
        return self.swarms[swarm_name].process(data, context)
