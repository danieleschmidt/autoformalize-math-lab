"""Reinforcement Learning Enhanced Formalization Pipeline.

This module implements a self-improving formalization pipeline that learns
from successes and failures using reinforcement learning techniques.
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

try:
    import gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    RL_AVAILABLE = True
except ImportError:
    gym = None
    torch = None
    nn = None
    optim = None
    RL_AVAILABLE = False

from .optimized_pipeline import OptimizedFormalizationPipeline, OptimizedFormalizationResult
from .config import FormalizationConfig
from ..utils.logging_config import setup_logger


class ActionType(Enum):
    """Types of actions the RL agent can take."""
    ADJUST_TEMPERATURE = "adjust_temperature"
    CHANGE_MODEL = "change_model"
    MODIFY_PROMPT = "modify_prompt"
    ADD_CONTEXT = "add_context"
    RETRY_STRATEGY = "retry_strategy"
    COMPLEXITY_ADJUSTMENT = "complexity_adjustment"


@dataclass
class RLState:
    """State representation for reinforcement learning."""
    latex_complexity: float
    domain_type: str
    previous_success_rate: float
    current_temperature: float
    model_confidence: float
    error_pattern: str
    attempt_count: int
    time_elapsed: float
    

@dataclass
class RLAction:
    """Action taken by the RL agent."""
    action_type: ActionType
    parameters: Dict[str, Any]
    confidence: float
    

@dataclass
class RLExperience:
    """Experience tuple for RL training."""
    state: RLState
    action: RLAction
    reward: float
    next_state: Optional[RLState]
    done: bool
    

class FormalizationRLAgent:
    """Reinforcement Learning agent for formalization optimization."""
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dim: int = 64,
        learning_rate: float = 0.001
    ):
        self.logger = setup_logger(__name__)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize neural networks if available
        self._initialize_networks(learning_rate)
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Performance tracking
        self.training_metrics = {
            "episodes_trained": 0,
            "average_reward": 0.0,
            "exploration_rate": self.epsilon,
            "policy_loss": 0.0,
            "value_loss": 0.0
        }
        
    def _initialize_networks(self, learning_rate: float):
        """Initialize neural networks for RL agent."""
        try:
            if RL_AVAILABLE and torch:
                # Policy network (Actor)
                self.policy_network = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.action_dim),
                    nn.Softmax(dim=-1)
                )
                
                # Value network (Critic)
                self.value_network = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1)
                )
                
                # Optimizers
                self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
                self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
                
                self.logger.info("RL networks initialized successfully")
            else:
                self.policy_network = None
                self.value_network = None
                self.policy_optimizer = None
                self.value_optimizer = None
                self.logger.warning("RL networks not available, using heuristic policy")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize RL networks: {e}")
            self.policy_network = None
            self.value_network = None
            
    def select_action(self, state: RLState) -> RLAction:
        """Select action using epsilon-greedy policy with neural network."""
        try:
            if self.policy_network and torch:
                state_tensor = self._state_to_tensor(state)
                
                # Epsilon-greedy exploration
                if np.random.random() < self.epsilon:
                    # Random exploration
                    action_idx = np.random.randint(0, self.action_dim)
                    action_probs = torch.zeros(self.action_dim)
                    action_probs[action_idx] = 1.0
                else:
                    # Policy-based action
                    with torch.no_grad():
                        action_probs = self.policy_network(state_tensor)
                        action_idx = torch.argmax(action_probs).item()
                
                # Convert action index to specific action
                return self._action_index_to_action(action_idx, state, action_probs[action_idx].item())
                
            else:
                # Heuristic policy when networks not available
                return self._heuristic_action_selection(state)
                
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return self._heuristic_action_selection(state)
            
    def _state_to_tensor(self, state: RLState) -> torch.Tensor:
        """Convert RLState to tensor representation."""
        state_vector = np.array([
            state.latex_complexity,
            hash(state.domain_type) % 100 / 100.0,  # Normalize domain hash
            state.previous_success_rate,
            state.current_temperature,
            state.model_confidence,
            hash(state.error_pattern) % 100 / 100.0,  # Normalize error hash
            min(state.attempt_count / 10.0, 1.0),  # Normalize attempts
            min(state.time_elapsed / 300.0, 1.0)   # Normalize time (5 min max)
        ])
        return torch.FloatTensor(state_vector)
        
    def _action_index_to_action(self, action_idx: int, state: RLState, confidence: float) -> RLAction:
        """Convert action index to specific RLAction."""
        action_types = list(ActionType)
        action_type = action_types[action_idx % len(action_types)]
        
        # Generate parameters based on action type and current state
        if action_type == ActionType.ADJUST_TEMPERATURE:
            delta = 0.1 if state.current_temperature < 0.5 else -0.1
            parameters = {"temperature_delta": delta}
            
        elif action_type == ActionType.CHANGE_MODEL:
            models = ["gpt-4", "claude-3", "gpt-3.5-turbo"]
            parameters = {"model_name": np.random.choice(models)}
            
        elif action_type == ActionType.MODIFY_PROMPT:
            modifications = ["add_examples", "simplify_language", "add_context", "be_more_specific"]
            parameters = {"modification_type": np.random.choice(modifications)}
            
        elif action_type == ActionType.ADD_CONTEXT:
            context_types = ["domain_specific", "mathematical_background", "proof_techniques"]
            parameters = {"context_type": np.random.choice(context_types)}
            
        elif action_type == ActionType.RETRY_STRATEGY:
            strategies = ["incremental", "complete_restart", "partial_modification"]
            parameters = {"strategy": np.random.choice(strategies)}
            
        else:  # COMPLEXITY_ADJUSTMENT
            adjustment = 0.1 if state.latex_complexity > 0.7 else -0.1
            parameters = {"complexity_adjustment": adjustment}
            
        return RLAction(
            action_type=action_type,
            parameters=parameters,
            confidence=confidence
        )
        
    def _heuristic_action_selection(self, state: RLState) -> RLAction:
        """Heuristic action selection when RL networks not available."""
        # Simple rule-based policy
        if state.previous_success_rate < 0.5:
            if state.current_temperature > 0.8:
                return RLAction(
                    action_type=ActionType.ADJUST_TEMPERATURE,
                    parameters={"temperature_delta": -0.2},
                    confidence=0.7
                )
            else:
                return RLAction(
                    action_type=ActionType.ADD_CONTEXT,
                    parameters={"context_type": "domain_specific"},
                    confidence=0.6
                )
        else:
            return RLAction(
                action_type=ActionType.COMPLEXITY_ADJUSTMENT,
                parameters={"complexity_adjustment": 0.1},
                confidence=0.5
            )
            
    def store_experience(self, experience: RLExperience):
        """Store experience in replay buffer."""
        self.experience_buffer.append(experience)
        
    def train(self) -> Dict[str, float]:
        """Train the RL agent using experience replay."""
        if len(self.experience_buffer) < self.batch_size:
            return self.training_metrics
            
        try:
            if not self.policy_network or not torch:
                # Mock training metrics when networks not available
                self.training_metrics["episodes_trained"] += 1
                self.training_metrics["average_reward"] = np.mean([exp.reward for exp in list(self.experience_buffer)[-100:]])
                return self.training_metrics
                
            # Sample batch from experience buffer
            batch = np.random.choice(list(self.experience_buffer), size=self.batch_size, replace=False)
            
            states = torch.stack([self._state_to_tensor(exp.state) for exp in batch])
            actions = torch.tensor([list(ActionType).index(exp.action.action_type) for exp in batch])
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
            next_states = torch.stack([
                self._state_to_tensor(exp.next_state) if exp.next_state else torch.zeros(self.state_dim) 
                for exp in batch
            ])
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool)
            
            # Compute value targets
            with torch.no_grad():
                next_values = self.value_network(next_states).squeeze()
                targets = rewards + 0.99 * next_values * (~dones)
                
            # Update value network
            current_values = self.value_network(states).squeeze()
            value_loss = nn.MSELoss()(current_values, targets)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Update policy network
            advantages = (targets - current_values).detach()
            action_probs = self.policy_network(states)
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
            policy_loss = -(action_log_probs * advantages).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Update exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Update metrics
            self.training_metrics.update({
                "episodes_trained": self.training_metrics["episodes_trained"] + 1,
                "average_reward": rewards.mean().item(),
                "exploration_rate": self.epsilon,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item()
            })
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return self.training_metrics


class ReinforcementLearningPipeline(OptimizedFormalizationPipeline):
    """Self-improving formalization pipeline using reinforcement learning."""
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = setup_logger(__name__)
        
        # Initialize RL agent
        self.rl_agent = FormalizationRLAgent()
        
        # RL-specific tracking
        self.episode_history: List[Dict[str, Any]] = []
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "start_time": time.time()
        }
        
        self.performance_window = deque(maxlen=100)  # Track recent performance
        
        self.logger.info("Reinforcement Learning Pipeline initialized")
        
    async def rl_enhanced_formalize(
        self,
        latex_content: str,
        max_iterations: int = 5,
        **kwargs
    ) -> OptimizedFormalizationResult:
        """Formalize with RL-guided optimization."""
        start_time = time.time()
        episode_id = len(self.episode_history)
        
        try:
            self.logger.info(f"Starting RL-enhanced formalization (Episode {episode_id})")
            
            # Initialize episode state
            current_state = self._create_initial_state(latex_content)
            best_result = None
            total_reward = 0.0
            
            for iteration in range(max_iterations):
                # Agent selects action
                action = self.rl_agent.select_action(current_state)
                self.logger.debug(f"Iteration {iteration}: Action {action.action_type.value}")
                
                # Apply action to modify pipeline parameters
                modified_kwargs = self._apply_action(action, kwargs.copy())
                
                # Attempt formalization with modified parameters
                result = await super().formalize(latex_content, **modified_kwargs)
                
                # Calculate reward based on result
                reward = self._calculate_reward(result, action, current_state)
                total_reward += reward
                
                # Update best result
                if not best_result or (result.success and result.metrics.get("processing_time", float('inf')) < 
                                     best_result.metrics.get("processing_time", float('inf'))):
                    best_result = result
                
                # Create next state
                next_state = self._create_next_state(current_state, result, action)
                
                # Store experience
                experience = RLExperience(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(iteration == max_iterations - 1) or result.success
                )
                self.rl_agent.store_experience(experience)
                
                # Update current state
                current_state = next_state
                
                # Early termination if successful
                if result.success and reward > 0.8:
                    self.logger.info(f"Early success in iteration {iteration}")
                    break
                    
            # End episode and train agent
            self._end_episode(total_reward, time.time() - start_time)
            training_metrics = self.rl_agent.train()
            
            # Enhance result with RL metrics
            rl_metrics = {
                "rl_episode_id": episode_id,
                "rl_total_reward": total_reward,
                "rl_iterations": iteration + 1,
                "rl_training_metrics": training_metrics
            }
            
            if best_result:
                best_result.optimization_stats.update(rl_metrics)
                
            return best_result or OptimizedFormalizationResult(
                success=False,
                error_message="All RL iterations failed",
                optimization_stats=rl_metrics
            )
            
        except Exception as e:
            self.logger.error(f"RL-enhanced formalization failed: {e}")
            self._end_episode(0.0, time.time() - start_time)
            return OptimizedFormalizationResult(
                success=False,
                error_message=str(e),
                optimization_stats={"rl_episode_id": episode_id, "rl_error": str(e)}
            )
            
    def _create_initial_state(self, latex_content: str) -> RLState:
        """Create initial RL state from LaTeX content."""
        # Estimate complexity (mock implementation)
        complexity = min(len(latex_content) / 1000.0, 1.0)
        
        # Determine domain type (mock implementation)
        domain_keywords = {
            "algebra": ["group", "ring", "field", "polynomial"],
            "analysis": ["limit", "continuous", "derivative", "integral"],
            "topology": ["manifold", "continuous", "homeomorphism", "compact"],
            "number_theory": ["prime", "divisor", "congruence", "gcd"]
        }
        
        domain = "general"
        for domain_name, keywords in domain_keywords.items():
            if any(keyword in latex_content.lower() for keyword in keywords):
                domain = domain_name
                break
                
        # Get recent success rate
        recent_results = list(self.performance_window)[-20:] if self.performance_window else []
        success_rate = np.mean([r for r in recent_results]) if recent_results else 0.5
        
        return RLState(
            latex_complexity=complexity,
            domain_type=domain,
            previous_success_rate=success_rate,
            current_temperature=0.7,  # Default temperature
            model_confidence=0.5,     # Initial confidence
            error_pattern="none",
            attempt_count=0,
            time_elapsed=0.0
        )
        
    def _apply_action(self, action: RLAction, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RL action to modify pipeline parameters."""
        modified_kwargs = kwargs.copy()
        
        try:
            if action.action_type == ActionType.ADJUST_TEMPERATURE:
                current_temp = modified_kwargs.get("temperature", 0.7)
                new_temp = np.clip(current_temp + action.parameters["temperature_delta"], 0.1, 1.0)
                modified_kwargs["temperature"] = new_temp
                
            elif action.action_type == ActionType.CHANGE_MODEL:
                modified_kwargs["model"] = action.parameters["model_name"]
                
            elif action.action_type == ActionType.MODIFY_PROMPT:
                # Add prompt modification flags
                modified_kwargs["prompt_modification"] = action.parameters["modification_type"]
                
            elif action.action_type == ActionType.ADD_CONTEXT:
                modified_kwargs["additional_context"] = action.parameters["context_type"]
                
            elif action.action_type == ActionType.RETRY_STRATEGY:
                modified_kwargs["retry_strategy"] = action.parameters["strategy"]
                
            elif action.action_type == ActionType.COMPLEXITY_ADJUSTMENT:
                # Modify timeout or other complexity-related parameters
                current_timeout = modified_kwargs.get("timeout", 30)
                adjustment = action.parameters["complexity_adjustment"]
                new_timeout = max(10, current_timeout + int(adjustment * 60))
                modified_kwargs["timeout"] = new_timeout
                
        except Exception as e:
            self.logger.warning(f"Failed to apply action {action.action_type}: {e}")
            
        return modified_kwargs
        
    def _calculate_reward(
        self, 
        result: OptimizedFormalizationResult, 
        action: RLAction, 
        state: RLState
    ) -> float:
        """Calculate reward based on formalization result."""
        reward = 0.0
        
        # Success bonus
        if result.success:
            reward += 1.0
            
            # Quality bonus based on verification
            if result.verification_status:
                reward += 0.5
                
            # Efficiency bonus (faster is better)
            processing_time = result.processing_time
            if processing_time < 10:
                reward += 0.3
            elif processing_time < 30:
                reward += 0.1
                
        else:
            # Penalty for failure
            reward -= 0.5
            
        # Action-specific rewards
        if action.action_type == ActionType.ADJUST_TEMPERATURE:
            # Reward temperature adjustments that lead to success
            if result.success:
                reward += 0.2 * action.confidence
                
        elif action.action_type == ActionType.CHANGE_MODEL:
            # Reward model changes that improve performance
            if result.success and result.metrics.get("model_confidence", 0) > state.model_confidence:
                reward += 0.3
                
        # Normalize reward to [-1, 1] range
        return np.clip(reward, -1.0, 1.0)
        
    def _create_next_state(
        self, 
        current_state: RLState, 
        result: OptimizedFormalizationResult, 
        action: RLAction
    ) -> RLState:
        """Create next state based on result."""
        return RLState(
            latex_complexity=current_state.latex_complexity,
            domain_type=current_state.domain_type,
            previous_success_rate=float(result.success),
            current_temperature=action.parameters.get("temperature_delta", 0) + current_state.current_temperature
                              if action.action_type == ActionType.ADJUST_TEMPERATURE else current_state.current_temperature,
            model_confidence=result.metrics.get("model_confidence", current_state.model_confidence),
            error_pattern=result.error_message[:50] if result.error_message else "none",
            attempt_count=current_state.attempt_count + 1,
            time_elapsed=current_state.time_elapsed + result.processing_time
        )
        
    def _end_episode(self, total_reward: float, episode_time: float):
        """End current episode and update performance tracking."""
        episode_data = {
            "episode_id": len(self.episode_history),
            "total_reward": total_reward,
            "episode_time": episode_time,
            "timestamp": time.time()
        }
        
        self.episode_history.append(episode_data)
        self.performance_window.append(1.0 if total_reward > 0 else 0.0)
        
        self.logger.info(f"Episode completed: Reward={total_reward:.3f}, Time={episode_time:.2f}s")
        
    def get_rl_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RL training metrics."""
        recent_episodes = self.episode_history[-50:] if self.episode_history else []
        
        return {
            "total_episodes": len(self.episode_history),
            "average_reward_recent": np.mean([ep["total_reward"] for ep in recent_episodes]) if recent_episodes else 0.0,
            "average_episode_time": np.mean([ep["episode_time"] for ep in recent_episodes]) if recent_episodes else 0.0,
            "success_rate_recent": np.mean(list(self.performance_window)) if self.performance_window else 0.0,
            "exploration_rate": self.rl_agent.epsilon,
            "experience_buffer_size": len(self.rl_agent.experience_buffer),
            **self.rl_agent.training_metrics
        }