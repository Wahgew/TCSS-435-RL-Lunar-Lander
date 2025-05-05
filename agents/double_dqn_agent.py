# agents/double_dqn_agent.py
from typing import Optional
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent that inherits from the base DQNAgent.
    Double DQN addresses the overestimation bias in regular DQN by decoupling
    action selection and action evaluation:
    - The online network selects the best action
    - The target network evaluates that action
    """

    def __init__( 
            self,
            state_size: int,
            action_size: int,
            device: torch.device,
            buffer_size: int = 100000,
            batch_size: int = 64,
            gamma: float = 0.99,
            tau: float = 1e-3, 
            lr: float = 5e-4, 
            update_every: int = 4,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            eps_decay: float = 0.995  
    ) -> None:
        """
        Initialize a Double DQN Agent with optimized hyperparameters.
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            device: Device to run the model on
            buffer_size: Replay memory size
            batch_size: Minibatch size
            gamma: Discount factor
            tau: Soft update parameter 
            lr: Learning rate 
            update_every: How often to update the target network
            eps_start: Starting value of epsilon
            eps_end: Minimum value of epsilon
            eps_decay: Multiplicative decay factor for epsilon per episode
        """
        # Initialize using the parent class constructor
        super(DoubleDQNAgent, self).__init__(
            state_size, action_size, device, buffer_size, batch_size,
            gamma, tau, lr, update_every, eps_start, eps_end, eps_decay
        )
        
        # Using AdamW optimizer 
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr, 
            amsgrad=True,
            weight_decay=1e-5  
        )

    def _learn(self) -> None:
        """
        Update policy network using a batch of experiences from memory.
        This implements the Double DQN update rule with some optimizations:
        1. Use online network to select actions, target network to evaluate
        2. Add gradient clipping to stabilize learning
        3. Use Huber loss for more robustness to outliers
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        states, actions, next_states, rewards, dones = self.memory.sample()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(states).gather(1, actions)

        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Get the actions that would be selected by the online network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            
            # Evaluate those actions using the target network
            next_state_values = self.target_net(next_states).gather(1, next_actions)
            
            # Set V(s) = 0 for terminal states
            next_state_values = next_state_values * (1 - dones)

        # Compute the expected Q values - Bellman equation
        expected_state_action_values = rewards + (self.gamma * next_state_values)

        # Compute Huber loss instead of MSE - more robust to outliers
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model with gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping to prevent exploding gradients 
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Soft update of the target network
        self._soft_update()