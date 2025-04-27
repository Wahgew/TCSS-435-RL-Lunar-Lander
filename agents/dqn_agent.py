# agents/dqn_agent.py
from typing import   Optional
import numpy as np
import random
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent based on the PyTorch tutorial, adapted for Lunar Lander.

    This agent implements the core DQN algorithm with:
    - Two networks (policy and target) for stable learning
    - Experience replay for decorrelating samples
    - Epsilon-greedy exploration
    - Soft target network updates
    """

    def __init__(
            self,
            state_size: int,
            action_size: int,
            device: torch.device,
            buffer_size: int = 10000,
            batch_size: int = 128,
            gamma: float = 0.99,
            tau: float = 0.005,
            lr: float = 1e-4,
            update_every: int = 4,
            eps_start: float = 0.9,
            eps_end: float = 0.05,
            eps_decay: float = 1000
    ) -> None:
        """
        Initialize a DQN Agent object.

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
            eps_decay: Decay rate of epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        # Epsilon parameters for exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        # Q-Networks (policy and target)
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Using AdamW optimizer with amsgrad as in the PyTorch example
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, device)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Process a step in the environment by storing experience and learning if appropriate.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert numpy arrays to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([[action]], dtype=torch.long, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        # Store transition in memory
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self._learn()

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state
            eps: Optional epsilon override, if None uses internal epsilon schedule

        Returns:
            Selected action
        """
        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Calculate epsilon threshold
        if eps is None:
            # Use internal epsilon with exponential decay
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
        else:
            # Use provided epsilon
            sample = random.random()
            eps_threshold = eps

        # Epsilon-greedy action selection
        if sample > eps_threshold:
            # Exploit: choose best action
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        else:
            # Explore: choose random action
            return random.randrange(self.action_size)

    def _learn(self) -> None:
        """
        Update policy network using a batch of experiences from memory.

        Follows the DQN update rule with MSE loss and gradient descent.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        states, actions, next_states, rewards, dones = self.memory.sample()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            # Get the maximum predicted Q-values from the target network
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Set V(s) = 0 for terminal states
            next_state_values = next_state_values * (1 - dones)

        # Compute the expected Q values
        expected_state_action_values = rewards + (self.gamma * next_state_values)

        # Compute loss using MSE
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping (as in PyTorch example)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network
        self._soft_update()

    def _soft_update(self) -> None:
        """
        Soft update of the target network parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str) -> None:
        """
        Save the policy network to a file.

        Args:
            filepath: Path to save the model
        """
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """
        Load a saved model into both policy and target networks.

        Args:
            filepath: Path to load the model from
        """
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(torch.load(filepath))