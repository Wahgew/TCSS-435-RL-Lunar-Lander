from typing import Optional
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import time  # Added for timing. tell us how long its running

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent based on the PyTorch tutorial, adapted and optimized for Lunar Lander.
    This agent implements the core DQN algorithm with:
    - Two networks (policy and target) for stable learning
    - Experience replay for decorrelating samples
    - Epsilon-greedy exploration with linear decay
    - Soft target network updates
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
            eps_decay: float = 0.995  # multiplicative decay
    ) -> None:
        """
        Initialize a DQN Agent object with optimized hyperparameters for Lunar Lander.
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            device: Device to run the model on
            buffer_size: Replay memory size (larger for more stable learning)
            batch_size: Minibatch size (smaller for more frequent updates)
            gamma: Discount factor
            tau: Soft update parameter (smaller for more stable target updates)
            lr: Learning rate (increased for faster learning)
            update_every: How often to update the target network
            eps_start: Starting value of epsilon (higher for more initial exploration)
            eps_end: Minimum value of epsilon (lower for more exploitation)
            eps_decay: Multiplicative decay factor for epsilon per episode
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

        # Using AdamW optimizer 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, device)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        # For timing
        self.start_time = time.time()

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
        Select an action using epsilon-greedy policy with linear decay.
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
            # Use internal epsilon with multiplicative decay
            eps_threshold = max(self.eps_end, self.eps_start)
            self.steps_done += 1
        else:
            # Use provided epsilon
            eps_threshold = eps

        # Epsilon-greedy action selection
        if random.random() > eps_threshold:
            # Exploit: choose best action
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        else:
            # Explore: choose random action
            return random.randrange(self.action_size)

    def update_epsilon(self) -> None:
        """Update epsilon after each episode using multiplicative decay."""
        self.eps_start = max(self.eps_end, self.eps_decay * self.eps_start)

    def _learn(self) -> None:
        """
        Update policy network using a batch of experiences from memory.
        This implements the standard DQN update rule:
        1. Sample a batch of transitions from memory
        2. Compute current Q-values and target Q-values
        3. Compute loss as Huber loss between current and target Q-values
        4. Update policy network with gradient descent
        5. Soft update target network
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

        # Compute the expected Q values - Bellman equation
        expected_state_action_values = rewards + (self.gamma * next_state_values)

        # Compute loss using Huber Loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model with gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network
        self._soft_update()

    def _soft_update(self) -> None:
        """
        Soft update of the target network parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        This slowly updates the target network, making learning more stable.
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

    def get_training_time(self) -> float:
        """Return the elapsed training time in seconds."""
        return time.time() - self.start_time