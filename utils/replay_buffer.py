from typing import Tuple
import numpy as np
import random
import torch
from collections import namedtuple, deque

# Define a transition tuple similar to PyTorch example
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Based on the ReplayMemory class from the PyTorch tutorial.
    """

    def __init__(self, capacity: int, batch_size: int, device: torch.device) -> None:
        """
        Initialize a ReplayBuffer object.
        Args:
            capacity: Maximum size of buffer
            batch_size: Size of each training batch
            device: Device to use for tensor operations
        """
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
             reward: torch.Tensor, done: torch.Tensor) -> None:
        """
        Add a new experience to memory.
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        Returns:
            Tuple of (states, actions, next_states, rewards, dones)
        """
        transitions = random.sample(self.memory, self.batch_size)

        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Convert to tensors and send to device
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        return (states, actions, next_states, rewards, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)