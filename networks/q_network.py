# networks/q_network.py
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.

    Based on the DQN architecture from the PyTorch tutorial,
    adapted for the Lunar Lander environment.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        """
        Initialize the Q-Network.

        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
        """
        super(QNetwork, self).__init__()

        # Define network layers
        # Using similar architecture to PyTorch example but with state_size inputs
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: State tensor

        Returns:
            Q-values for each action
        """
        # Apply ReLU activations between layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # No activation on output layer (Q-values can be any real number)