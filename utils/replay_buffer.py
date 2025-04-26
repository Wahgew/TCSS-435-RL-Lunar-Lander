# utils/replay_buffer.py
from collections import namedtuple, deque
import random
import torch

# Named tuple that represents a singular transition.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# cyclic buffer that holds the observed transitions
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # method for selecting a random batch of transitions for training.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)