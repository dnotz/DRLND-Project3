from collections import deque
from collections import namedtuple
import random
import numpy as np
import torch


class ReplayBuffer():
    """ Replay buffer to store experience tuples """

    def __init__(self, buffer_size, device, seed=0):
        """ Initialize a replay buffer
        Params
        ======
            buffer_size (int): the size of the replay buffer
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.device = device
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(
            state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in samples])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in samples])).float().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in samples])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in samples])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples]).astype(
            np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
