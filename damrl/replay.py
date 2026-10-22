"""Uniform experience replay buffer backed by preallocated numpy arrays."""
from typing import Dict

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.ones((capacity, n_actions), dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, next_mask=None) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.next_mask[i] = True if next_mask is None else next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {"obs": self.obs[idx], "action": self.action[idx], "reward": self.reward[idx],
                "next_obs": self.next_obs[idx], "done": self.done[idx],
                "next_mask": self.next_mask[idx]}

    def __len__(self) -> int:
        return self.size
