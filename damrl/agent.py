"""DQN and Double DQN agent for discrete release control.

Q-network: MLP obs_dim -> 128 -> 128 -> n_actions. Target network with periodic hard
updates, Huber loss, Adam. `double=True` switches the target to Double DQN: the online
network picks argmax a', the target network evaluates it. Optional action masking sets
Q of infeasible actions to -inf when choosing actions and when forming targets.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from damrl.replay import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    target_update_every: int = 1_000    # environment steps between hard target copies
    train_every: int = 1
    hidden: int = 128
    double: bool = False
    use_mask: bool = False
    grad_clip: float = 10.0


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: Optional[DQNConfig] = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or DQNConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, n_actions, seed=seed)
        self.steps = 0
        self.name = ("ddqn" if self.cfg.double else "dqn") + ("_mask" if self.cfg.use_mask else "")

    # ------------------------------------------------------------------ acting
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q(x).squeeze(0).cpu().numpy()

    def select_action(self, obs: np.ndarray, epsilon: float, mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy over (optionally masked) Q-values."""
        if mask is None or not self.cfg.use_mask:
            mask = np.ones(self.n_actions, dtype=bool)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        q = self.q_values(obs)
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def act(self, obs: np.ndarray, env) -> int:
        """Greedy action; matches the baseline policy interface used by the evaluator."""
        mask = env.action_mask() if self.cfg.use_mask else None
        return self.select_action(obs, epsilon=0.0, mask=mask)

    # ------------------------------------------------------------------ learning
    def store(self, obs, action, reward, next_obs, done, next_mask=None) -> None:
        self.buffer.add(obs, action, reward, next_obs, done,
                        next_mask if self.cfg.use_mask else None)
        self.steps += 1
        if self.steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def ready(self) -> bool:
        return (len(self.buffer) >= self.cfg.learning_starts
                and self.steps % self.cfg.train_every == 0)

    def update(self) -> float:
        """One gradient step on a sampled batch. Returns the loss."""
        c = self.cfg
        b = self.buffer.sample(c.batch_size)
        obs = torch.as_tensor(b["obs"], device=self.device)
        act = torch.as_tensor(b["action"], device=self.device)
        rew = torch.as_tensor(b["reward"], device=self.device)
        nxt = torch.as_tensor(b["next_obs"], device=self.device)
        done = torch.as_tensor(b["done"], device=self.device)
        nmask = torch.as_tensor(b["next_mask"], device=self.device)

        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_target = self.q_target(nxt)
            if c.use_mask:
                q_next_target = q_next_target.masked_fill(~nmask, -1e9)
            if c.double:
                q_next_online = self.q(nxt)
                if c.use_mask:
                    q_next_online = q_next_online.masked_fill(~nmask, -1e9)
                a_star = q_next_online.argmax(dim=1, keepdim=True)
                q_next = q_next_target.gather(1, a_star).squeeze(1)
            else:
                q_next = q_next_target.max(dim=1).values
            target = rew + c.gamma * (1.0 - done) * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.optim.step()
        return float(loss.item())

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "cfg": asdict(self.cfg),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "steps": self.steps}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], DQNConfig(**ckpt["cfg"]), device=device)
        agent.q.load_state_dict(ckpt["q"])
        agent.q_target.load_state_dict(ckpt["q"])
        agent.steps = ckpt.get("steps", 0)
        agent.q.eval()
        return agent
