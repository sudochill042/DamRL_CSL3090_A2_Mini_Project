"""Training loop: epsilon-greedy episodes, periodic validation, best-checkpoint saving.

Each call to `train` writes a CSV log with one row per episode (return, mean loss,
epsilon, and validation return whenever validation ran) and saves the checkpoint with
the best validation return.
"""
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from damrl.agent import DQNAgent, DQNConfig
from damrl.env import ReservoirEnv
from damrl.evaluation import evaluate
from damrl.scenarios import Episode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_epsilon(step: int, total_steps: int, start: float = 1.0, end: float = 0.05,
                   fraction: float = 0.6) -> float:
    """Decay from `start` to `end` over the first `fraction` of training, then hold."""
    decay_steps = max(1, int(fraction * total_steps))
    return max(end, start + (end - start) * min(1.0, step / decay_steps))


def train(env: ReservoirEnv, agent: DQNAgent, n_episodes: int, seed: int,
          val_episodes: Optional[Sequence[Episode]] = None, val_every: int = 25,
          log_path: Optional[str] = None, ckpt_path: Optional[str] = None,
          eps_end: float = 0.05, eps_fraction: float = 0.6, verbose: bool = True) -> pd.DataFrame:
    set_seed(seed)
    env.reset(seed=seed)
    total_steps = n_episodes * env.n_days
    val_env = ReservoirEnv(env.cfg, use_forecast=env.use_forecast,
                           forecast_scale=env.forecast_scale)

    rows, best_val, t0 = [], -np.inf, time.time()
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        ep_return, losses, done = 0.0, [], False
        while not done:
            eps = linear_epsilon(agent.steps, total_steps, end=eps_end, fraction=eps_fraction)
            mask = env.action_mask() if agent.cfg.use_mask else None
            action = agent.select_action(obs, eps, mask)
            next_obs, reward, done, _, _ = env.step(action)
            next_mask = env.action_mask() if agent.cfg.use_mask else None
            agent.store(obs, action, reward, next_obs, done, next_mask)
            if agent.ready():
                losses.append(agent.update())
            obs = next_obs
            ep_return += reward

        row = {"episode": ep, "scenario": info["scenario"], "return": ep_return,
               "loss": float(np.mean(losses)) if losses else np.nan, "epsilon": eps,
               "steps": agent.steps, "val_return": np.nan, "time_s": time.time() - t0}

        if val_episodes is not None and (ep % val_every == 0 or ep == n_episodes):
            val_df = evaluate(val_env, agent, val_episodes)
            row["val_return"] = float(val_df["return"].mean())
            if row["val_return"] > best_val:
                best_val = row["val_return"]
                if ckpt_path:
                    agent.save(ckpt_path)
        rows.append(row)

        if verbose and (ep % 10 == 0 or ep == 1):
            msg = (f"ep {ep:4d} {info['scenario']:8s} return {ep_return:8.1f} "
                   f"loss {row['loss']:.4f} eps {eps:.2f}")
            if not np.isnan(row["val_return"]):
                msg += f"  val {row['val_return']:.1f} (best {best_val:.1f})"
            print(msg, flush=True)

    if ckpt_path and val_episodes is None:
        agent.save(ckpt_path)
    log = pd.DataFrame(rows)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
    return log


SCENARIO_MIXES = {
    "ordinary": "ordinary",
    "mixed": {"ordinary": 0.6, "drought": 0.2, "extreme": 0.2},
}
