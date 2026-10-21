"""Evaluation harness: frozen episode sets, per-episode metrics and summary tables.

Every method (baseline or agent) is run on exactly the same seeded episodes, so the
numbers are directly comparable. Seeds come from configs/default.yaml (`evaluation`).
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from damrl.config import Config
from damrl.env import ReservoirEnv
from damrl.scenarios import Episode, episode_from_seed

METRICS = ["return", "shortage", "shortage_days", "spill", "flood_days",
           "downstream_days", "clipped_days", "power"]


def build_episode_set(cfg: Config, split: str) -> List[Episode]:
    """Frozen episodes for 'val' or 'test'. Scenario counts and seed ranges are in config."""
    e = cfg.evaluation
    if split == "val":
        start = e.val_seed_start
        counts = {"ordinary": e.val_ordinary, "drought": e.val_drought, "extreme": e.val_extreme}
    elif split == "test":
        start = e.test_seed_start
        counts = {"ordinary": e.test_ordinary, "drought": e.test_drought,
                  "extreme": e.test_extreme, "shifted": e.test_shifted}
    else:
        raise ValueError("split must be 'val' or 'test'")
    episodes, seed = [], start
    for scenario, n in counts.items():
        for _ in range(n):
            episodes.append(episode_from_seed(cfg, scenario, seed))
            seed += 1
    return episodes


def run_episode(env: ReservoirEnv, policy, episode: Episode) -> Dict[str, float]:
    """Roll one policy through one frozen episode and return its metrics."""
    obs, _ = env.reset(episode=episode)
    m = {k: 0.0 for k in METRICS}
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, reward, done, _, info = env.step(action)
        m["return"] += reward
        m["shortage"] += info["shortage"]
        m["shortage_days"] += float(info["shortage"] > 1e-9)
        m["spill"] += info["spill"]
        m["flood_days"] += float(info["flood_pool"])
        m["downstream_days"] += float(info["downstream_exceed"])
        m["clipped_days"] += float(info["clipped"])
        m["power"] += info["power"]
    m["scenario"] = episode.scenario
    return m


def evaluate(env: ReservoirEnv, policy, episodes: Sequence[Episode]) -> pd.DataFrame:
    """One row per episode, columns = METRICS + scenario + method."""
    rows = [run_episode(env, policy, ep) for ep in episodes]
    df = pd.DataFrame(rows)
    df["method"] = getattr(policy, "name", type(policy).__name__)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric per (method, scenario), plus an 'all' row per method."""
    per_scenario = df.groupby(["method", "scenario"])[METRICS].mean()
    overall = df.groupby("method")[METRICS].mean()
    overall["scenario"] = "all"
    overall = overall.reset_index().set_index(["method", "scenario"])
    out = pd.concat([per_scenario, overall]).sort_index()
    return out.round(2)


def mean_return_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Mean return per method with a 95% confidence interval over episodes."""
    g = df.groupby("method")["return"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["n"])
    return out.round(2)
