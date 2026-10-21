"""Non-learning release policies used as reference points for the RL agents.

Every policy exposes `act(obs, env) -> int` (an action index) and `name`. Baselines may
read the simulator's observable state through `env` (storage, day, demand), the same
information the agent sees in its observation vector. None of them see future inflow.
"""
from typing import Optional

import numpy as np

from damrl.config import Config
from damrl.env import ReservoirEnv


def nearest_level_index(levels: np.ndarray, release: float) -> int:
    """Index of the discrete release level closest to `release`."""
    return int(np.argmin(np.abs(levels - release)))


class FixedReleasePolicy:
    """Release the same amount every day regardless of storage or season."""

    def __init__(self, level_index: int):
        self.level_index = int(level_index)
        self.name = f"fixed_{self.level_index}"

    def act(self, obs: np.ndarray, env: ReservoirEnv) -> int:
        return self.level_index


class GreedyDemandPolicy:
    """Myopic operator: match today's demand, dump water only once the flood pool is hit.

    Reacts to the current state only, with no seasonal planning. It meets demand with the
    nearest release level and switches to maximum release when storage is already above
    the flood threshold, so it reacts to floods instead of preparing for them.
    """

    name = "greedy_demand"

    def act(self, obs: np.ndarray, env: ReservoirEnv) -> int:
        r = env.cfg.reservoir
        if env.storage > r.flood_threshold:
            return env.n_actions - 1
        demand = float(env.demand[min(env.t, env.n_days - 1)])
        return nearest_level_index(env.levels, demand)


class RuleCurvePolicy:
    """Seasonal rule curve: the standard operating rule used by real reservoirs.

    A target storage S*(t) is defined for every day of the year. It is kept low before
    the monsoon (to leave room for floods) and high after it (to carry water through
    the dry season). Each day the policy releases demand plus a correction that pushes
    storage toward the target:

        release = demand + gain * (S - S*(t)) / max_release

    The target curve is a cosine between `low` and `high` storage, lowest on
    `low_day` and highest on `low_day + 182`. `low`, `high`, `low_day` and `gain`
    are tuned on validation episodes (scripts/tune_rule_curve.py).
    """

    def __init__(self, cfg: Config, low: float, high: float, low_day: int, gain: float,
                 name: Optional[str] = None):
        self.cfg = cfg
        self.low, self.high, self.low_day, self.gain = float(low), float(high), int(low_day), float(gain)
        self.name = name or "rule_curve"
        days = np.arange(cfg.reservoir.episode_days)
        mid, half = 0.5 * (high + low), 0.5 * (high - low)
        self.target = mid - half * np.cos(2.0 * np.pi * (days - self.low_day) / 365.0)

    def act(self, obs: np.ndarray, env: ReservoirEnv) -> int:
        t = min(env.t, env.n_days - 1)
        demand = float(env.demand[t])
        correction = self.gain * (env.storage - self.target[t]) / self.cfg.reservoir.max_release
        release = float(np.clip(demand + correction, 0.0, self.cfg.reservoir.max_release))
        return nearest_level_index(env.levels, release)


class RandomPolicy:
    """Uniform random action; the floor every learned policy must beat."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: ReservoirEnv) -> int:
        return int(self.rng.integers(env.n_actions))
