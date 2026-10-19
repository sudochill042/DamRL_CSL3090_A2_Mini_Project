"""Episodes: one fully specified simulated year (inflow, forecast noise, initial storage).

Freezing an Episode makes every method face exactly the same conditions. Evaluation
sets are lists of Episodes built from fixed seeds.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from damrl.config import Config
from damrl.inflow import generate_inflow


@dataclass(frozen=True, eq=False)
class Episode:
    scenario: str
    inflow: np.ndarray            # (n_days,) MCM per day
    forecast_noise: np.ndarray    # (n_days, horizon) standard-normal draws for forecast errors
    initial_storage: float
    event_start: Optional[int] = None


def make_episode(cfg: Config, scenario: str, rng: np.random.Generator) -> Episode:
    sample = generate_inflow(cfg, scenario, rng)
    noise = rng.standard_normal((len(sample.inflow), cfg.forecast.horizon))
    r = cfg.reservoir
    storage0 = float(rng.uniform(r.initial_storage_min, r.initial_storage_max))
    return Episode(scenario, sample.inflow, noise, storage0, sample.event_start)


def episode_from_seed(cfg: Config, scenario: str, seed: int) -> Episode:
    return make_episode(cfg, scenario, np.random.default_rng(seed))
