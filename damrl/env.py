"""Single-reservoir environment: one release decision per day for a simulated year.

Units are MCM (million cubic metres) and MCM per day. All constants come from the config.
Interface follows the usual reset/step convention (Gymnasium-style) without the dependency.
"""
from typing import Mapping, Optional, Union

import numpy as np

from damrl.config import Config, load_config
from damrl.inflow import SCENARIOS, demand_profile
from damrl.scenarios import Episode, make_episode

ScenarioSpec = Union[str, Mapping[str, float]]


class ReservoirEnv:
    def __init__(self, cfg: Optional[Config] = None, scenario: ScenarioSpec = "ordinary",
                 use_forecast: bool = True, forecast_scale: float = 1.0,
                 seed: Optional[int] = None):
        self.cfg = cfg or load_config()
        self.scenario_probs = self._parse_scenario(scenario)
        self.use_forecast = use_forecast
        self.forecast_scale = forecast_scale
        self.rng = np.random.default_rng(seed)

        self.levels = np.asarray(self.cfg.release_levels, dtype=float)
        self.n_actions = len(self.levels)
        self.n_days = self.cfg.reservoir.episode_days
        self.horizon = self.cfg.forecast.horizon
        self.demand = demand_profile(self.cfg)
        self.obs_dim = 7 + (self.horizon if use_forecast else 0)
        self._scale = self.cfg.reservoir.max_release   # scale for flows in the observation

        self.episode: Optional[Episode] = None
        self.t = 0
        self.storage = 0.0
        self._recent = []
        self._prev_release = 0.0

    @staticmethod
    def _parse_scenario(spec: ScenarioSpec) -> dict:
        probs = {spec: 1.0} if isinstance(spec, str) else dict(spec)
        if not probs or any(name not in SCENARIOS for name in probs):
            raise ValueError(f"scenario names must be in {SCENARIOS}")
        total = sum(probs.values())
        if min(probs.values()) < 0 or total <= 0:
            raise ValueError("scenario probabilities must be non-negative and sum to > 0")
        return {name: p / total for name, p in probs.items()}

    # ------------------------------------------------------------------ episode control
    def reset(self, seed: Optional[int] = None, episode: Optional[Episode] = None):
        """Start a new year. Pass `episode` to replay a frozen scenario."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if episode is None:
            names = list(self.scenario_probs)
            if len(names) == 1:
                name = names[0]
            else:
                name = str(self.rng.choice(names, p=[self.scenario_probs[n] for n in names]))
            episode = make_episode(self.cfg, name, self.rng)
        self.episode = episode
        self.t = 0
        self.storage = float(episode.initial_storage)
        self._recent = [self.cfg.inflow.base] * 7   # prior for the first week
        self._prev_release = 0.0
        return self._obs(), {"scenario": episode.scenario}

    def action_mask(self) -> np.ndarray:
        """Actions that cannot breach dead storage even with zero inflow (observable only)."""
        feasible = self.levels <= (self.storage - self.cfg.reservoir.dead_storage) + 1e-9
        feasible[0] = True
        return feasible

    # ------------------------------------------------------------------ dynamics
    def step(self, action: int):
        if self.episode is None:
            raise RuntimeError("call reset() before step()")
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions})")
        r, w = self.cfg.reservoir, self.cfg.reward
        t, storage = self.t, self.storage
        inflow = float(self.episode.inflow[t])
        demand = float(self.demand[t])

        evaporation = r.evaporation_rate * storage
        available = max(0.0, storage + inflow - evaporation - r.dead_storage)
        command = float(self.levels[int(action)])
        release = min(command, available)
        clipped = command > available + 1e-9

        new_storage = storage + inflow - release - evaporation
        spill = max(0.0, new_storage - r.capacity)
        new_storage = min(new_storage, r.capacity)

        shortage = max(0.0, demand - release)
        flood = max(0.0, new_storage - r.flood_threshold) / (r.capacity - r.flood_threshold)
        downstream = max(0.0, release + spill - r.safe_downstream_flow) / r.max_release
        frac_before = (storage - r.dead_storage) / (r.capacity - r.dead_storage)
        power = release / r.max_release * (0.5 + 0.5 * frac_before)

        reward = (-w.w_shortage * shortage / w.demand_ref
                  - w.w_flood * flood
                  - w.w_downstream * downstream
                  - w.w_clip * float(clipped)
                  + w.w_power * power)

        self.storage = new_storage
        self._recent = self._recent[1:] + [inflow]
        self._prev_release = release
        self.t += 1
        terminated = self.t >= self.n_days

        info = {
            "day": t, "inflow": inflow, "demand": demand, "storage": new_storage,
            "release": release, "command": command, "evaporation": evaporation,
            "spill": spill, "shortage": shortage, "power": power, "clipped": clipped,
            "flood_pool": new_storage > r.flood_threshold,
            "downstream_exceed": (release + spill) > r.safe_downstream_flow,
        }
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ observation
    def _forecast(self) -> np.ndarray:
        """Noisy inflow forecast for today and the next days; error grows with lead time."""
        t = min(self.t, self.n_days - 1)
        lead = np.arange(self.horizon)
        truth = self.episode.inflow[np.minimum(t + lead, self.n_days - 1)]
        sigma = self.cfg.forecast.rel_sigma_per_day * (lead + 1) * self.forecast_scale
        return np.maximum(0.0, truth * (1.0 + sigma * self.episode.forecast_noise[t]))

    def _obs(self) -> np.ndarray:
        r = self.cfg.reservoir
        angle = 2.0 * np.pi * self.t / 365.0
        day = min(self.t, self.n_days - 1)
        obs = [
            (self.storage - r.dead_storage) / (r.capacity - r.dead_storage),
            np.sin(angle), np.cos(angle),
            self._recent[-1] / self._scale,
            float(np.mean(self._recent)) / self._scale,
            self.demand[day] / self.cfg.reward.demand_ref,
            self._prev_release / r.max_release,
        ]
        if self.use_forecast:
            obs.extend(self._forecast() / self._scale)
        return np.asarray(obs, dtype=np.float32)
