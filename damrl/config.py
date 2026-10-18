"""Typed access to configs/default.yaml. Every simulator constant lives in the YAML file."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass(frozen=True)
class ReservoirConfig:
    capacity: float
    dead_storage: float
    flood_threshold: float
    max_release: float
    safe_downstream_flow: float
    evaporation_rate: float
    initial_storage_min: float
    initial_storage_max: float
    episode_days: int


@dataclass(frozen=True)
class DemandConfig:
    base: float
    amplitude: float
    peak_day: int


@dataclass(frozen=True)
class InflowConfig:
    base: float
    monsoon_amplitude: float
    monsoon_center: float
    monsoon_width: float
    ar_coefficient: float
    log_sigma: float


@dataclass(frozen=True)
class ScenarioConfig:
    drought_factor: float
    extreme_start_min: int
    extreme_start_max: int
    extreme_wet_days: int
    extreme_wet_multiplier: float
    extreme_pulse_days: int
    extreme_pulse_multiplier: float
    shifted_monsoon_center: float
    shifted_log_sigma: float


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int
    rel_sigma_per_day: float


@dataclass(frozen=True)
class RewardConfig:
    w_shortage: float
    w_flood: float
    w_downstream: float
    w_clip: float
    w_power: float
    demand_ref: float


@dataclass(frozen=True)
class EvaluationConfig:
    train_seed_start: int
    val_seed_start: int
    test_seed_start: int
    val_ordinary: int
    val_drought: int
    val_extreme: int
    test_ordinary: int
    test_drought: int
    test_extreme: int
    test_shifted: int


@dataclass(frozen=True)
class Config:
    reservoir: ReservoirConfig
    demand: DemandConfig
    inflow: InflowConfig
    scenarios: ScenarioConfig
    release_levels: Tuple[float, ...]
    forecast: ForecastConfig
    reward: RewardConfig
    evaluation: EvaluationConfig

    @property
    def n_actions(self) -> int:
        return len(self.release_levels)


def _validate(cfg: Config) -> None:
    r = cfg.reservoir
    if not (0 <= r.dead_storage < r.flood_threshold < r.capacity):
        raise ValueError("need 0 <= dead_storage < flood_threshold < capacity")
    if not (r.dead_storage <= r.initial_storage_min <= r.initial_storage_max <= r.capacity):
        raise ValueError("initial storage range must lie within [dead_storage, capacity]")
    levels = cfg.release_levels
    if list(levels) != sorted(levels) or levels[0] < 0 or levels[-1] > r.max_release:
        raise ValueError("release_levels must be increasing and within [0, max_release]")
    s = cfg.scenarios
    if not (0 < s.extreme_start_min <= s.extreme_start_max < r.episode_days):
        raise ValueError("extreme event start range must lie within the episode")


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load and validate a YAML config (defaults to configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG_PATH, "r") as f:
        d = yaml.safe_load(f)
    cfg = Config(
        reservoir=ReservoirConfig(**d["reservoir"]),
        demand=DemandConfig(**d["demand"]),
        inflow=InflowConfig(**d["inflow"]),
        scenarios=ScenarioConfig(**d["scenarios"]),
        release_levels=tuple(float(x) for x in d["actions"]["release_levels"]),
        forecast=ForecastConfig(**d["forecast"]),
        reward=RewardConfig(**d["reward"]),
        evaluation=EvaluationConfig(**d["evaluation"]),
    )
    _validate(cfg)
    return cfg
