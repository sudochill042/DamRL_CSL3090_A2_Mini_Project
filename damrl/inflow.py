"""Synthetic seasonal inflow and demand profiles. All parameters are assumptions."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from damrl.config import Config

SCENARIOS = ("ordinary", "drought", "extreme", "shifted")


@dataclass(frozen=True)
class InflowSample:
    scenario: str
    inflow: np.ndarray           # MCM per day, shape (n_days,)
    event_start: Optional[int]   # first day of the flood pulse (extreme scenario only)


def seasonal_mean(days, base, amplitude, center, width):
    """Expected daily inflow: a baseflow plus a Gaussian monsoon bump."""
    return base + amplitude * np.exp(-((days - center) ** 2) / (2.0 * width ** 2))


def demand_profile(cfg: Config, n_days: Optional[int] = None) -> np.ndarray:
    """Seasonal water demand (MCM per day), highest around cfg.demand.peak_day."""
    n = n_days or cfg.reservoir.episode_days
    t = np.arange(n)
    d = cfg.demand
    return d.base * (1.0 + d.amplitude * np.cos(2.0 * np.pi * (t - d.peak_day) / 365.0))


def generate_inflow(cfg: Config, scenario: str, rng: np.random.Generator,
                    n_days: Optional[int] = None) -> InflowSample:
    """Draw one year of daily inflow.

    Daily inflow = seasonal mean x lognormal noise with AR(1) persistence, so wet and dry
    spells last several days. The noise has mean 1. Drought, extreme and shifted scenarios
    modify this baseline. For a given seed, ordinary, drought and extreme share the same
    underlying noise path, which makes scenario effects easy to test.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    n = n_days or cfg.reservoir.episode_days
    p, s = cfg.inflow, cfg.scenarios

    center, sigma = p.monsoon_center, p.log_sigma
    if scenario == "shifted":
        center, sigma = s.shifted_monsoon_center, s.shifted_log_sigma
    mu = seasonal_mean(np.arange(n), p.base, p.monsoon_amplitude, center, p.monsoon_width)

    phi = p.ar_coefficient
    eps = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = eps[0]
    for t in range(1, n):
        z[t] = phi * z[t - 1] + np.sqrt(1.0 - phi ** 2) * eps[t]
    inflow = mu * np.exp(sigma * z - 0.5 * sigma ** 2)

    event_start = None
    if scenario == "drought":
        inflow = inflow * s.drought_factor
    elif scenario == "extreme":
        event_start = int(rng.integers(s.extreme_start_min, s.extreme_start_max + 1))
        wet_from = max(0, event_start - s.extreme_wet_days)
        inflow[wet_from:event_start] *= s.extreme_wet_multiplier
        inflow[event_start:event_start + s.extreme_pulse_days] *= s.extreme_pulse_multiplier

    return InflowSample(scenario=scenario, inflow=inflow, event_start=event_start)
