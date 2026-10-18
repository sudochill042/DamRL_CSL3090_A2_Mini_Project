import numpy as np
import pytest

from damrl.config import load_config
from damrl.inflow import SCENARIOS, demand_profile, generate_inflow, seasonal_mean


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_shape_and_nonnegative(cfg, scenario):
    sample = generate_inflow(cfg, scenario, np.random.default_rng(0))
    assert sample.inflow.shape == (cfg.reservoir.episode_days,)
    assert np.all(sample.inflow >= 0)
    assert np.all(np.isfinite(sample.inflow))


def test_same_seed_is_reproducible(cfg):
    a = generate_inflow(cfg, "extreme", np.random.default_rng(5))
    b = generate_inflow(cfg, "extreme", np.random.default_rng(5))
    assert np.array_equal(a.inflow, b.inflow)
    assert a.event_start == b.event_start


def test_different_seeds_differ(cfg):
    a = generate_inflow(cfg, "ordinary", np.random.default_rng(1))
    b = generate_inflow(cfg, "ordinary", np.random.default_rng(2))
    assert not np.array_equal(a.inflow, b.inflow)


def test_noise_preserves_seasonal_mean(cfg):
    p = cfg.inflow
    mu_total = seasonal_mean(np.arange(365), p.base, p.monsoon_amplitude,
                             p.monsoon_center, p.monsoon_width).sum()
    totals = [generate_inflow(cfg, "ordinary", np.random.default_rng(i)).inflow.sum()
              for i in range(200)]
    assert np.mean(totals) == pytest.approx(mu_total, rel=0.03)


def test_drought_scales_ordinary(cfg):
    base = generate_inflow(cfg, "ordinary", np.random.default_rng(3)).inflow
    dry = generate_inflow(cfg, "drought", np.random.default_rng(3)).inflow
    assert np.allclose(dry, cfg.scenarios.drought_factor * base)


def test_extreme_applies_wet_spell_and_pulse(cfg):
    s = cfg.scenarios
    base = generate_inflow(cfg, "ordinary", np.random.default_rng(4)).inflow
    sample = generate_inflow(cfg, "extreme", np.random.default_rng(4))
    start = sample.event_start
    assert s.extreme_start_min <= start <= s.extreme_start_max
    ratio = sample.inflow / base
    assert np.allclose(ratio[start:start + s.extreme_pulse_days], s.extreme_pulse_multiplier)
    assert np.allclose(ratio[start - s.extreme_wet_days:start], s.extreme_wet_multiplier)
    outside = np.ones(len(base), dtype=bool)
    outside[start - s.extreme_wet_days:start + s.extreme_pulse_days] = False
    assert np.allclose(ratio[outside], 1.0)


def test_shifted_monsoon_peaks_later(cfg):
    ord_mean = np.mean([generate_inflow(cfg, "ordinary", np.random.default_rng(i)).inflow
                        for i in range(100)], axis=0)
    shift_mean = np.mean([generate_inflow(cfg, "shifted", np.random.default_rng(i)).inflow
                          for i in range(100)], axis=0)
    assert np.argmax(shift_mean) > np.argmax(ord_mean)


def test_unknown_scenario_rejected(cfg):
    with pytest.raises(ValueError):
        generate_inflow(cfg, "flood", np.random.default_rng(0))


def test_demand_peaks_in_dry_season_and_inflow_covers_it(cfg):
    demand = demand_profile(cfg)
    assert demand.shape == (365,)
    assert np.argmax(demand) == cfg.demand.peak_day
    p = cfg.inflow
    annual_inflow = seasonal_mean(np.arange(365), p.base, p.monsoon_amplitude,
                                  p.monsoon_center, p.monsoon_width).sum()
    assert annual_inflow > demand.sum()
