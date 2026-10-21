import numpy as np
import pytest

from damrl.baselines import (FixedReleasePolicy, GreedyDemandPolicy, RandomPolicy,
                             RuleCurvePolicy, nearest_level_index)
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import METRICS, build_episode_set, evaluate, run_episode, summarise


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_nearest_level_index():
    levels = np.array([0.0, 5.0, 10.0, 15.0])
    assert nearest_level_index(levels, 0.0) == 0
    assert nearest_level_index(levels, 6.0) == 1
    assert nearest_level_index(levels, 12.6) == 3
    assert nearest_level_index(levels, 99.0) == 3


def test_fixed_policy_is_constant(cfg):
    env = ReservoirEnv(cfg)
    obs, _ = env.reset(seed=0)
    policy = FixedReleasePolicy(3)
    assert all(policy.act(obs, env) == 3 for _ in range(10))


def test_greedy_tracks_demand_and_dumps_in_flood_pool(cfg):
    env = ReservoirEnv(cfg)
    obs, _ = env.reset(seed=0)
    policy = GreedyDemandPolicy()
    for _ in range(50):
        a = policy.act(obs, env)
        demand = env.demand[env.t]
        assert abs(env.levels[a] - demand) <= 2.5 + 1e-9    # nearest of 0,5,...,30
        obs, *_ = env.step(a)
    env.storage = cfg.reservoir.flood_threshold + 1.0
    assert policy.act(obs, env) == env.n_actions - 1


def test_rule_curve_target_shape(cfg):
    policy = RuleCurvePolicy(cfg, low=300, high=700, low_day=180, gain=2.0)
    assert policy.target.shape == (cfg.reservoir.episode_days,)
    assert policy.target.min() == pytest.approx(300.0)
    assert policy.target.max() == pytest.approx(700.0, abs=1.0)
    assert int(np.argmin(policy.target)) == 180


def test_rule_curve_releases_more_when_above_target(cfg):
    env = ReservoirEnv(cfg)
    policy = RuleCurvePolicy(cfg, low=300, high=700, low_day=180, gain=4.0)
    obs, _ = env.reset(seed=0)
    env.storage = 900.0
    high = policy.act(obs, env)
    env.storage = 150.0
    low = policy.act(obs, env)
    assert high > low


def test_episode_sets_are_frozen_and_sized(cfg):
    e = cfg.evaluation
    val = build_episode_set(cfg, "val")
    test = build_episode_set(cfg, "test")
    assert len(val) == e.val_ordinary + e.val_drought + e.val_extreme
    assert len(test) == e.test_ordinary + e.test_drought + e.test_extreme + e.test_shifted
    assert {ep.scenario for ep in test} == {"ordinary", "drought", "extreme", "shifted"}
    again = build_episode_set(cfg, "val")
    assert all(np.array_equal(a.inflow, b.inflow) for a, b in zip(val, again))
    with pytest.raises(ValueError):
        build_episode_set(cfg, "train")


def test_run_episode_metrics(cfg):
    env = ReservoirEnv(cfg)
    ep = build_episode_set(cfg, "val")[0]
    m = run_episode(env, FixedReleasePolicy(2), ep)
    assert set(METRICS) <= set(m) and m["scenario"] == "ordinary"
    assert np.isfinite(m["return"])
    assert 0 <= m["flood_days"] <= cfg.reservoir.episode_days
    assert m["shortage_days"] >= 0 and m["spill"] >= 0


def test_evaluate_and_summarise(cfg):
    env = ReservoirEnv(cfg)
    eps = build_episode_set(cfg, "val")[:5]
    df = evaluate(env, RandomPolicy(seed=1), eps)
    assert len(df) == 5 and (df["method"] == "random").all()
    s = summarise(df)
    assert ("random", "all") in s.index
    assert s.loc[("random", "all"), "return"] == pytest.approx(df["return"].mean(), abs=0.01)
