import numpy as np
import pytest

from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.inflow import SCENARIOS
from damrl.scenarios import Episode, episode_from_seed


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def flat_episode(cfg, inflow, storage, n=None):
    n = n or cfg.reservoir.episode_days
    return Episode("ordinary", np.full(n, float(inflow)),
                   np.zeros((n, cfg.forecast.horizon)), float(storage))


def test_observation_shapes(cfg):
    env = ReservoirEnv(cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,) and env.obs_dim == 10
    assert np.all(np.isfinite(obs)) and info["scenario"] == "ordinary"
    assert ReservoirEnv(cfg, use_forecast=False).reset(seed=0)[0].shape == (7,)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_mass_balance_and_bounds(cfg, scenario):
    r = cfg.reservoir
    env = ReservoirEnv(cfg, scenario=scenario)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    done, steps = False, 0
    while not done:
        before = env.storage
        obs, reward, done, _, info = env.step(int(rng.integers(env.n_actions)))
        expected = (before + info["inflow"] - info["release"]
                    - info["evaporation"] - info["spill"])
        assert info["storage"] == pytest.approx(expected, abs=1e-9)
        assert r.dead_storage - 1e-9 <= info["storage"] <= r.capacity + 1e-9
        assert info["release"] <= info["command"] + 1e-12
        assert np.isfinite(reward) and np.all(np.isfinite(obs)) and np.abs(obs).max() < 10
        steps += 1
    assert steps == cfg.reservoir.episode_days


def test_terminates_only_at_end_of_year(cfg):
    env = ReservoirEnv(cfg)
    env.reset(seed=0)
    for day in range(cfg.reservoir.episode_days):
        _, _, done, truncated, _ = env.step(2)
        assert truncated is False
        assert done == (day == cfg.reservoir.episode_days - 1)


def test_spill_fills_to_capacity_and_is_penalised(cfg):
    r, w = cfg.reservoir, cfg.reward
    env = ReservoirEnv(cfg)
    env.reset(episode=flat_episode(cfg, 100.0, 990.0))
    _, reward, _, _, info = env.step(0)
    assert info["spill"] > 0 and info["storage"] == pytest.approx(r.capacity)
    downstream = (info["release"] + info["spill"] - r.safe_downstream_flow) / r.max_release
    flood = (r.capacity - r.flood_threshold) / (r.capacity - r.flood_threshold)
    expected = (-w.w_shortage * info["shortage"] / w.demand_ref - w.w_flood * flood
                - w.w_downstream * downstream)
    assert reward == pytest.approx(expected)


def test_release_clipped_to_protect_dead_storage(cfg):
    r = cfg.reservoir
    env = ReservoirEnv(cfg)
    env.reset(episode=flat_episode(cfg, 0.0, 105.0))
    _, reward, _, _, info = env.step(env.n_actions - 1)   # command the maximum release
    assert info["clipped"] and info["release"] < info["command"]
    assert info["storage"] == pytest.approx(r.dead_storage, abs=1e-9)
    assert reward < 0


def test_reward_for_a_normal_day(cfg):
    r, w = cfg.reservoir, cfg.reward
    env = ReservoirEnv(cfg)
    env.reset(episode=flat_episode(cfg, 8.0, 500.0))
    _, reward, _, _, info = env.step(2)                   # release level index 2 = 10 MCM/day
    assert info["release"] == 10.0 and info["shortage"] == 0.0 and not info["clipped"]
    frac = (500.0 - r.dead_storage) / (r.capacity - r.dead_storage)
    assert reward == pytest.approx(w.w_power * (10.0 / r.max_release) * (0.5 + 0.5 * frac))


def test_shortage_penalty(cfg):
    w = cfg.reward
    env = ReservoirEnv(cfg)
    env.reset(episode=flat_episode(cfg, 8.0, 500.0))
    _, reward, _, _, info = env.step(0)                   # release nothing
    assert info["shortage"] == pytest.approx(env.demand[0])
    assert reward == pytest.approx(-w.w_shortage * env.demand[0] / w.demand_ref)


def test_action_mask(cfg):
    env = ReservoirEnv(cfg)
    env.reset(episode=flat_episode(cfg, 5.0, 112.0))
    mask = env.action_mask()
    assert mask.tolist() == [True, True, True, False, False, False, False]
    env.reset(episode=flat_episode(cfg, 5.0, 100.0))
    assert env.action_mask()[0] and env.action_mask().sum() == 1


def test_forecast_is_exact_at_zero_noise_and_noisy_otherwise(cfg):
    ep = episode_from_seed(cfg, "ordinary", 11)
    perfect = ReservoirEnv(cfg, forecast_scale=0.0)
    perfect.reset(episode=ep)
    for _ in range(10):
        obs, *_ = perfect.step(2)
    t = perfect.t
    expected = ep.inflow[t:t + 3] / cfg.reservoir.max_release
    assert np.allclose(obs[7:10], expected, atol=1e-6)

    noisy = ReservoirEnv(cfg, forecast_scale=1.0)
    noisy.reset(episode=ep)
    for _ in range(10):
        obs_noisy, *_ = noisy.step(2)
    assert not np.allclose(obs_noisy[7:10], expected, atol=1e-6)


def test_same_episode_replays_identically(cfg):
    ep = episode_from_seed(cfg, "extreme", 3)
    runs = []
    for _ in range(2):
        env = ReservoirEnv(cfg)
        env.reset(episode=ep)
        total, done = 0.0, False
        while not done:
            _, reward, done, _, _ = env.step(3)
            total += reward
        runs.append((total, env.storage))
    assert runs[0] == runs[1]


def test_same_seed_gives_same_start(cfg):
    a = ReservoirEnv(cfg).reset(seed=9)[0]
    b = ReservoirEnv(cfg).reset(seed=9)[0]
    assert np.array_equal(a, b)


def test_mixed_scenarios_sample_all_types(cfg):
    env = ReservoirEnv(cfg, scenario={"ordinary": 0.6, "drought": 0.2, "extreme": 0.2}, seed=0)
    seen = {env.reset()[1]["scenario"] for _ in range(60)}
    assert seen == {"ordinary", "drought", "extreme"}


def test_bad_inputs_rejected(cfg):
    with pytest.raises(ValueError):
        ReservoirEnv(cfg, scenario="flood")
    env = ReservoirEnv(cfg)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)
