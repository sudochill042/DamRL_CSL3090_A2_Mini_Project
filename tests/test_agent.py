import numpy as np
import pytest

torch = pytest.importorskip("torch")

from damrl.agent import DQNAgent, DQNConfig
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.replay import ReplayBuffer
from damrl.scenarios import Episode
from damrl.train import linear_epsilon, train


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def small_cfg(**kw):
    base = dict(batch_size=16, buffer_size=500, learning_starts=32, target_update_every=50, hidden=32)
    base.update(kw)
    return DQNConfig(**base)


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(capacity=5, obs_dim=3, n_actions=4, seed=0)
    for i in range(7):
        buf.add(np.full(3, i), i % 4, float(i), np.full(3, i + 1), i == 6)
    assert len(buf) == 5
    assert set(buf.reward.tolist()) == {2.0, 3.0, 4.0, 5.0, 6.0}
    b = buf.sample(8)
    assert b["obs"].shape == (8, 3) and b["action"].shape == (8,)
    assert b["next_mask"].shape == (8, 4) and b["next_mask"].all()


def test_epsilon_schedule():
    assert linear_epsilon(0, 1000) == pytest.approx(1.0)
    assert linear_epsilon(600, 1000) == pytest.approx(0.05)
    assert linear_epsilon(300, 1000) == pytest.approx(0.525)
    assert linear_epsilon(5000, 1000) == pytest.approx(0.05)


def test_agent_action_and_update_shapes(cfg):
    env = ReservoirEnv(cfg)
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(env.obs_dim, env.n_actions, small_cfg(), seed=0)
    assert agent.q_values(obs).shape == (env.n_actions,)
    assert 0 <= agent.select_action(obs, 0.0) < env.n_actions
    assert 0 <= agent.act(obs, env) < env.n_actions
    for _ in range(40):
        a = agent.select_action(obs, 1.0)
        nxt, r, d, _, _ = env.step(a)
        agent.store(obs, a, r, nxt, d)
        obs = nxt
    loss = agent.update()
    assert np.isfinite(loss) and loss >= 0


@pytest.mark.parametrize("double", [False, True])
def test_update_moves_q_toward_target(cfg, double):
    env = ReservoirEnv(cfg)
    obs, _ = env.reset(seed=1)
    agent = DQNAgent(env.obs_dim, env.n_actions, small_cfg(double=double), seed=1)
    for _ in range(64):
        a = agent.select_action(obs, 1.0)
        nxt, r, d, _, _ = env.step(a)
        agent.store(obs, a, r, nxt, d)
        obs = nxt
    before = agent.q_values(obs).copy()
    for _ in range(20):
        agent.update()
    assert not np.allclose(before, agent.q_values(obs))


def test_masked_agent_never_picks_infeasible_action(cfg):
    env = ReservoirEnv(cfg)
    n = cfg.reservoir.episode_days
    ep = Episode("ordinary", np.full(n, 0.0), np.zeros((n, cfg.forecast.horizon)), 107.0)
    obs, _ = env.reset(episode=ep)
    agent = DQNAgent(env.obs_dim, env.n_actions, small_cfg(use_mask=True), seed=0)
    mask = env.action_mask()
    assert mask.sum() < env.n_actions
    for eps in (0.0, 1.0):
        for _ in range(20):
            assert mask[agent.select_action(obs, eps, mask)]


def test_save_and_load_roundtrip(cfg, tmp_path):
    env = ReservoirEnv(cfg)
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(env.obs_dim, env.n_actions, small_cfg(double=True), seed=0)
    path = tmp_path / "a.pt"
    agent.save(path)
    loaded = DQNAgent.load(path)
    assert loaded.cfg.double and loaded.obs_dim == env.obs_dim
    assert np.allclose(agent.q_values(obs), loaded.q_values(obs))


def test_train_on_tiny_deterministic_reservoir(cfg, tmp_path):
    """Sanity check: a few short episodes on a flat-inflow reservoir run end to end,
    the log has the right shape, and a checkpoint is written."""
    env = ReservoirEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.n_actions, small_cfg(), seed=0)
    n = cfg.reservoir.episode_days
    val = [Episode("ordinary", np.full(n, 8.0), np.zeros((n, cfg.forecast.horizon)), 500.0)]
    log = train(env, agent, n_episodes=3, seed=0, val_episodes=val, val_every=3,
                log_path=str(tmp_path / "log.csv"), ckpt_path=str(tmp_path / "m.pt"),
                verbose=False)
    assert len(log) == 3 and log["steps"].iloc[-1] == 3 * n
    assert np.isfinite(log["return"]).all()
    assert not np.isnan(log["val_return"].iloc[-1])
    assert (tmp_path / "m.pt").exists() and (tmp_path / "log.csv").exists()
