import pytest
import yaml

from damrl.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.reservoir.capacity == 1000.0
    assert cfg.reservoir.episode_days == 365
    assert cfg.n_actions == 7


def test_release_levels_fit_outlet():
    cfg = load_config()
    assert cfg.release_levels[0] == 0.0
    assert cfg.release_levels[-1] == cfg.reservoir.max_release


def test_validation_rejects_bad_storage_order(tmp_path):
    with open(DEFAULT_CONFIG_PATH) as f:
        d = yaml.safe_load(f)
    d["reservoir"]["flood_threshold"] = 50.0  # below dead storage
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        load_config(bad)
