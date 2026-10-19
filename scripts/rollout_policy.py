"""Run a simple policy for one simulated year and plot inflow, storage and release.

Used to check that the environment behaves sensibly before any learning happens.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.inflow import SCENARIOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="ordinary", choices=SCENARIOS)
    parser.add_argument("--policy", default="random", choices=["random", "fixed"])
    parser.add_argument("--level", type=int, default=2, help="action index for the fixed policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    env = ReservoirEnv(cfg, scenario=args.scenario)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    rows, total, done = [], 0.0, False
    while not done:
        action = int(rng.integers(env.n_actions)) if args.policy == "random" else args.level
        _, reward, done, _, info = env.step(action)
        total += reward
        rows.append(info)

    col = lambda key: np.array([row[key] for row in rows], dtype=float)
    r = cfg.reservoir
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(col("inflow"), label="inflow")
    axes[0].plot(col("demand"), "--", label="demand")
    axes[0].set_ylabel("MCM per day"); axes[0].legend()
    axes[1].plot(col("storage"), label="storage")
    axes[1].axhline(r.flood_threshold, color="r", ls=":", label="flood threshold")
    axes[1].axhline(r.dead_storage, color="k", ls=":", label="dead storage")
    axes[1].set_ylabel("MCM"); axes[1].legend()
    axes[2].step(range(len(rows)), col("release"), where="post", label="release")
    axes[2].plot(col("spill"), color="r", label="spill")
    axes[2].set_ylabel("MCM per day"); axes[2].set_xlabel("day of year"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{args.policy} policy, {args.scenario} scenario")
    fig.tight_layout()

    out = Path("results/figures") / f"rollout_{args.policy}_{args.scenario}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"return {total:.1f}, shortage {col('shortage').sum():.0f}, "
          f"spill {col('spill').sum():.0f}, saved {out}")


if __name__ == "__main__":
    main()
