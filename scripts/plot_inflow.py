"""Sanity-check plot: one sample year of each inflow scenario against demand."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from damrl.config import load_config
from damrl.inflow import SCENARIOS, demand_profile, generate_inflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/figures/inflow_scenarios.png")
    args = parser.parse_args()

    cfg = load_config()
    demand = demand_profile(cfg)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        sample = generate_inflow(cfg, scenario, np.random.default_rng(args.seed))
        ax.plot(sample.inflow, label="inflow")
        ax.plot(demand, "--", label="demand")
        ax.set_title(f"{scenario}: inflow {sample.inflow.sum():.0f}, demand {demand.sum():.0f} MCM/yr")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    for ax in axes[1]:
        ax.set_xlabel("day of year")
    for ax in axes[:, 0]:
        ax.set_ylabel("MCM per day")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
