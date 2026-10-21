"""Evaluate every baseline on the frozen test set and write the results table.

Outputs:
  results/baselines_episodes.csv   one row per (method, episode)
  results/baselines_summary.csv    mean metrics per (method, scenario)
  results/figures/baselines_return.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from damrl.baselines import FixedReleasePolicy, GreedyDemandPolicy, RandomPolicy, RuleCurvePolicy
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import build_episode_set, evaluate, mean_return_ci, summarise


def load_rule_curve(cfg, path="configs/rule_curve.yaml"):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            params = yaml.safe_load(f)
    else:   # untuned fallback so the script still runs
        params = {"low": 300.0, "high": 700.0, "low_day": 180, "gain": 2.0}
    return RuleCurvePolicy(cfg, **params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config()
    env = ReservoirEnv(cfg)
    episodes = build_episode_set(cfg, args.split)

    policies = [RandomPolicy(seed=0), FixedReleasePolicy(2), GreedyDemandPolicy(),
                load_rule_curve(cfg)]

    frames = []
    for policy in policies:
        df = evaluate(env, policy, episodes)
        frames.append(df)
        print(f"{policy.name:14s} mean return {df['return'].mean():8.1f}  "
              f"shortage {df['shortage'].mean():7.1f}  spill {df['spill'].mean():7.1f}")
    all_df = pd.concat(frames, ignore_index=True)

    Path("results").mkdir(exist_ok=True)
    all_df.to_csv(f"results/baselines_episodes_{args.split}.csv", index=False)
    summary = summarise(all_df)
    summary.to_csv(f"results/baselines_summary_{args.split}.csv")
    print("\n", summary.to_string())
    print("\n", mean_return_ci(all_df).to_string())

    # bar chart: mean return per method and scenario
    pivot = all_df.groupby(["scenario", "method"])["return"].mean().unstack("method")
    ax = pivot.plot.bar(figsize=(9, 4.5))
    ax.set_ylabel("mean episode return")
    ax.set_title(f"Baselines on {args.split} episodes")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), title="method")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = Path("results/figures") / f"baselines_return_{args.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
