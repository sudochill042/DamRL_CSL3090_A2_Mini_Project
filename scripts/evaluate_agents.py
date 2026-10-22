"""Evaluate trained checkpoints on the frozen test set next to the baselines.

  python scripts/evaluate_agents.py --tags dqn_ordinary ddqn_ordinary

Every models/<tag>_seed*.pt is evaluated; agent rows are averaged over seeds.
Outputs results/agents_episodes_test.csv, results/comparison_test.csv and
results/figures/comparison_return_test.png.
"""
import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from damrl.agent import DQNAgent
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import build_episode_set, evaluate, mean_return_ci, summarise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    cfg = load_config()
    episodes = build_episode_set(cfg, args.split)
    frames = []
    for tag in args.tags:
        for path in sorted(glob.glob(f"models/{tag}_seed*.pt")):
            agent = DQNAgent.load(path)
            env = ReservoirEnv(cfg, use_forecast=(agent.obs_dim > 7))
            df = evaluate(env, agent, episodes)
            df["method"] = tag
            df["seed"] = Path(path).stem.split("seed")[-1]
            frames.append(df)
            print(f"{path}: mean return {df['return'].mean():.1f}")
    if not frames:
        raise SystemExit("no checkpoints found for the given tags")
    agents = pd.concat(frames, ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    agents.to_csv(f"results/agents_episodes_{args.split}.csv", index=False)

    base_path = Path(f"results/baselines_episodes_{args.split}.csv")
    combined = agents
    if base_path.exists():
        combined = pd.concat([pd.read_csv(base_path), agents], ignore_index=True)
    summary = summarise(combined)
    summary.to_csv(f"results/comparison_{args.split}.csv")
    print("\n", summary.to_string())
    print("\n", mean_return_ci(combined).to_string())

    pivot = combined.groupby(["scenario", "method"])["return"].mean().unstack("method")
    ax = pivot.plot.bar(figsize=(10, 4.5))
    ax.set_ylabel("mean episode return")
    ax.set_title(f"All methods on {args.split} episodes")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), title="method", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = Path("results/figures") / f"comparison_return_{args.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
