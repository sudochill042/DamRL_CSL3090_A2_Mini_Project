"""Regenerate every report figure from saved checkpoints and result CSVs.

  python scripts/make_figures.py --tags dqn_ordinary ddqn_ordinary dqn_mixed dqn_mask_ordinary

Produces in results/figures/:
  trajectory_extreme.png     DQN vs rule curve on one extreme test year
  trajectory_drought.png     DQN vs rule curve on one drought test year
  comparison_return_test.png (via evaluate_agents) is NOT rebuilt here
  learning_curves_report.png learning curves for the given tags
  failure_cases.csv          worst five DQN test episodes with their metrics
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from damrl.agent import DQNAgent
from damrl.baselines import RuleCurvePolicy
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import build_episode_set, run_episode

FIG = Path("results/figures")


def rollout(env, policy, episode):
    obs, _ = env.reset(episode=episode)
    rows, done = [], False
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, env))
        rows.append(info)
    return pd.DataFrame(rows)


def trajectory_figure(cfg, agent, rule, episode, out, title):
    env = ReservoirEnv(cfg)
    a, b = rollout(env, agent, episode), rollout(env, rule, episode)
    r = cfg.reservoir
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(a["inflow"], color="tab:blue", label="inflow")
    axes[0].plot(a["demand"], "--", color="tab:orange", label="demand")
    axes[0].set_ylabel("MCM/day")
    axes[1].plot(a["storage"], label=agent.name.upper().replace("_", " "))
    axes[1].plot(b["storage"], label="rule curve")
    axes[1].axhline(r.flood_threshold, color="r", ls=":", label="flood threshold")
    axes[1].axhline(r.dead_storage, color="k", ls=":", label="dead storage")
    axes[1].set_ylabel("storage (MCM)")
    axes[2].step(a.index, a["release"] + a["spill"], where="post", label=f"{agent.name} release+spill")
    axes[2].step(b.index, b["release"] + b["spill"], where="post", label="rule curve release+spill")
    axes[2].axhline(r.safe_downstream_flow, color="r", ls=":", label="safe downstream flow")
    axes[2].set_ylabel("MCM/day"); axes[2].set_xlabel("day of year")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["dqn_ordinary", "ddqn_ordinary"])
    p.add_argument("--main", default="dqn_ordinary", help="tag used for trajectory plots")
    args = p.parse_args()

    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    with open("configs/rule_curve.yaml") as f:
        rule = RuleCurvePolicy(cfg, **yaml.safe_load(f))
    agent = DQNAgent.load(f"models/{args.main}_seed0.pt")
    agent.name = args.main

    test = build_episode_set(cfg, "test")
    extreme = next(ep for ep in test if ep.scenario == "extreme")
    drought = next(ep for ep in test if ep.scenario == "drought")
    trajectory_figure(cfg, agent, rule, extreme, FIG / "trajectory_extreme.png",
                      "Extreme inflow year: DQN vs rule curve")
    trajectory_figure(cfg, agent, rule, drought, FIG / "trajectory_drought.png",
                      "Drought year: DQN vs rule curve")

    # failure analysis: the worst test episodes for the main agent
    env = ReservoirEnv(cfg)
    rows = []
    for i, ep in enumerate(test):
        m = run_episode(env, agent, ep)
        m["episode_index"] = i
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("return").head(5)
    df.to_csv("results/failure_cases.csv", index=False)
    print("worst five test episodes:\n", df.round(1).to_string(index=False))

    # learning curves for the report
    subprocess.run([sys.executable, "scripts/plot_learning_curves.py", "--tags", *args.tags,
                    "--out", str(FIG / "learning_curves_report.png")], check=True)


if __name__ == "__main__":
    main()
