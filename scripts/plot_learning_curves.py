"""Plot training and validation return curves from results/train_*.csv.

  python scripts/plot_learning_curves.py --tags dqn_ordinary ddqn_ordinary

Each tag is averaged over every seed found (results/train_<tag>_seed*.csv); the shaded
band is +-1 std across seeds. The rule-curve test return is drawn as a reference line
if results/baselines_summary_test.csv exists.
"""
import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(x, k=10):
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag in args.tags:
        files = sorted(glob.glob(f"results/train_{tag}_seed*.csv"))
        if not files:
            print(f"no logs for {tag}")
            continue
        logs = [pd.read_csv(f) for f in files]
        n = min(len(l) for l in logs)
        train_ret = np.stack([smooth(l["return"].to_numpy()[:n]) for l in logs])
        ep = np.arange(1, n + 1)
        m, s = train_ret.mean(0), train_ret.std(0)
        axes[0].plot(ep, m, label=f"{tag} ({len(logs)} seeds)")
        axes[0].fill_between(ep, m - s, m + s, alpha=0.2)

        val = [l.dropna(subset=["val_return"]) for l in logs]
        nv = min(len(v) for v in val)
        v_ep = val[0]["episode"].to_numpy()[:nv]
        v_ret = np.stack([v["val_return"].to_numpy()[:nv] for v in val])
        vm, vs = v_ret.mean(0), v_ret.std(0)
        axes[1].plot(v_ep, vm, marker="o", ms=3, label=tag)
        axes[1].fill_between(v_ep, vm - vs, vm + vs, alpha=0.2)

    summary = Path("results/baselines_summary_test.csv")
    if summary.exists():
        df = pd.read_csv(summary)
        rc = df[(df["method"] == "rule_curve") & (df["scenario"] == "all")]
        if len(rc):
            for ax in axes:
                ax.axhline(float(rc["return"].iloc[0]), color="k", ls="--", lw=1, label="rule curve (test)")

    axes[0].set_title("training return (10-episode moving average)")
    axes[1].set_title("validation return (90 frozen episodes)")
    for ax in axes:
        ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or f"results/figures/learning_curves_{'_'.join(args.tags)}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
