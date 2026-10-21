"""Grid-search the seasonal rule curve on the validation episodes.

Writes the best parameters to configs/rule_curve.yaml and the full grid to
results/rule_curve_grid.csv. Only validation seeds are used, never test seeds.
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd
import yaml

from damrl.baselines import RuleCurvePolicy
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import build_episode_set, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="configs/rule_curve.yaml")
    args = parser.parse_args()

    cfg = load_config()
    env = ReservoirEnv(cfg)
    val = build_episode_set(cfg, "val")

    lows = [100, 150, 200, 300]
    highs = [400, 500, 600, 700]
    low_days = [120, 150, 180]      # day the target is lowest (just before the monsoon)
    gains = [0.25, 0.5, 1.0, 2.0, 4.0]

    rows = []
    for low, high, low_day, gain in itertools.product(lows, highs, low_days, gains):
        policy = RuleCurvePolicy(cfg, low, high, low_day, gain)
        df = evaluate(env, policy, val)
        rows.append({"low": low, "high": high, "low_day": low_day, "gain": gain,
                     "val_return": df["return"].mean(),
                     "val_shortage": df["shortage"].mean(),
                     "val_spill": df["spill"].mean()})
        print(f"low={low} high={high} low_day={low_day} gain={gain}: "
              f"return {rows[-1]['val_return']:.1f}")

    grid = pd.DataFrame(rows).sort_values("val_return", ascending=False)
    Path("results").mkdir(exist_ok=True)
    grid.to_csv("results/rule_curve_grid.csv", index=False)

    best = grid.iloc[0]
    params = {"low": float(best.low), "high": float(best.high),
              "low_day": int(best.low_day), "gain": float(best.gain)}
    with open(args.out, "w") as f:
        yaml.safe_dump(params, f)
    print(f"\nbest {params} with validation return {best.val_return:.1f}; saved {args.out}")


if __name__ == "__main__":
    main()
