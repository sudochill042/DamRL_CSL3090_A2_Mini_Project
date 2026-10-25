# DamRL: Robust Reservoir Release Control with DQN and Double DQN

**Team:** Nakshatra Cial (B24CM1090), Harshita Pareek (B24CM1028)  **Course:** AI, Assignment 2 (Real-Life Use Cases of RL)

## Overview
A reservoir operator must decide every day how much water to release. Releasing too
little risks flooding downstream during the monsoon; releasing too much risks
shortages in the dry season. This project formulates daily release control as a
Markov decision process on a synthetic single-reservoir simulator and trains DQN and
Double DQN agents. They are compared with random, fixed-release, greedy and seasonal
rule-curve baselines, including on drought, extreme-inflow and shifted-season
scenarios the agents never trained on.

## Repository layout
```
damrl/        simulator, baselines, agents, evaluation code
configs/      default.yaml holds every simulator constant
scripts/      figure and experiment scripts
tests/        pytest unit tests
results/      CSV results and figures
models/       trained agent checkpoints
report/       two-page report
```

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Reproduction
```
python scripts/plot_inflow.py                     # sample year of each scenario
python scripts/rollout_policy.py --policy fixed   # sanity rollout of the environment
python scripts/tune_rule_curve.py                 # grid-search rule curve on validation seeds
python scripts/run_baselines.py --split test      # all baselines on the frozen test set
```
Baseline results land in `results/baselines_summary_test.csv` and
`results/figures/baselines_return_test.png`.

Training and evaluation of the agents:
```
python scripts/train_dqn.py --algo dqn  --scenario ordinary --episodes 300 --seed 0
python scripts/train_dqn.py --algo ddqn --scenario ordinary --episodes 300 --seed 0
python scripts/plot_learning_curves.py --tags dqn_ordinary ddqn_ordinary
python scripts/evaluate_agents.py --tags dqn_ordinary ddqn_ordinary --split test
```
Flags: `--mask` enables action masking, `--scenario mixed` trains on a 60/20/20 mix of
ordinary, drought and extreme years, `--no-forecast` removes the inflow forecast from
the state. Checkpoints go to `models/`, logs to `results/train_*.csv`.

Runs used in the report (3 seeds for DQN and Double DQN, 2 for mixed, 1 for masked):
```
for s in 0 1 2: python scripts/train_dqn.py --algo dqn  --scenario ordinary --seed $s
for s in 0 1 2: python scripts/train_dqn.py --algo ddqn --scenario ordinary --seed $s
for s in 0 1:   python scripts/train_dqn.py --algo dqn  --scenario mixed    --seed $s
python scripts/train_dqn.py --algo dqn --scenario ordinary --seed 0 --mask
```
Then evaluate everything and regenerate every figure and table:
```
python scripts/evaluate_agents.py --tags dqn_ordinary ddqn_ordinary dqn_mixed dqn_mask_ordinary
python scripts/make_figures.py    --tags dqn_ordinary ddqn_ordinary dqn_mixed dqn_mask_ordinary
```
`results/comparison_test.csv` is the main results table; figures are in `results/figures/`.
The report source is `report/report.tex` (compile with `pdflatex` from inside `report/`).

## Results (test set, 250 frozen years)
| Method | Return | Flood days | Shortage (MCM) |
|---|---|---|---|
| Random | -196.4 | 30.1 | 1058 |
| Fixed 10 MCM/day | -143.3 | 70.2 | 447 |
| Greedy | -102.1 | 30.6 | 543 |
| Rule curve (tuned) | -49.9 | 11.5 | 531 |
| DQN | -37.9 | 7.6 | 456 |
| Double DQN | -36.2 | 6.7 | 447 |
| DQN, mixed-scenario training | -36.5 | 7.7 | 451 |
| DQN, action masking | -40.3 | 5.8 | 506 |

## Limitations
All simulator parameters are assumptions. The inflow is synthetic, the hydropower
term is a simple proxy, and downstream river hydraulics are not modelled.

## Status
- [x] Project setup
- [x] Configuration and inflow generator
- [x] Reservoir environment
- [x] Baselines and evaluation
- [x] DQN and Double DQN
- [x] Robustness experiments
- [x] Report
