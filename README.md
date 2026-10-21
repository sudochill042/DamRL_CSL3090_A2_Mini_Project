# DamRL: Robust Reservoir Release Control with DQN and Double DQN

**Team:** (add names)  **Course:** AI, Assignment 2 (Real-Life Use Cases of RL)

## Overview
A reservoir operator must decide every day how much water to release. Releasing too
little risks flooding downstream during the monsoon; releasing too much risks
shortages in the dry season. This project formulates daily release control as a
Markov decision process on a synthetic single-reservoir simulator and trains DQN and
Double DQN agents. They are compared with fixed-release, greedy, seasonal rule-curve
and stochastic-DP baselines, including on drought, extreme-inflow and shifted-season
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
`results/figures/baselines_return_test.png`. Training and agent evaluation commands are
added below as they are implemented.

## Limitations
All simulator parameters are assumptions. The inflow is synthetic, the hydropower
term is a simple proxy, and downstream river hydraulics are not modelled.

## Status
- [x] Project setup
- [x] Configuration and inflow generator
- [x] Reservoir environment
- [x] Baselines and evaluation
- [ ] DQN and Double DQN
- [ ] Robustness experiments
- [ ] Report
