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

## Research questions
1. Does Double DQN outperform DQN on this task?
2. Does training on mixed (ordinary, drought, flood) years improve robustness?
3. Does feasible-action masking beat penalty-only constraint handling?
4. (Stretch) How much does forecast quality matter for RL versus rule-based policies?

## Problem formulation
See `docs/experiment_plan.md` (state, action, reward and dynamics are filled in as the
environment is implemented).

## Repository layout
```
damrl/        simulator, baselines, agents, evaluation code
configs/      default.yaml holds every simulator constant
scripts/      figure and experiment scripts
tests/        pytest unit tests
results/      CSV results and figures
models/       trained agent checkpoints
report/       two-page report
docs/         experiment plan
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
Filled in as experiments are added (training, evaluation, figure commands).

## Limitations
All simulator parameters are assumptions. The inflow is synthetic, the hydropower
term is a simple proxy, and downstream river hydraulics are not modelled.

## Status
- [x] Project setup
- [x] Configuration and inflow generator
- [ ] Reservoir environment
- [ ] Baselines and evaluation
- [ ] DQN and Double DQN
- [ ] Robustness experiments
- [ ] Report
