# DamRL formulation (source for the report)

Units: MCM (million cubic metres) and MCM per day. Constants: `configs/default.yaml`.
Implementation: `damrl/env.py`.

## State (10 dimensions; 7 without forecast)
| # | Feature | Scaling |
|---|---------|---------|
| 0 | storage fraction | (S - S_min) / (S_max - S_min) |
| 1-2 | day of year | sin, cos of 2*pi*t/365 |
| 3 | yesterday's inflow | / 30 |
| 4 | mean inflow of last 7 days | / 30 |
| 5 | today's demand | / 10 |
| 6 | previous release | / 30 |
| 7-9 | noisy inflow forecast for days t, t+1, t+2 | / 30 |

Forecast error: truth x (1 + sigma_k * N(0,1)), sigma_k = 0.15 * (k+1) * forecast_scale.
The agent never sees the true inflow of the current day or whether it will be clipped.

## Actions (7, discrete)
Release 0, 5, 10, 15, 20, 25 or 30 MCM per day.

## Dynamics
E = 0.0005 S; available = max(0, S + I - E - S_min); R = min(command, available);
S' = S + I - R - E - spill, where spill = max(0, S + I - R - E - S_max).
A command above `available` is clipped and counted as a violation.

## Reward (per day)
r = -w_s * shortage/10 - w_f * F - w_d * DF - w_c * clip + w_p * P

- shortage = max(0, D - R)
- F = max(0, S' - S_flood) / (S_max - S_flood)
- DF = max(0, R + spill - Q_safe) / R_max
- P = (R / R_max) * (0.5 + 0.5 * storage fraction before the step)
- weights: w_s 1.0, w_f 1.0, w_d 3.0, w_c 0.5, w_p 0.2 (to be tuned on validation scenarios)

## Episode
365 days, initial storage Uniform(450, 650). The year ends after day 365 (terminal).

## Action masking (RQ3)
Mask out actions with release > S - S_min. Uses observable storage only.
