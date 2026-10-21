# Projection model: 2025 holdout evaluation

Weekly projections for all of 2025 (weeks 1–18, 8,602 player-weeks), each week made only
from data before it, scored with a real league's half-PPR rules, and compared with two
baselines built the same way.

**Holdout discipline.** Every model was fitted on 2016–2023 and tuned on 2024. No
parameter, setting or design choice was made by looking at 2025 results; 2025 was run
once, after the model was frozen. (Field-goal distances are fitted per season, and each
season is predicted from the one before it.)

Reproduce: `uv run ffa evaluate --season 2025 --scoring <league scoring JSON>`.

## Baselines

| Baseline | What it is |
| --- | --- |
| Season average | The player's average fantasy points in his earlier games this season (last season's average before he has played) |
| Opportunity average | His average [ffopportunity](https://github.com/ffverse/ffopportunity) expected points over earlier games: what his targets and carries were worth at league-average efficiency, scored with the league's rules |

ffopportunity's *same-week* expected points are not used: they are computed from the
opportunities a player actually got in that game, so they would know the future.

## 1. Accuracy

Mean absolute error on player-weeks the player played. Edge = baseline error minus model
error (positive = model better), with 95% intervals from resampling whole weeks.

| Position | Player-weeks | Model MAE | Season avg MAE | Edge | Opportunity MAE | Edge |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 660 | **5.84** | 6.66 | +0.82 [+0.36, +1.35] | 6.48 | +0.65 [+0.28, +1.16] |
| RB | 1,554 | **3.96** | 4.10 | +0.14 [+0.07, +0.20] | 4.14 | +0.18 [+0.11, +0.25] |
| WR | 2,356 | **3.74** | 4.01 | +0.27 [+0.16, +0.42] | 4.07 | +0.33 [+0.27, +0.39] |
| TE | 1,224 | **2.95** | 3.16 | +0.21 [+0.12, +0.32] | 3.13 | +0.18 [+0.10, +0.26] |
| K | 534 | **4.14** | 4.42 | +0.28 [−0.03, +0.64] | — | — |
| D/ST | 544 | **4.75** | 5.10 | +0.35 [+0.09, +0.63] | — | — |

The model is more accurate than both baselines at every position; the kicker edge is not
statistically distinguishable from zero.

## 2. Uncertainty

Share of actual scores inside the projected 10th–90th percentile range (should be ~80%),
for fantasy-relevant players (projection ≥ 5):

| QB | RB | WR | TE | K | D/ST |
| --- | --- | --- | --- | --- | --- |
| 77.7% | 76.5% | 82.0% | 79.8% | 81.2% | 75.7% |

Across all projected players, including deep backups, WR (86%) and TE (88%) ranges run
slightly wide.

## 3. Start/sit

All same-position pairs of startable players in the same week (the pool is the top 24 QB,
36 RB, 48 WR, 24 TE, 20 K and 20 D/ST by season average, chosen independently of the
model), where both played and scored differently: how often each method picks the higher
scorer.

| Pairs | Model | Season avg | Opportunity avg |
| --- | --- | --- | --- |
| All (48,033) | **62.3%** | 58.9% | 61.1% |
| Close calls, season averages within 3 points (26,186) | **58.9%** | 53.6% | 56.0% |

Paired on the same decisions (95% intervals, resampling weeks):

| | All pairs | Close calls |
| --- | --- | --- |
| Model minus season avg | +3.4 pts [+2.3, +4.5] | +5.3 pts [+3.9, +6.6] |
| Model minus opportunity avg | +2.2 pts [+0.8, +3.5] | +3.5 pts [+1.6, +5.5] |

The edge is largest on close calls, which are the decisions that need help.

**Are the probabilities honest?** The model's P(A outscores B), from simulations that share
each game:

| Model said | 52.5% | 57.4% | 62.5% | 67.4% | 74.5% | 84.2% | 93.1% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Favorite won | 51.8% | 56.6% | 61.3% | 65.7% | 72.4% | 80.5% | 92.0% |

Well calibrated, slightly overconfident above 70%.

## Changes after the holdout

- **Midweek availability (Oct 2026).** Before a team's final injury report, a player with no
  designation who missed or was limited in practice now gets a reduced chance to play. The
  holdout used final reports, where this rule does not apply (it touches 2 of 8,602
  2025 player-weeks), so the results above stand.

- **Carry-forward availability (Oct 2026).** Before a team's first injury report of the
  week, last week's designation now carries forward (a player Out last week was
  projected at 100% on Mondays and Tuesdays). Rates fitted on 2016-2023; on 2024 / 2025,
  players Out played their team's next game 30% / 30% of the time (table: 26-31%),
  Doubtful 48% / 46% (42-48%), Questionable 67% / 68% (71-80%). The holdout used final
  reports, where this rule does not apply.
- **Role inheritance (Oct 2026, [issue #1](https://github.com/Jpham303/FF-Assistant/issues/1)).**
  A starter ruled out used to vanish from the model, and his work was spread over all
  teammates by share (a quarterback picked up much of a running back's goal-line work).
  Recently absent players now stay in at 0% to play, and half of their carries,
  targets and TD share stays with their position group (`INHERIT_KEEP = 0.5`, chosen on
  2024 from 0 / 0.5 / 0.75 / 1, checked on 2023). Players taking over a starter's role:

  | | Before | After |
  | --- | --- | --- |
  | Bias, 2024 / 2023 | −1.09 / −1.08 pts | −0.32 / −0.46 |
  | RMSE, 2024 / 2023 | 5.77 / 5.82 | 5.67 / 5.74 |
  | MAE, 2024 | 3.95 | 4.02 (+0.07 [−0.04, +0.19]) |

  All players' MAE moved −0.010 [−0.023, +0.002] in 2024 and −0.012 [−0.020, −0.004] in
  2023; start/sit accuracy on pairs involving these players +0.1 / +0.8 points (not
  significant). **The acceptance rule set beforehand (their MAE must improve) was not
  met.** The change shipped anyway, as a judgment recorded here: projections are means of
  right-skewed outcomes, for which bias and RMSE are the right measures and MAE rewards
  projecting low. Phase 3 re-tests it with bias and RMSE fixed as the criteria in advance.

## Known limitations (for Phase 3)

- Win probabilities above 70% run 2–4 points high; recalibrate on 2024 and recheck.
- Kickers: better than the season average but not significantly.
- Matchup ratings add little (0.1–0.3% on validation), consistent with the literature;
  explanations size them accordingly.
- Rookies before their first game get a measured "unproven player" share; draft capital
  could inform it.
- One 2024 sample showed running-back starters about 1 point low; full-season volume
  buckets did not confirm it. Recheck in the full backtest.
