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

- **Quarterback starter (Oct 2026).** Each simulation used to hand the whole game to one
  active QB drawn with odds equal to his dropback share. Backups' shares are shrunk toward
  the position average (0.4-0.6 for a backup with little history), so an active starter
  took no snaps in 30-56% of simulations, which gave healthy starters a 10th percentile of
  0. Now the active QB with the highest share starts every simulation, and how much of
  the game he plays is drawn from the fitted distribution (2016-2023, games he played:
  under 50% of dropbacks in 2.8%, under 10% in 0.5%); the next QB gets the rest. Game
  statuses: the practice note from the official report is now kept when Sleeper supplies
  the designation (it was dropped, so midweek Questionable players all got the
  "practice unknown" rate). 2025 holdout, QBs who played:

  | | Before | After |
  | --- | --- | --- |
  | Starters (projected 12+, n=497): bias | −1.03 pts | +0.04 |
  | Starters: MAE | 6.25 | 6.09 |
  | Starters: below p10 / above p90 | 9.9% / 13.5% | 16.3% / 13.1% |
  | Backups who played (n=180): MAE | 4.34 | 4.86 |
  | All QBs: MAE / start-sit accuracy | 5.81 / 62.5% | 5.82 / 62.5% |

  The old ranges looked calibrated only because the spurious zeros filled the low tail.
  Starters who played the full game now land below p10 11.4% of the time and above p90
  14.1%; the rest of the excess is 2025's higher exit rate (3.8% of starter games under
  50% of dropbacks, which account for 22% of the misses below p10). Backups look worse
  because these rows are conditioned on the backup playing, which the old random starts
  happened to anticipate.

## Rest of season (Oct 2026)

Trade decisions need every remaining week, not just the next one. Rest-of-season
projections run the weekly model forward with two additions:

- **Team ratings** (`ffa.model.ratings`) give future games their lines: each team's
  offense and defense, plus home field, fitted by weighted ridge regression on this
  season's posted implied totals (weight 0.6 per week back, ridge 0.3, chosen on 2024).
  For games 2+ weeks ahead, their implied totals miss the eventual closing line by
  2.0 / 2.3 / 2.4 points on average in 2023 / 2024 / 2025. A team's average implied total
  so far misses by 2.5 / 2.6 / 2.9. **The 1.5-point target proposed in the trade engine
  spec was not met.** Lines that far out move with injuries and news no model has yet,
  and tuning plateaued near 2.3.
- **Availability horizon** (`ffa.model.horizon`): how often players in each spot today
  play their team's k-th game from now (k = 1-8, byes skipped). Spots: healthy,
  Questionable, Doubtful, Out, inactive, or reserve list (IR, PUP, NFI). Split by
  position and role: top producer (top 12 QB/TE, top 36 RB/WR by points per game),
  starter (70%+ of snaps), role player (30-70%), depth. Fitted on every completed season.
  For example (2016-2025 fit), a player who is Out today plays his team's next game 29%
  of the time and the one after 47%. A starting WR on a reserve list plays 7% of the next
  game and 33% eight games out.

Backtest: from weeks 4, 8 and 12, each player's projected points for the rest of the
season through week 17 (standard half-PPR), against what happened. Missed games count
as 0. Baselines: points per game so far × team games left (season-average pace), and
the same for the last 4 games (last-4 pace). Only data known at the cutoff week was used;
lines after it were masked. 2024 was used for tuning, 2025 is the holdout. MAE of the
rest-of-season total; edge = baseline MAE − model MAE, with a 95% bootstrap interval.

Fantasy-relevant players (top 12 QB/TE, top 36 RB/WR by points per game at the cutoff),
2025 holdout:

| Pos | Model MAE | Edge vs season-avg pace | Edge vs last-4 pace | Rank corr. model / season-avg / last-4 |
| --- | --- | --- | --- | --- |
| QB | 40.3 | +17.3 [+4.4, +31.4] | +19.2 [+5.1, +34.0] | 0.23 / 0.20 / 0.38 |
| RB | 30.9 | +7.7 [+2.3, +12.7] | +10.4 [+4.5, +16.2] | 0.76 / 0.57 / 0.55 |
| WR | 29.5 | +17.9 [+11.3, +24.6] | +18.7 [+12.2, +25.1] | 0.54 / 0.32 / 0.37 |
| TE | 21.1 | +18.3 [+6.9, +30.1] | +21.8 [+10.4, +33.3] | 0.61 / 0.00 / −0.01 |

All players with 2+ games: the model wins at every position in 2025, by 4.8 to 14.5
points, every interval above zero (re-run after the bye-week fix below: 4.0 to 11.1). **Gate passed:** the trade engine was cleared to
build on these projections only if they beat season-average pace on the 2025 holdout.

Not met, or not clean:

- 2025 top-12 QB ranking: last-4 pace ranked them better (0.38 vs 0.23; 36 player-cutoffs,
  noisy). The proposed criterion "higher rank correlation than both paces" fails in this cell.
- 2024 (tuning season) top-12 QBs: the model's error was 8.6 points larger than
  season-average pace (interval −23.1 to +5.4), with a −32 point bias. Those QBs averaged
  about 21 points a game when they played, against 18 projected. After the bye-week fix:
  9.1 points larger, −33 bias.

Bugs found and fixed during the backtest and the first live run:

- The line fill left future spreads and totals empty. The team-volume model then ran
  future weeks with no lines, cutting projected volume. A guard now refuses to simulate a
  game without lines.
- A player who missed practice after being Out, with no designation yet, was treated as
  healthy for the weeks after. Today's category now comes from this week's full picture:
  the combined designation, or the chance to play when no designation exists yet.
- Future weeks re-estimated each player as if his history were older (recency weights
  measured from the projected week), pulling stars toward average: Bijan Robinson slid
  from 22.6 to 19.4 points if active between weeks 8 and 17 with nothing in his schedule
  to explain it. Estimates are now made once, as of the current week, and held flat, as
  the spec says. The table above is from the fixed code (margins within 0.5 points of
  the earlier run; top-36 RB bias −4.4 → −2.8 in 2024).
- Teams on bye at the cutoff week were missing entirely (found live, Oct 26: a trade was
  valued against a player with no projection). nflverse's weekly rosters leave out teams
  on bye, and the roster step used only the file's latest week. Each team now uses its own
  latest roster week. Three of the four affected cutoffs gained 12-20% more rows (2024 week 8 had no byes); the table above is
  the re-run. The gate still passes: the model beats both paces at every position in 2025.
  The trade engine also now refuses to value a player with no projection instead of
  counting him as 0.

## Known limitations (for Phase 3)

- Win probabilities above 70% run 2–4 points high; recalibrate on 2024 and recheck.
- QB ranges: 16% of 2025 starter outcomes fell below p10 (target 10%). Fit the exit rate
  on recent seasons and check QB scoring dispersion on 2024.
- Rest of season: player shares and efficiency are held flat (no role trends), and the
  availability horizon doesn't know injury type or how long a player has been on IR.
- Practice trend (Wednesday-Friday) isn't used: nflverse keeps only the latest practice
  status. Our daily snapshots will allow fitting it after a season.
- Kickers: better than the season average but not significantly.
- Matchup ratings add little (0.1–0.3% on validation), consistent with the literature;
  explanations size them accordingly.
- Rookies before their first game get a measured "unproven player" share; draft capital
  could inform it.
- One 2024 sample showed running-back starters about 1 point low; full-season volume
  buckets did not confirm it. Recheck in the full backtest.
