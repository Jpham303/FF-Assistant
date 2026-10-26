"""Playoff odds: the rest of the fantasy regular season, simulated many times.

Each remaining week, every team scores its best lineup's expected points plus noise (sd
from its starters' projected spreads, treated as independent), and plays its scheduled
opponent from Sleeper's matchups. The current week, if under way, starts from the points
already scored. The top `playoff_teams` by wins (ties count half), then points for, make
the playoffs (Sleeper's default tiebreak).
"""

from __future__ import annotations

import numpy as np

N_SEASONS = 5000


def odds(records: dict[str, tuple[float, float]], weekly: dict[str, dict[int, tuple[float, float]]],
         schedule: dict[int, list[tuple[str, str]]], playoff_teams: int,
         n: int = N_SEASONS, seed: int = 0) -> dict[str, float]:  # fmt: skip
    """records: team -> (wins so far, points for so far); weekly: team -> week ->
    (expected points, variance); schedule: week -> [(team a, team b), ...].
    Returns team -> P(makes the playoffs)."""
    teams = sorted(records)
    ix = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    wins = np.tile(np.array([records[t][0] for t in teams], float), (n, 1))
    pf = np.tile(np.array([records[t][1] for t in teams], float), (n, 1))
    for w in sorted(schedule):
        mu = np.array([weekly.get(t, {}).get(w, (0.0, 0.0))[0] for t in teams])
        sd = np.sqrt([max(weekly.get(t, {}).get(w, (0.0, 0.0))[1], 0.0) for t in teams])
        pts = mu + sd * rng.standard_normal((n, len(teams)))
        pf += pts
        for a, b in schedule[w]:
            if a not in ix or b not in ix:
                continue
            ia, ib = ix[a], ix[b]
            wins[:, ia] += (pts[:, ia] > pts[:, ib]) + 0.5 * (pts[:, ia] == pts[:, ib])
            wins[:, ib] += (pts[:, ib] > pts[:, ia]) + 0.5 * (pts[:, ia] == pts[:, ib])
    # Rank by wins, then points for: a tiny points-for term breaks ties within a win total.
    score = wins + pf / (pf.max() * 10 + 1)
    cut = np.sort(score, axis=1)[:, -playoff_teams][:, None]
    made = score >= cut
    return {t: round(float(made[:, ix[t]].mean()), 3) for t in teams}
