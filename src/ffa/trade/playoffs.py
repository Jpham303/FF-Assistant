"""Playoff odds: the rest of the fantasy regular season, simulated many times.

Each remaining week, every team scores its best lineup's expected points plus noise (sd
from its starters' projected spreads, treated as independent), and plays its scheduled
opponent from Sleeper's matchups. The current week, if under way, starts from the points
already scored. The top `playoff_teams` by wins (ties count half), then points for, make
the playoffs (Sleeper's default tiebreak).

Title odds (Oct 28): each simulated season goes on through the league's bracket. Seeds by
the same ranking; a fixed bracket (Sleeper's default: no re-seeding), byes for the top
seeds when the field isn't a power of 2 (6 teams: seeds 1-2), each round over its
playoff weeks (Sleeper's playoff_round_type: 0 one week a round, 1 a two-week final, 2
two weeks a round), points summed over a round's weeks, ties to the higher seed.
"""

from __future__ import annotations

import numpy as np

N_SEASONS = 5000


def odds(records: dict[str, tuple[float, float]], weekly: dict[str, dict[int, tuple[float, float]]],
         schedule: dict[int, list[tuple[str, str]]], playoff_teams: int,
         n: int = N_SEASONS, seed: int = 0) -> dict[str, float]:  # fmt: skip
    """team -> P(makes the playoffs). See `season`."""
    out = season(records, weekly, schedule, playoff_teams, n=n, seed=seed)
    return {t: v["playoffs"] for t, v in out.items()}


def bracket_order(size: int) -> list[int]:
    """Seed in each bracket position, top to bottom: 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        m = 2 * len(order) + 1
        order = [x for s in order for x in (s, m - s)]
    return order


def round_weeks(weeks: list[int], rounds: int, round_type: int) -> list[list[int]] | None:
    """The playoff weeks each round is played over, or None if there aren't enough."""
    lengths = (
        [2] * rounds if round_type == 2 else [1] * (rounds - 1) + [2 if round_type == 1 else 1]
    )
    weeks = sorted(weeks)
    if sum(lengths) > len(weeks):
        return None
    out, i = [], 0
    for n_ in lengths:
        out.append(weeks[i : i + n_])
        i += n_
    return out


def season(records: dict[str, tuple[float, float]],
           weekly: dict[str, dict[int, tuple[float, float]]],
           schedule: dict[int, list[tuple[str, str]]], playoff_teams: int,
           playoff_weeks: list[int] | None = None, round_type: int = 0,
           n: int = N_SEASONS, seed: int = 0) -> dict[str, dict[str, float | None]]:  # fmt: skip
    """records: team -> (wins so far, points for so far); weekly: team -> week ->
    (expected points, variance), regular season and playoff weeks; schedule: week ->
    [(team a, team b), ...] (regular season). Returns team -> {"playoffs": P(makes
    the playoffs), "title": P(wins the championship), None if the bracket's weeks
    aren't all projected}."""
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
    title = _title(score, teams, weekly, playoff_teams, playoff_weeks or [], round_type, rng)
    return {t: {"playoffs": round(float(made[:, ix[t]].mean()), 3),
                "title": None if title is None else round(float(title[ix[t]]), 3)}
            for t in teams}  # fmt: skip


def _title(score: np.ndarray, teams: list[str], weekly: dict, playoff_teams: int,
           weeks: list[int], round_type: int, rng) -> np.ndarray | None:  # fmt: skip
    """P(champion) per team, from each simulated season's seeding."""
    n, t = score.shape
    field_ = min(int(playoff_teams or 0), t)
    if field_ < 1:
        return None
    seeds = np.argsort(-score, axis=1, kind="stable")  # seeds[:, 0] = the 1 seed's team
    if field_ == 1:  # one playoff spot: the top seed is the champion
        return np.array([(seeds[:, 0] == i).mean() for i in range(t)])
    if not weeks:
        return None
    size = 1 << (field_ - 1).bit_length()
    rounds = size.bit_length() - 1
    plan = round_weeks(weeks, rounds, round_type)
    if plan is None or any(w not in weekly.get(tm, {}) for tm in teams for r in plan for w in r):
        return None
    rank = np.empty_like(seeds)
    rank[np.arange(n)[:, None], seeds] = np.arange(t)[None, :]
    slots = np.full((n, size), -1)
    for pos, s_ in enumerate(bracket_order(size)):
        if s_ <= field_:
            slots[:, pos] = seeds[:, s_ - 1]
    rows = np.arange(n)
    for wks in plan:
        tot = np.zeros((n, t))
        for w in wks:
            mu = np.array([weekly[tm][w][0] for tm in teams])
            sd = np.sqrt([max(weekly[tm][w][1], 0.0) for tm in teams])
            tot += mu + sd * rng.standard_normal((n, t))
        a, b = slots[:, 0::2], slots[:, 1::2]
        pa = np.where(a >= 0, tot[rows[:, None], np.maximum(a, 0)], -np.inf)
        pb = np.where(b >= 0, tot[rows[:, None], np.maximum(b, 0)], -np.inf)
        ra = np.where(a >= 0, rank[rows[:, None], np.maximum(a, 0)], t)
        rb = np.where(b >= 0, rank[rows[:, None], np.maximum(b, 0)], t)
        a_wins = (b < 0) | ((a >= 0) & ((pa > pb) | ((pa == pb) & (ra < rb))))
        slots = np.where(a_wins, a, b)
    champ = slots[:, 0]
    return np.array([(champ == i).mean() for i in range(t)])
