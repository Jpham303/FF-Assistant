"""Start/sit answers from stored simulations: projections, comparisons, lineups.

Every number returned here comes from the simulations or the model inputs behind them,
so an explanation built on this output never needs a number from anywhere else. Drivers
are plain-language statements, each carrying the figure it rests on.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import polars as pl

from ffa.ingest.ids import normalize_name
from ffa.leagues.base import SLOT_ELIGIBILITY
from ffa.scoring import breakdown, score
from ffa.scoring.rules import ScoringRules

CLOSE_CALL = 0.55  # below this P(A beats B), call it a coin flip
STAT_LABELS = {
    "pass_yd": "passing yards", "pass_td": "passing TDs", "pass_int": "interceptions",
    "rush_att": "carries", "rush_yd": "rushing yards", "rush_td": "rushing TDs",
    "rec_tgt": "targets", "rec": "receptions", "rec_yd": "receiving yards",
    "rec_td": "receiving TDs", "fgm": "field goals", "fgm_yds": "field-goal yards",
    "xpm": "extra points", "sack": "sacks", "int": "interceptions", "fum_rec": "fumble recoveries",
    "def_td": "defensive TDs", "pts_allow": "points allowed",
}  # fmt: skip
POSITION_STATS = {
    "QB": ["pass_yd", "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td"],
    "RB": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td"],
    "WR": ["rec_tgt", "rec", "rec_yd", "rec_td", "rush_att", "rush_yd"],
    "TE": ["rec_tgt", "rec", "rec_yd", "rec_td"],
    "K": ["fgm", "fgm_yds", "xpm"],
    "DEF": ["sack", "int", "fum_rec", "def_td", "pts_allow"],
}
MATCHUP_TEXT = {
    "m_tgt": "{pos}s draw {pct} {dir} targets against {opp} than they usually do",
    "m_catch": "{pos}s catch {pct} {dir} of their targets against {opp} than usual",
    "m_ypr": "{pos}s gain {pct} {dir} yards per catch against {opp} than usual",
    "m_ypc": "{pos}s gain {pct} {dir} yards per carry against {opp} than usual",
}


@dataclass
class Week:
    season: int
    week: int
    sims: pl.DataFrame  # simulated stat lines for the requested players
    inputs: pl.DataFrame  # model inputs per player
    teams: pl.DataFrame  # game context per team


@dataclass
class PlayerView:
    key: str
    name: str
    team: str
    position: str
    opponent: str | None
    mean: float
    median: float
    p10: float
    p90: float
    play_probability: float
    injury_status: str | None
    expected_stats: dict[str, float]
    points_by_rule: dict[str, float]
    drivers: list[str]
    p_best: float | None = None


@dataclass
class Comparison:
    season: int
    week: int
    players: list[PlayerView]
    pairwise: list[dict]  # {"a", "b", "p_a_beats_b"}
    recommendation: str
    close_call: bool
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- loading


def load_week(con, season: int, week: int, keys: list[str]) -> Week:
    def get(table: str, extra: str = "") -> pl.DataFrame:
        return con.sql(
            f"select * exclude (_ingested_at) from {table} where season = {int(season)} "
            f"and week = {int(week)} {extra}"
        ).pl()

    in_list = ", ".join("'" + k.replace("'", "''") + "'" for k in keys) or "''"
    return Week(
        season, week,
        get("projection_sims", f"and key in ({in_list})"),
        get("projection_inputs", f"and key in ({in_list})"),
        get("projection_teams"),
    )  # fmt: skip


def search(con, season: int, week: int, query: str, position: str | None = None,
           limit: int = 5) -> list[dict]:  # fmt: skip
    """Best name matches among all active NFL players and defenses, typos allowed.

    Each result has `match` (1.0 = exact) and `projected` (whether the player has a
    projection this week; False on byes). Callers must not treat a close match as the
    player asked for: a bye-week player must come back as "not playing", never be swapped
    for a similar name."""
    proj = set(
        con.sql(
            f"""select distinct key from projection_sims
                where season = {int(season)} and week = {int(week)}"""
        )
        .pl()["key"]
        .to_list()
    )
    players = con.sql(
        f"""select gsis_id as key, name, team, position from player_ids
            where position in ('QB', 'RB', 'WR', 'TE', 'K') and last_season >= {int(season) - 1}"""
    ).pl()
    teams = (
        con.sql(f"select distinct team from team_games where season = {int(season)}")
        .pl()["team"]
        .to_list()
    )
    dst = pl.DataFrame({"key": [f"DST_{t}" for t in teams], "name": [f"{t} D/ST" for t in teams],
                        "team": teams, "position": ["DEF"] * len(teams)})  # fmt: skip
    universe = pl.concat([players, dst])
    if position:
        universe = universe.filter(pl.col("position") == position.upper())
    q = normalize_name(query)
    code = query.strip().upper()

    def score_row(name: str, team: str | None, pos: str) -> float:
        n = normalize_name(name)
        if pos == "DEF" and code in (team, f"{team} D/ST", f"{team} DST"):
            return 1.0
        if n == q:
            return 1.0
        if q and (q in n or n in q):
            return 0.9
        return difflib.SequenceMatcher(None, q, n).ratio()

    scored = [
        r | {"match": round(score_row(r["name"] or "", r["team"], r["position"]), 3),
             "projected": r["key"] in proj}
        for r in universe.iter_rows(named=True)
    ]  # fmt: skip
    scored.sort(key=lambda r: (r["match"], r["projected"]), reverse=True)
    return [r for r in scored if r["match"] >= 0.6][:limit]


# ---------------------------------------------------------------- views


def _pct(x: float) -> str:
    return f"{abs(x - 1) * 100:.0f}%"


def drivers(row: dict, team: dict | None, position: str) -> list[str]:
    """Plain-language reasons, each with the number it rests on."""
    out = []
    if team:
        # spread_for is the team's expected margin (positive = favored). Written out in
        # words: a signed "spread" reads the opposite way to bettors (-3.5 = favored).
        m = team["spread_for"]
        line = (f"favored by {m:.1f}" if m > 0 else f"a {-m:.1f}-point underdog" if m < 0
                else "a pick'em")  # fmt: skip
        out.append(
            f"{row['team']} is {line} vs {team['opponent']} (total {team['total_line']:.1f}), "
            f"implied to score {team['implied']:.1f} points; "
            f"projected {team['mu_plays']:.0f} plays, {team['p_pass'] * 100:.0f}% passes"
        )
    status = row.get("injury_status")
    if status:
        practice = f", {row['practice'].lower()}" if row.get("practice") else ""
        out.append(f"Injury report: {status}{practice}; plays in "
                   f"{row['p_play'] * 100:.0f}% of simulations")  # fmt: skip
    if position in ("RB", "QB") and (row.get("car_share_s") or 0) > 0.02:
        out.append(f"Expected share of designed runs: {row['car_share_s'] * 100:.1f}% "
                   f"(red zone {row['rz_car_share_s'] * 100:.1f}%)")  # fmt: skip
    if position in ("RB", "WR", "TE") and row.get("tgt_share_s") is not None:
        out.append(f"Expected share of team targets: {row['tgt_share_s'] * 100:.1f}% "
                   f"(red zone {row['rz_tgt_share_s'] * 100:.1f}%)")  # fmt: skip
    if position == "QB" and row.get("db_share_s") is not None:
        out.append(f"Expected share of team dropbacks: {row['db_share_s'] * 100:.0f}%")
    if position in ("WR", "TE", "RB") and row.get("catch_rate_s") is not None:
        out.append(f"Catches {row['catch_rate_s'] * 100:.0f}% of targets for "
                   f"{row['ypr_s']:.1f} yards per catch (career-weighted)")  # fmt: skip
    if (
        position in ("RB", "QB")
        and row.get("ypc_s") is not None
        and (row.get("car_share_s") or 0) > 0.02
    ):
        out.append(f"Gains {row['ypc_s']:.1f} yards per carry (career-weighted)")
    for col, text in MATCHUP_TEXT.items():
        v = row.get(col)
        if v is not None and abs(v - 1) >= 0.02:
            out.append("Matchup: " + text.format(
                pos=position, pct=_pct(v), dir="more" if v > 1 else "fewer" if "tgt" in col
                else "more" if v > 1 else "fewer", opp=row.get("opponent"),
            ) + f" (rating {v:.2f}; matchups are a small factor)")  # fmt: skip
    return out


def player_views(w: Week, rules: ScoringRules) -> dict[str, PlayerView]:
    if w.sims.height == 0:
        return {}
    pts = score(w.sims, rules).select("key", "sim", "fpts")
    terms = breakdown(w.sims, rules, ids=["key", "sim"])
    n = int(w.sims["sim"].max()) + 1
    inputs = {r["key"]: r for r in w.inputs.iter_rows(named=True)}
    teams = {r["team"]: r for r in w.teams.iter_rows(named=True)}
    views = {}
    for (key,), g in w.sims.group_by("key"):
        meta = g.row(0, named=True)
        pos = meta["position"]
        f = pts.filter(pl.col("key") == key)["fpts"]
        row = inputs.get(key, {})
        team = teams.get(meta["team"])
        rule_pts = (
            terms.filter(pl.col("key") == key).group_by("term")
            .agg((pl.col("points").sum() / n).alias("p")).sort("p", descending=True)
        )  # fmt: skip
        views[key] = PlayerView(
            key=key, name=meta["name"], team=meta["team"], position=pos,
            opponent=row.get("opponent") or (team or {}).get("opponent"),
            mean=float(f.mean()), median=float(f.median()),
            p10=float(f.quantile(0.1)), p90=float(f.quantile(0.9)),
            play_probability=float(row.get("p_play", 1.0) or 0.0),
            injury_status=row.get("injury_status"),
            expected_stats={STAT_LABELS[s]: round(float(g[s].mean()), 2)
                            for s in POSITION_STATS.get(pos, []) if s in g.columns},
            points_by_rule={t: round(float(p), 2) for t, p in rule_pts.iter_rows()},
            drivers=drivers(row | {"team": meta["team"]}, team, pos),
        )  # fmt: skip
    return views


def compare(w: Week, rules: ScoringRules) -> Comparison:
    """Start/sit between 2+ players: who to start, how likely each is to outscore the
    others (from simulations that share each game), and why."""
    views = player_views(w, rules)
    keys = list(views)
    pts = score(w.sims, rules).select("key", "sim", "fpts")
    mat = pts.pivot(on="key", index="sim", values="fpts").sort("sim")
    arr = mat.select(keys).to_numpy()
    best = arr.argmax(axis=1)
    for i, k in enumerate(keys):
        views[k].p_best = float((best == i).mean())
    pairwise = []
    for a, b in permutations(range(len(keys)), 2):
        p = float(((arr[:, a] > arr[:, b]) + 0.5 * (arr[:, a] == arr[:, b])).mean())
        pairwise.append({"a": keys[a], "b": keys[b], "p_a_beats_b": round(p, 3)})
    ranked = sorted(views.values(), key=lambda v: v.mean, reverse=True)
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    p_top = next((x["p_a_beats_b"] for x in pairwise
                  if second and x["a"] == top.key and x["b"] == second.key), 1.0)  # fmt: skip
    notes = []
    if second and p_top < CLOSE_CALL:
        notes.append(f"Close call: {top.name} outscores {second.name} in only "
                     f"{p_top * 100:.0f}% of simulations.")  # fmt: skip
    for v in ranked:
        if v.play_probability < 0.95:
            notes.append(f"{v.name} is {v.injury_status or 'uncertain'}: projection includes a "
                         f"{(1 - v.play_probability) * 100:.0f}% chance he sits.")  # fmt: skip
    return Comparison(
        season=w.season, week=w.week, players=ranked, pairwise=pairwise,
        recommendation=top.key, close_call=bool(second and p_top < CLOSE_CALL), notes=notes,
    )  # fmt: skip


# ---------------------------------------------------------------- lineups


def best_lineup(views: dict[str, PlayerView], slots: list[str]) -> list[dict]:
    """Fill starting slots to maximize expected points. Most restrictive slots first, so
    a flex never takes the only eligible QB; ties broken by projection."""
    order = sorted(range(len(slots)), key=lambda i: len(SLOT_ELIGIBILITY.get(slots[i], ())))
    used: set[str] = set()
    filled: dict[int, dict] = {}
    for i in order:
        eligible = SLOT_ELIGIBILITY.get(slots[i], frozenset())
        cands = [v for v in views.values() if v.position in eligible and v.key not in used]
        if not cands:
            filled[i] = {"slot": slots[i], "key": None, "name": None, "mean": 0.0}
            continue
        pick = max(cands, key=lambda v: v.mean)
        used.add(pick.key)
        filled[i] = {"slot": slots[i], "key": pick.key, "name": pick.name,
                     "mean": round(pick.mean, 2), "median": round(pick.median, 2)}  # fmt: skip
    return [filled[i] for i in range(len(slots))]


def as_dict(obj) -> dict:
    from dataclasses import asdict

    d = asdict(obj)
    return _round(d)


def _round(x):
    if isinstance(x, float):
        return round(x, 2) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x
