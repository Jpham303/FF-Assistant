"""Start/sit answers from stored simulations: projections, comparisons, lineups.

Every number returned here comes from the simulations or the model inputs behind them,
so an explanation built on this output never needs a number from anywhere else. Drivers
are plain-language statements, each carrying the figure it rests on.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import permutations
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ffa.config import settings
from ffa.ingest.ids import normalize_name
from ffa.leagues.base import SLOT_ELIGIBILITY
from ffa.scoring import breakdown, score
from ffa.scoring.rules import ScoringRules
from ffa.warehouse import duck

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
    # Per team, mean per simulated game of carries, targets and TDs (all players), so a
    # player's realized share can be shown; and teammates unlikely to play this week.
    usage: pl.DataFrame | None = None
    absent: pl.DataFrame | None = None


@dataclass
class PlayerView:
    key: str
    name: str
    team: str
    position: str
    opponent: str | None
    mean: float
    median: float
    mean_if_active: float | None
    p10: float
    p90: float
    play_probability: float
    injury_status: str | None
    injury_source: str | None
    injury_updated_at: str | None
    injury_fetched_at: str | None
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
    """Simulations, inputs and game context for `keys`, read from this week's files only."""
    try:
        sims_t = duck.week(con, "projection_sims", season, week)
        inp_t = duck.week(con, "projection_inputs", season, week)
        teams_t = duck.week(con, "projection_teams", season, week)
    except duck.MissingPartition:
        empty = pl.DataFrame(schema={"key": pl.Utf8, "team": pl.Utf8})
        return Week(season, week, empty, empty, empty)

    def get(table: str, extra: str = "") -> pl.DataFrame:
        return con.sql(f"select * exclude (_ingested_at) from {table} where true {extra}").pl()

    in_list = ", ".join("'" + k.replace("'", "''") + "'" for k in keys) or "''"
    inputs = get(inp_t, f"and key in ({in_list})")
    teams_in = ", ".join(f"'{t}'" for t in inputs["team"].unique().to_list()) or "''"
    usage = con.sql(
        f"""select team, avg(ra) as rush_att, avg(tg) as rec_tgt, avg(rt) as rush_td,
                   avg(ct) as rec_td
            from (select team, sim, sum(rush_att) ra, sum(rec_tgt) tg, sum(rush_td) rt,
                         sum(rec_td) ct
                  from {sims_t} where team in ({teams_in}) group by team, sim)
            group by team"""
    ).pl()
    absent = con.sql(
        f"""select key, team, name, position, p_play, car_share_s, tgt_share_s
            from {inp_t}
            where team in ({teams_in}) and p_play < 0.5
              and (coalesce(car_share_s, 0) >= 0.15 or coalesce(tgt_share_s, 0) >= 0.12)"""
    ).pl()
    return Week(
        season, week,
        get(sims_t, f"and key in ({in_list})"),
        inputs,
        get(teams_t),
        usage, absent,
    )  # fmt: skip


def search(con, season: int, week: int, query: str, position: str | None = None,
           limit: int = 5) -> list[dict]:  # fmt: skip
    """Best name matches among all active NFL players and defenses, typos allowed.

    Each result has `match` (1.0 = exact) and `projected` (whether the player has a
    projection this week; False on byes). Callers must not treat a close match as the
    player asked for: a bye-week player must come back as "not playing", never be swapped
    for a similar name."""
    try:
        sims_t = duck.week(con, "projection_sims", season, week)
        proj = set(con.sql(f"select distinct key from {sims_t}").pl()["key"].to_list())
    except duck.MissingPartition:
        proj = set()
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


OUT_THRESHOLD = 0.02  # below this chance to play, treat the player as ruled out
PRACTICE_SHORT = {
    "Did Not Participate In Practice": "did not practice",
    "Limited Participation in Practice": "limited in practice",
    "Full Participation in Practice": "full practice",
}


def when(ts) -> str | None:
    """A timestamp as local time, e.g. 'Fri Oct 16 5:02 PM CDT'."""
    if ts is None:
        return None
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    local = ts.astimezone(ZoneInfo(settings.timezone))
    return local.strftime("%a %b %d %I:%M %p %Z").replace(" 0", " ")


def provenance(row: dict) -> str:
    """Which source the injury information came from and how fresh it is."""
    src = row.get("injury_source") or "nflverse"
    parts = []
    if row.get("injury_updated_at") is not None:
        parts.append(f"updated {when(row['injury_updated_at'])}")
    if row.get("injury_fetched_at") is not None:
        parts.append(f"fetched {when(row['injury_fetched_at'])}")
    return f"{src}, {'; '.join(parts)}" if parts else src


def injury_line(row: dict) -> str | None:
    """The injury picture with its source and time, or None for a healthy player."""
    status, practice = row.get("injury_status"), row.get("practice")
    p = float(row.get("p_play", 1.0) or 0.0)
    roster = row.get("roster_status")
    if roster in ("INA", "RES") and not status:
        what = "On injured reserve" if roster == "RES" else "Inactive this week"
        return f"{what} [team roster, nflverse]; plays in 0% of simulations"
    injury = f" ({row['injury'].lower()})" if row.get("injury") else ""
    prev = "; was Out last week" if row.get("prev_out") else ""
    if status:
        prac = f", {PRACTICE_SHORT.get(practice, practice).lower()}" if practice else ""
        head = "OUT" if status == "Out" else status
        return (f"Injury: {head}{injury}{prac}{prev} [{provenance(row)}]; "
                f"plays in {p * 100:.0f}% of simulations")  # fmt: skip
    if p < 1.0 and not row.get("report_published", True) and row.get("prev_status"):
        wk = f" in week {row['prev_week']}" if row.get("prev_week") else " last week"
        return (f"No injury report yet this week; was {row['prev_status']}{wk} "
                f"[nflverse, fetched {when(row.get('injury_fetched_at')) or 'time unknown'}]; "
                f"plays in {p * 100:.0f}% of simulations (players in that spot played their "
                "next game at this rate historically; updates with the team's first report, "
                "usually Wednesday)")  # fmt: skip
    if p < 1.0 and practice:
        return (f"No game status yet, but {PRACTICE_SHORT.get(practice, practice)} this week"
                f"{injury}{prev} [{provenance(row)}]; plays in {p * 100:.0f}% of simulations. "
                "Statuses usually come with Friday's final report")  # fmt: skip
    return None


def _usage_lines(row: dict, position: str, usage: dict | None) -> list[str]:
    """Carries, targets and TDs as simulated, next to the player's usual share. The
    simulation hands an absent teammate's work to active players, so with someone out the
    usual share understates the role; show both, and say who is out."""
    out = []
    car, tgt = row.get("car_share_s") or 0, row.get("tgt_share_s")
    if not usage:  # no simulation summary: usual shares only, labelled as such
        if position in ("RB", "QB") and car > 0.02:
            out.append(f"Usual share of designed runs: {car * 100:.1f}%")
        if position in ("RB", "WR", "TE") and tgt is not None:
            out.append(f"Usual share of team targets: {tgt * 100:.1f}%")
        return out
    me, team = usage["player"], usage["team"]

    def gone(kind: str) -> str:  # teammates out whose carries/targets are being shared
        names = usage.get(kind) or []
        return f" with {', '.join(names)} unlikely to play" if names else ""

    def pct(a: float, b: float) -> str:
        return f"{a / b * 100:.0f}%" if b > 0 else "n/a"

    if position in ("RB", "QB") and car > 0.02:
        share = pct(me["rush_att"], team["rush_att"])
        out.append(f"Projected {me['rush_att']:.1f} carries, {share} of the team's rushing "
                   f"attempts{gone('absent_car')} (usual share of designed runs "
                   f"{car * 100:.1f}%)")  # fmt: skip
        out.append(f"Projected rushing TDs: {me['rush_td']:.2f} of the team's "
                   f"{team['rush_td']:.2f} ({pct(me['rush_td'], team['rush_td'])})")  # fmt: skip
    if position in ("RB", "WR", "TE") and tgt is not None:
        share = pct(me["rec_tgt"], team["rec_tgt"])
        out.append(f"Projected {me['rec_tgt']:.1f} targets, {share} of team targets"
                   f"{gone('absent_tgt')} (usual share {tgt * 100:.1f}%)")  # fmt: skip
        out.append(f"Projected receiving TDs: {me['rec_td']:.2f} of the team's "
                   f"{team['rec_td']:.2f} ({pct(me['rec_td'], team['rec_td'])})")  # fmt: skip
    return out


def drivers(row: dict, team: dict | None, position: str,
            usage: dict | None = None) -> list[str]:  # fmt: skip
    """Plain-language reasons, each with the number it rests on. `usage` holds what the
    simulation actually gave the player and his team (see `_usage_lines`)."""
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
    line = injury_line(row)
    if line:
        out.append(line)
    out += _usage_lines(row, position, usage)
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


def _usage(g: pl.DataFrame, team: str, key: str, usage: dict, absent) -> dict | None:
    """The player's simulated volume (in games he plays) and his team's, for drivers."""
    if team not in usage:
        return None
    played = g.filter(pl.col("active")) if "active" in g.columns else g
    played = played if played.height else g
    me = {c: float(played[c].mean()) for c in ("rush_att", "rec_tgt", "rush_td", "rec_td")
          if c in played.columns}  # fmt: skip
    if len(me) < 4:
        return None

    def names(col: str, floor: float) -> list[str]:
        if absent is None:
            return []
        rows = absent.filter(
            (pl.col("team") == team) & (pl.col("key") != key)
            & (pl.col(col).fill_null(0) >= floor)
        )  # fmt: skip
        return [
            f"{r['name']} ({r['p_play'] * 100:.0f}% to play)" for r in rows.iter_rows(named=True)
        ]

    return {
        "player": me,
        "team": usage[team],
        "absent_car": names("car_share_s", 0.15),
        "absent_tgt": names("tgt_share_s", 0.12),
    }


def _if_active(pts: pl.DataFrame, g: pl.DataFrame, key: str) -> float | None:
    """Mean points in the simulations where he plays (the plays-only projection)."""
    if "active" not in g.columns:
        return None
    act = g.filter(pl.col("active").fill_null(True)).select("sim")
    p = pts.filter(pl.col("key") == key).join(act, on="sim", how="semi")["fpts"]
    return float(p.mean()) if p.len() else None


def player_views(w: Week, rules: ScoringRules) -> dict[str, PlayerView]:
    if w.sims.height == 0:
        return {}
    pts = score(w.sims, rules).select("key", "sim", "fpts")
    terms = breakdown(w.sims, rules, ids=["key", "sim"])
    n = int(w.sims["sim"].max()) + 1
    inputs = {r["key"]: r for r in w.inputs.iter_rows(named=True)}
    teams = {r["team"]: r for r in w.teams.iter_rows(named=True)}
    usage = {r["team"]: r for r in w.usage.iter_rows(named=True)} if w.usage is not None else {}
    absent = w.absent if w.absent is not None else None
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
            mean_if_active=_if_active(pts, g, key),
            p10=float(f.quantile(0.1)), p90=float(f.quantile(0.9)),
            play_probability=float(row.get("p_play", 1.0) or 0.0),
            injury_status=row.get("injury_status"),
            injury_source=row.get("injury_source") if row.get("injury_status") or
            row.get("practice") else None,
            injury_updated_at=when(row.get("injury_updated_at")),
            injury_fetched_at=when(row.get("injury_fetched_at")),
            expected_stats={STAT_LABELS[s]: round(float(g[s].mean()), 2)
                            for s in POSITION_STATS.get(pos, []) if s in g.columns},
            points_by_rule={t: round(float(p), 2) for t, p in rule_pts.iter_rows()},
            drivers=drivers(row | {"team": meta["team"]}, team, pos,
                            _usage(g, meta["team"], key, usage, absent)),
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
    # A ruled-out player is never the recommendation, whatever his mean says.
    playing = [v for v in ranked if v.play_probability >= OUT_THRESHOLD] or ranked
    ranked = playing + [v for v in ranked if v not in playing]
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    p_top = next((x["p_a_beats_b"] for x in pairwise
                  if second and x["a"] == top.key and x["b"] == second.key), 1.0)  # fmt: skip
    notes = []
    for v in ranked:
        if v.play_probability < OUT_THRESHOLD:
            notes.append(f"{v.name} is ruled OUT or very unlikely to play "
                         f"({v.play_probability * 100:.0f}% in simulations; "
                         f"{v.injury_source or 'nflverse'}, "
                         f"{v.injury_updated_at or v.injury_fetched_at or 'time unknown'}). "
                         "Don't start him.")  # fmt: skip
    if second and p_top < CLOSE_CALL and second.play_probability >= OUT_THRESHOLD:
        notes.append(f"Close call: {top.name} outscores {second.name} in only "
                     f"{p_top * 100:.0f}% of simulations.")  # fmt: skip
    for v in ranked:
        if OUT_THRESHOLD <= v.play_probability < 0.95:
            stamp = v.injury_updated_at or v.injury_fetched_at or "time unknown"
            notes.append(f"{v.name} is {v.injury_status or 'not designated yet'}: projection "
                         f"includes a {(1 - v.play_probability) * 100:.0f}% chance he sits "
                         f"({v.injury_source or 'nflverse'}, {stamp}).")  # fmt: skip
    close = bool(second and p_top < CLOSE_CALL and second.play_probability >= OUT_THRESHOLD)
    return Comparison(
        season=w.season, week=w.week, players=ranked, pairwise=pairwise,
        recommendation=top.key, close_call=close, notes=notes,
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
        cands = [
            v
            for v in views.values()
            if v.position in eligible and v.key not in used and v.play_probability >= OUT_THRESHOLD
        ]
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
