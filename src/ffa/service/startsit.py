"""Start/sit answers from stored simulations: projections, comparisons, lineups.

Every number returned here comes from the simulations or the model inputs behind them,
so an explanation built on this output never needs a number from anywhere else. Drivers
are plain-language statements, each carrying the figure it rests on.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import permutations
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import polars as pl

from ffa.config import settings
from ffa.ingest.ids import normalize_name
from ffa.leagues.base import SLOT_ELIGIBILITY
from ffa.scoring import breakdown, score
from ffa.scoring.rules import ScoringRules
from ffa.service import gamestate, ifactive
from ffa.warehouse import duck

CLOSE_CALL = 0.55  # below this P(A beats B), call it a coin flip
MIN_BOTH_ACTIVE = 100  # simulations where both play, needed to report the if-both-active odds
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
    forced: pl.DataFrame | None = None  # forced-active batches for low play chances
    kickoffs: dict[str, datetime] = field(default_factory=dict)  # team -> kickoff (UTC)


@dataclass
class PlayerView:
    key: str
    name: str
    team: str
    position: str
    opponent: str | None
    # Headline: the projection if he plays, with its range (null when not playing)
    proj_if_active: float | None
    median_if_active: float | None
    p10_if_active: float | None
    p25_if_active: float | None
    p75_if_active: float | None
    p90_if_active: float | None
    play_probability: float
    expected_points: float  # play_probability x proj_if_active: what lineups rank by
    active_sims: int  # simulations behind the if-active numbers
    forced_active_batch: bool  # if-active numbers from an extra forced-active batch
    expected_stats_if_active: dict[str, float]
    injury_status: str | None
    injury_source: str | None
    injury_updated_at: str | None
    injury_fetched_at: str | None
    expected_stats: dict[str, float]  # deprecated: all simulations, zeros included
    points_by_rule: dict[str, float]
    drivers: list[str]
    # Deprecated copies, kept for one release: mean == expected_points; median, p10 and
    # p90 over all simulations, zeros included.
    mean: float = 0.0
    median: float = 0.0
    p10: float = 0.0
    p90: float = 0.0
    p_best: float | None = None
    # For a player who may not play: when and where to confirm (statuses are fetched once
    # a day, so a later scratch isn't in the numbers).
    game_day_check: str | None = None
    # upcoming / in_progress / final (kickoff + 4 h); None when the kickoff is unknown.
    # actual_points: what Sleeper has awarded so far (set by the API for league players).
    game_status: str | None = None
    actual_points: float | None = None


@dataclass
class Comparison:
    season: int
    week: int
    players: list[PlayerView]
    pairwise: list[dict]  # {"a", "b", "p_a_beats_b"}
    recommendation: str
    close_call: bool
    notes: list[str] = field(default_factory=list)
    # What separates the recommendation from the runner-up, biggest first (points if
    # active by stat group, plus availability): [{"factor", "points", "favors", "detail"}]
    decision_drivers: list[dict] = field(default_factory=list)
    # Your chance of winning this week's fantasy matchup with each candidate starting
    # (set by the API when a league and your team are known)
    matchup: dict | None = None


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
    try:
        f_t = duck.week(con, "projection_sims_forced", season, week)
        forced = con.sql(
            f"select * exclude (_ingested_at) from {f_t} where forced_key in ({in_list})"
        ).pl()
    except duck.MissingPartition:
        forced = None
    from ffa.tracking.snapshots import kickoffs  # local: tracking imports this package

    try:  # kickoff times only feed the game-day check: never fail a projection over them
        ko = {r["team"]: r["kickoff"] for r in kickoffs(con, season, week).iter_rows(named=True)}
    except duckdb.Error:
        ko = {}
    return Week(
        season, week,
        get(sims_t, f"and key in ({in_list})"),
        inputs,
        get(teams_t),
        usage, absent, forced, ko,
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


DESIGNATION = {"IR": "injured reserve", "PUP": "PUP list", "Sus": "suspended",
               "NA": "not with team", "DNR": "did not report", "COV": "COVID list"}  # fmt: skip
RESERVE_LISTS = {"IR", "PUP", "NA", "Sus", "DNR"}
EARLIER_INJURY_DAYS = 120  # an injury on file older than this isn't this week's news


def injury_note(row: dict) -> str:
    """The injury on file, honestly dated. The official report's injury is this week's.
    Sleeper keeps a player's injury note until something replaces it, so its note can be
    from an earlier injury (Charbonnet's ACL from last season, Oct 27): with a start
    date it's dated, flagged when old; without one, on a reserve list it's called the
    injury on file."""
    text = (row.get("injury") or "").strip().lower()
    if not text:
        return ""
    if row.get("injury_source") != "Sleeper":
        return f" ({text})"
    since = row.get("injury_since")
    ref = row.get("injury_fetched_at") or datetime.now(UTC)
    try:
        start = datetime.fromisoformat(str(since)[:10]).replace(tzinfo=UTC) if since else None
    except ValueError:
        start = None
    if start is None:
        # Undated: a reserve listing (IR, PUP...) is where old notes linger; a weekly
        # designation's note is almost always this week's.
        long_term = row.get("designation") in RESERVE_LISTS or row.get("roster_status") == "RES"
        return f" (injury on file: {text})" if long_term else f" ({text})"
    age = (ref.replace(tzinfo=ref.tzinfo or UTC) - start).days
    if age > EARLIER_INJURY_DAYS:
        return f" ({text}, from an earlier injury: since {start:%b %Y})"
    return f" ({text}, since {start:%b %d})"


def injury_line(row: dict) -> str | None:
    """The injury picture with its source and time, or None for a healthy player."""
    status, practice = row.get("injury_status"), row.get("practice")
    p = float(row.get("p_play", 1.0) or 0.0)
    roster = row.get("roster_status")
    if roster in ("INA", "RES") and not status:
        what = "On injured reserve" if roster == "RES" else "Inactive this week"
        return f"{what} [team roster, nflverse]; plays in 0% of simulations"
    injury = injury_note(row)
    prev = "; was Out last week" if row.get("prev_out") else ""
    if status:
        prac = f", {PRACTICE_SHORT.get(practice, practice).lower()}" if practice else ""
        head = "OUT" if status == "Out" else status
        des = row.get("designation")
        if des and des not in (status, "Out") and status == "Out":
            head = f"OUT ({DESIGNATION.get(des, des)})"  # Sleeper's own word: IR, PUP...
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


INACTIVES_BEFORE = timedelta(minutes=90)


def game_day_check(p_play: float, status: str | None, kickoff: datetime | None,
                   now: datetime | None = None) -> str | None:  # fmt: skip
    """What to confirm before lineups lock, for a player who may not play (0 < p < 1).
    Game statuses are fetched once a day (Sleeper's limit), so a scratch announced later
    isn't in the projection."""
    if not 0 < p_play < 1:
        return None
    what = status or "Not certain to play"
    if kickoff is None:
        return (f"{what}: confirm he is active before lineups lock; game statuses here are "
                "fetched once a day and won't show a later scratch")  # fmt: skip
    now = now or datetime.now(UTC)
    if now >= kickoff:
        return None
    inactives = kickoff - INACTIVES_BEFORE
    return (f"{what}: check inactives, announced about 90 minutes before kickoff "
            f"({when(inactives)}). Game statuses here are fetched once a day and won't "
            "show a later scratch")  # fmt: skip


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
    game_keys = ("spread_for", "opponent", "total_line", "implied", "mu_plays", "p_pass")
    if team and all(team.get(k) is not None for k in game_keys):
        # spread_for is the team's expected margin (positive = favored). Written out in
        # words: a signed "spread" reads the opposite way to bettors (-3.5 = favored).
        m = team["spread_for"]
        line = (f"favored by {m:.1f}" if m > 0 else f"a {-m:.1f}-point underdog" if m < 0
                else "a pick'em")  # fmt: skip
        out.append(
            f"{row['team']} is {line} vs {team['opponent']} (total {team['total_line']:.1f}), "
            f"implied to score {team['implied']:.1f} points; "
            f"projected {team['mu_plays']:.0f} plays, {team['p_pass'] * 100:.0f}% passes"
            + (" (no betting line posted yet: spread and total estimated from team ratings "
               "fitted on this season's lines)" if team.get("lines_from_ratings") else "")
        )  # fmt: skip
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


def player_views(w: Week, rules: ScoringRules) -> dict[str, PlayerView]:
    if w.sims.height == 0:
        return {}
    sims = w.sims.with_columns(ifactive.active_flag(w.sims).alias("active"))
    pts = score(sims, rules).select("key", "sim", "fpts", "active")
    forced_pts = None
    if w.forced is not None and w.forced.height:
        forced_pts = score(w.forced, rules).select(pl.col("forced_key").alias("key"), "sim",
                                                   "fpts")  # fmt: skip
    inputs = {r["key"]: r for r in w.inputs.iter_rows(named=True)}
    play = {k: float(r.get("p_play") if r.get("p_play") is not None else 1.0)
            for k, r in inputs.items()}  # fmt: skip
    summary = {r["key"]: r for r in ifactive.summarize(pts, play, forced_pts)
               .iter_rows(named=True)}  # fmt: skip
    terms = breakdown(w.sims, rules, ids=["key", "sim"])
    n = int(w.sims["sim"].max()) + 1
    teams = {r["team"]: r for r in w.teams.iter_rows(named=True)}
    usage = {r["team"]: r for r in w.usage.iter_rows(named=True)} if w.usage is not None else {}
    absent = w.absent if w.absent is not None else None
    views = {}
    for (key,), g in sims.group_by("key"):
        meta = g.row(0, named=True)
        pos = meta["position"]
        f = pts.filter(pl.col("key") == key)["fpts"]
        sm = summary[key]
        row = inputs.get(key, {})
        team = teams.get(meta["team"])
        rule_pts = (
            terms.filter(pl.col("key") == key).group_by("term")
            .agg((pl.col("points").sum() / n).alias("p")).sort("p", descending=True)
        )  # fmt: skip
        stat_src = g.filter(pl.col("active"))
        if sm["forced_active_batch"] and w.forced is not None:
            stat_src = w.forced.filter(pl.col("forced_key") == key)
        rnd = lambda x: None if x is None else round(float(x), 2)  # noqa: E731
        views[key] = PlayerView(
            key=key, name=meta["name"], team=meta["team"], position=pos,
            opponent=row.get("opponent") or (team or {}).get("opponent"),
            proj_if_active=rnd(sm["proj_if_active"]),
            median_if_active=rnd(sm["median_if_active"]),
            p10_if_active=rnd(sm["p10_if_active"]), p25_if_active=rnd(sm["p25_if_active"]),
            p75_if_active=rnd(sm["p75_if_active"]), p90_if_active=rnd(sm["p90_if_active"]),
            play_probability=float(sm["play_probability"]),
            expected_points=float(sm["expected_points"]),
            active_sims=int(sm["active_sims"]),
            forced_active_batch=bool(sm["forced_active_batch"]),
            expected_stats_if_active={} if sm["proj_if_active"] is None else {
                STAT_LABELS[s]: round(float(stat_src[s].mean()), 2)
                for s in POSITION_STATS.get(pos, []) if s in stat_src.columns},
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
            game_day_check=game_day_check(
                float(sm["play_probability"]), row.get("injury_status"),
                w.kickoffs.get(meta["team"])),
            game_status=gamestate.status(w.kickoffs.get(meta["team"])),
            mean=float(sm["expected_points"]), median=float(f.median()),
            p10=float(f.quantile(0.1)), p90=float(f.quantile(0.9)),
        )  # fmt: skip
    return views


def disagreements(views: dict[str, PlayerView], pairwise: list[dict]) -> list[str]:
    """When the better player if both play is not the better start once the chance of
    sitting is counted, say so in plain words."""
    out = []
    for x in pairwise:
        a, b = views[x["a"]], views[x["b"]]
        p_all, p_both = x["p_a_beats_b"], x["p_a_beats_b_if_both_active"]
        if p_both is None or x["a"] > x["b"]:  # each pair once
            continue
        if (p_both - 0.5) * (p_all - 0.5) < 0:
            talent, safe = (a, b) if p_both > 0.5 else (b, a)
            p_talent = p_both if p_both > 0.5 else 1 - p_both
            p_safe = 1 - p_all if p_both > 0.5 else p_all
            out.append(
                f"{talent.name} is the better play if both are active "
                f"({p_talent * 100:.0f}%), but {safe.name} wins {p_safe * 100:.0f}% once "
                f"{talent.name}'s {(1 - talent.play_probability) * 100:.0f}% chance of sitting "
                "is included."
            )
    return out


# Scoring terms grouped the way a manager reads them, with the expected stats (if
# active) that explain each group.
TERM_GROUPS = {
    "Receptions": (("rec",), ("receptions", "targets")),
    "Receiving yards": (("rec_yd",), ("receiving yards", "targets")),
    "Receiving TDs": (("rec_td",), ("receiving TDs",)),
    "Rushing yards": (("rush_yd",), ("rushing yards", "carries")),
    "Rushing TDs": (("rush_td",), ("rushing TDs", "carries")),
    "Passing yards": (("pass_yd",), ("passing yards",)),
    "Passing TDs": (("pass_td",), ("passing TDs",)),
    "Interceptions": (("pass_int",), ("interceptions",)),
    "Kicking": (("fgm_", "xpm", "fgmiss", "xpmiss"), ("field goals", "extra points")),
    "Sacks and takeaways": (("sack", "int", "fum_rec", "ff", "safe"),
                            ("sacks", "interceptions", "fumble recoveries")),
    "Defensive and return TDs": (("def_td", "def_st_td", "st_td", "fum_rec_td"),
                                 ("defensive TDs",)),
    "Points allowed": (("pts_allow",), ("points allowed",)),
    "Fumbles": (("fum_lost",), ()),
    "Two-point conversions": (("pass_2pt", "rush_2pt", "rec_2pt"), ()),
}  # fmt: skip
MIN_DRIVER_POINTS = 0.1


def _group(term: str) -> str:
    for name, (prefixes, _) in TERM_GROUPS.items():
        if any(term == p or (p.endswith("_") or p == "pts_allow") and term.startswith(p)
               for p in prefixes):  # fmt: skip
            return name
    return "Other"


def _if_active_terms(w: Week, rules: ScoringRules, view: PlayerView) -> dict[str, float]:
    """Expected points by stat group in the simulations where he plays (his forced-active
    batch when he rarely does), so the split matches proj_if_active."""
    if view.forced_active_batch and w.forced is not None:
        src = w.forced.filter(pl.col("forced_key") == view.key)
    else:
        src = w.sims.filter(pl.col("key") == view.key)
        src = src.filter(ifactive.active_flag(src))
    n = src["sim"].n_unique() if src.height else 0
    if n == 0:
        return {}
    t = breakdown(src.with_columns(pl.lit(view.key).alias("key")), rules, ids=["key", "sim"])
    out: dict[str, float] = {}
    for term, pts in t.group_by("term").agg(pl.col("points").sum()).iter_rows():
        g = _group(term)
        out[g] = out.get(g, 0.0) + float(pts) / n
    return out


def decision_drivers(w: Week, rules: ScoringRules, views: dict[str, PlayerView],
                     a: str, b: str, top: int = 3) -> list[dict]:  # fmt: skip
    """The biggest contributions to the gap between `a` (recommended) and `b`: the
    if-active gap split by stat group, each explained with both players' expected stats,
    plus availability (expected-points gap minus if-active gap) when it matters.
    Positive points favor `a`."""
    va, vb = views[a], views[b]
    ta, tb = _if_active_terms(w, rules, va), _if_active_terms(w, rules, vb)
    rows = []
    for g in sorted(set(ta) | set(tb)):
        d = ta.get(g, 0.0) - tb.get(g, 0.0)
        stats = TERM_GROUPS.get(g, ((), ()))[1]
        detail = "; ".join(
            f"{va.expected_stats_if_active.get(st, 0):.1f} vs "
            f"{vb.expected_stats_if_active.get(st, 0):.1f} {st}"
            for st in stats
            if st in va.expected_stats_if_active or st in vb.expected_stats_if_active
        )  # fmt: skip
        rows.append({"factor": g, "points": d, "detail": detail or None})
    if va.proj_if_active is not None and vb.proj_if_active is not None:
        avail = (va.expected_points - vb.expected_points) - (va.proj_if_active - vb.proj_if_active)
        rows.append({"factor": "Availability", "points": avail,
                     "detail": f"plays in {va.play_probability * 100:.0f}% vs "
                               f"{vb.play_probability * 100:.0f}% of simulations"})  # fmt: skip
    rows = [r for r in rows if abs(r["points"]) >= MIN_DRIVER_POINTS]
    rows.sort(key=lambda r: -abs(r["points"]))
    names = {a: va.name, b: vb.name}
    return [
        r | {"points": round(r["points"], 2), "favors": names[a] if r["points"] > 0 else names[b]}
        for r in rows[:top]
    ]


def compare(w: Week, rules: ScoringRules, actual: dict[str, float] | None = None) -> Comparison:
    """Start/sit between 2+ players: who to start, how likely each is to outscore the
    others (from simulations that share each game), and why. `actual`: points awarded so
    far (players whose games have started)."""
    views = player_views(w, rules)
    for k, v in views.items():
        if v.game_status in gamestate.STARTED and actual:
            v.actual_points = actual.get(k)
    keys = list(views)
    sims = w.sims.with_columns(ifactive.active_flag(w.sims).alias("active"))
    pts = score(sims, rules).select("key", "sim", "fpts", "active")
    mat = pts.pivot(on="key", index="sim", values="fpts").sort("sim")
    arr = mat.select(keys).to_numpy()
    act = pts.pivot(on="key", index="sim", values="active").sort("sim").select(keys).to_numpy()
    best = arr.argmax(axis=1)
    for i, k in enumerate(keys):
        views[k].p_best = float((best == i).mean())
    pairwise = []
    for a, b in permutations(range(len(keys)), 2):
        beats = (arr[:, a] > arr[:, b]) + 0.5 * (arr[:, a] == arr[:, b])
        both = act[:, a].astype(bool) & act[:, b].astype(bool)
        # Talent alone: only simulations where both play (too few -> not reported)
        p_both = float(beats[both].mean()) if both.sum() >= MIN_BOTH_ACTIVE else None
        pairwise.append({"a": keys[a], "b": keys[b], "p_a_beats_b": round(float(beats.mean()), 3),
                         "p_a_beats_b_if_both_active":
                         None if p_both is None else round(p_both, 3),
                         "median_gap":
                         round(float(np.median(arr[:, a] - arr[:, b])), 2)})  # fmt: skip
    ranked = sorted(views.values(), key=lambda v: v.expected_points, reverse=True)
    # A ruled-out player, or one whose game has started (his slot is locked), is never the
    # recommendation, whatever his mean says.
    playing = [v for v in ranked if v.play_probability >= OUT_THRESHOLD
               and v.game_status not in gamestate.STARTED] or ranked  # fmt: skip
    ranked = playing + [v for v in ranked if v not in playing]
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    p_top = next((x["p_a_beats_b"] for x in pairwise
                  if second and x["a"] == top.key and x["b"] == second.key), 1.0)  # fmt: skip
    notes = []
    for v in ranked:
        if v.game_status in gamestate.STARTED:
            pts = f" ({v.actual_points:.1f} points)" if v.actual_points is not None else ""
            what = "is over" if v.game_status == gamestate.FINAL else "has started"
            notes.append(f"{v.name}'s game {what}{pts}: his lineup slot is locked in Sleeper, "
                         "so he can't be started or benched now.")  # fmt: skip
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
    notes += disagreements(views, pairwise)
    close = bool(second and p_top < CLOSE_CALL and second.play_probability >= OUT_THRESHOLD
                 and second.game_status not in gamestate.STARTED)  # fmt: skip
    return Comparison(
        season=w.season, week=w.week, players=ranked, pairwise=pairwise,
        recommendation=top.key, close_call=close, notes=notes,
        decision_drivers=decision_drivers(w, rules, views, top.key, second.key)
        if second else [],
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
            filled[i] = {"slot": slots[i], "key": None, "name": None, "expected_points": 0.0,
                         "mean": 0.0}  # fmt: skip
            continue
        pick = max(cands, key=lambda v: v.expected_points)
        used.add(pick.key)
        filled[i] = {"slot": slots[i], "key": pick.key, "name": pick.name,
                     "proj_if_active": pick.proj_if_active,
                     "p10_if_active": pick.p10_if_active, "p90_if_active": pick.p90_if_active,
                     "play_probability": round(pick.play_probability, 3),
                     "expected_points": round(pick.expected_points, 2),
                     "injury_status": pick.injury_status,
                     "game_day_check": pick.game_day_check,
                     "mean": round(pick.expected_points, 2),
                     "median": round(pick.median, 2)}  # fmt: skip
    return [filled[i] for i in range(len(slots))]


def _slot_row(slot: str, v: PlayerView | None, *, locked: bool, actual: float | None,
              status: str | None, name: str | None = None) -> dict:  # fmt: skip
    """A lineup slot. A final game counts its awarded points; otherwise the projection."""
    final = status == gamestate.FINAL and actual is not None
    exp = actual if final else (v.expected_points if v else 0.0)
    return {
        "slot": slot, "key": v.key if v else None, "name": v.name if v else name,
        "proj_if_active": v.proj_if_active if v else None,
        "p10_if_active": v.p10_if_active if v else None,
        "p90_if_active": v.p90_if_active if v else None,
        "play_probability": round(v.play_probability, 3) if v else None,
        "expected_points": round(float(exp), 2),
        "injury_status": v.injury_status if v else None,
        "game_day_check": None if locked or not v else v.game_day_check,
        "locked": locked, "game_status": status, "actual_points": actual,
        "mean": round(float(exp), 2), "median": round(v.median, 2) if v else None,
    }  # fmt: skip


def locked_lineup(views: dict[str, PlayerView], slots: list[str],
                  set_starters: list[str | None], status: dict[str, str | None],
                  actual: dict[str, float], *, force: str | None = None,
                  exclude: frozenset[str] = frozenset(),
                  names: dict[str, str] | None = None) -> list[dict]:  # fmt: skip
    """The lineup as it can still be: slots whose set starter's game has kicked off are
    locked to him (as set in Sleeper); the open slots get the best lineup from players
    whose games haven't started. `force` puts a player in an open slot (start/sit odds);
    `exclude` keeps players out (the other candidates)."""
    started = {k for k, st in status.items() if st in gamestate.STARTED}
    locked = {i: k for i, k in enumerate(set_starters) if k and k in started}
    open_idx = [i for i in range(len(slots)) if i not in locked]
    pool = {k: v for k, v in views.items()
            if k not in started and k not in exclude and k not in locked.values()}  # fmt: skip
    if force is not None and force in pool:
        from dataclasses import replace

        pool[force] = replace(pool[force], expected_points=1e9)
    fill = best_lineup(pool, [slots[i] for i in open_idx]) if open_idx else []
    out: dict[int, dict] = {}
    for i, k in locked.items():
        out[i] = _slot_row(slots[i], views.get(k), locked=True, actual=actual.get(k),
                           status=status.get(k), name=(names or {}).get(k, k))  # fmt: skip
    for i, row in zip(open_idx, fill, strict=True):
        v = views.get(row["key"]) if row["key"] else None
        out[i] = _slot_row(row["slot"], v, locked=False, actual=None,
                           status=status.get(row["key"]) if row["key"] else None)  # fmt: skip
    return [out[i] for i in range(len(slots))]


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
