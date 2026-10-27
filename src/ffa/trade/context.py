"""Assemble a league for the trade engine: rest-of-season projections (latest daily run),
rosters and free agents, season-to-date ranks, records and the remaining schedule."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import polars as pl
from sqlalchemy import select

from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.scoring import score
from ffa.scoring.rules import ScoringRules
from ffa.service import gamestate, ros
from ffa.tracking import snapshots
from ffa.trade import core

NON_ROSTER = ("ir", "taxi")  # IR players are added back as Team.ir (tradeable, no roster spot)


@dataclass
class Context:
    league: core.League
    me: str  # team id
    records: dict[str, tuple[float, float]]  # team -> (wins, points for)
    schedule: dict[int, list[tuple[str, str]]] = field(default_factory=dict)  # regular season
    current: dict[str, tuple[float, float]] = field(default_factory=dict)  # this week so far
    current_week: int | None = None  # set when this week is under way
    playoff_teams: int | None = None
    as_of: str | None = None
    scoring: str = "default"
    notes: list[str] = field(default_factory=list)


def build(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
          now: datetime | None = None, fetch=None) -> Context:  # fmt: skip
    now = now or datetime.now(UTC)
    fetch = fetch or gamestate.fetch_matchups
    snap = ros.current(con, db, season, week)  # this week: the live weekly projection
    if snap.height == 0:
        raise LookupError("No rest-of-season projections yet: run ros_projections in Dagster")
    have = set(snap["league"].unique().to_list())
    lg_id = row.id if row.id in have else "default"
    snap = snap.filter(pl.col("league") == lg_id)
    notes = [] if lg_id == row.id else [
        "Rest-of-season numbers are in standard half-PPR: this league isn't in the latest "
        "run yet (it's added at the next daily run)."]  # fmt: skip

    # This week counts only while none of its games has started (trades can't move
    # locked players, and the lineup view handles the rest of this week).
    try:
        ko = snapshots.kickoffs(con, season, week)
    except Exception:  # schedule without kickoff times: treat the week as not started
        ko = pl.DataFrame(schema={"team": pl.Utf8, "kickoff": pl.Datetime("us", "UTC")})
    started = any(gamestate.status(k, now) in gamestate.STARTED for k in ko["kickoff"].to_list())
    weeks = sorted(w for w in snap["target_week"].unique().to_list() if w > week or not started)
    settings = row.settings or {}
    pstart = settings.get("playoff_week_start")
    playoff_weeks = {w for w in weeks if pstart and w >= int(pstart)}

    players: dict[str, core.Player] = {}
    for (key,), g in snap.group_by("key"):
        r0 = g.row(0, named=True)
        ifa = g.filter(pl.col("target_week").is_in(weeks))["proj_if_active"].drop_nulls() \
            if "proj_if_active" in g.columns else pl.Series([], dtype=pl.Float64)  # fmt: skip
        players[key] = core.Player(
            key, r0["name"] or key, r0["position"],
            {int(r["target_week"]): (float(r["expected_points"]), float(r["sd"] or 0.0))
             for r in g.iter_rows(named=True)},  # every week: this week's partial needs it
            nfl_team=r0.get("team"),
            per_game=round(float(ifa.mean()), 1) if ifa.len() else None,
        )  # fmt: skip
    rank_error = _ranks(con, rules, season, week, players)
    if rank_error:
        notes.append(f"Season-to-date ranks unavailable ({rank_error}); trades are judged "
                     "on projections only.")  # fmt: skip

    teams_rows = list(db.scalars(select(TeamRow).where(TeamRow.league_id == row.id)))
    me = next((t for t in teams_rows if t.is_me), None)
    if me is None:
        raise LookupError("Your team isn't identified in this league; sync with --me")
    limit = len(row.lineup_slots) + int(row.bench_slots or 0)
    rostered: set[str] = set()
    teams = {}
    to_key: dict[str, str] = {}
    for t in teams_rows:
        ents = list(db.scalars(select(RosterEntryRow).where(RosterEntryRow.team_id == t.id)))
        to_key |= {e.platform_player_id: e.player_key for e in ents if e.player_key}
        keys = [e.player_key for e in ents if e.player_key and e.slot_type not in NON_ROSTER]
        ir = {e.player_key for e in ents if e.player_key and e.slot_type == "ir"}
        rostered |= {e.player_key for e in ents if e.player_key}
        for e in ents:  # rostered players without a projection still occupy a spot
            if e.player_key and e.player_key not in players and e.slot_type != "taxi":
                players[e.player_key] = core.Player(e.player_key, e.name or e.player_key,
                                                    e.position or "?", {})  # fmt: skip
        teams[t.id] = core.Team(t.id, t.name, keys + sorted(ir), limit, ir,
                                int(row.ir_slots or 0))  # fmt: skip
    free = [k for k in players if k not in rostered]
    lg = core.League(list(row.lineup_slots), weeks, playoff_weeks, players, free, teams)
    ctx = Context(lg, me.id, {t.id: (t.wins + 0.5 * t.ties, float(t.points_for or 0.0))
                              for t in teams_rows},  # fmt: skip
                  playoff_teams=settings.get("playoff_teams"),
                  as_of=str(snap["_ingested_at"].max()) if "_ingested_at" in snap.columns else None,
                  scoring=lg_id, notes=notes)  # fmt: skip
    if pstart:
        status = {r["team"]: gamestate.status(r["kickoff"], now) for r in ko.iter_rows(named=True)}
        _schedule(ctx, row, teams_rows, week, int(pstart), started, fetch, to_key, status)
    return ctx


def _ranks(con, rules, season: int, week: int,
           players: dict[str, core.Player]) -> str | None:  # fmt: skip
    """Season-to-date points and positional rank (league scoring), and positional rank by
    projected points per game if active the rest of the way (byes and missed games left
    out, so it compares with what managers see). Returns why season-to-date ranks are
    unavailable, if they are."""
    error = None
    try:
        pg = con.sql(f"select * from player_games where season = {int(season)} "
                     f"and week < {int(week)}").pl()  # fmt: skip
        pts = score(pg, rules).group_by("key", "position").agg(pl.col("fpts").sum())
    except Exception as exc:  # said in the notes, never silently
        error = f"{type(exc).__name__}: {exc}"[:300]
        pts = pl.DataFrame(schema={"key": pl.Utf8, "position": pl.Utf8, "fpts": pl.Float64})
    pts = pts.with_columns(pl.col("fpts").rank("ordinal", descending=True).over("position")
                           .alias("rk"))  # fmt: skip
    for k, _pos, f, rk in pts.iter_rows():
        if k in players:
            players[k].season_points, players[k].season_rank = round(float(f), 1), int(rk)
    by_pos: dict[str, list[core.Player]] = {}
    for p in players.values():
        by_pos.setdefault(p.position, []).append(p)
    for ps in by_pos.values():
        ranked = sorted((p for p in ps if p.per_game is not None), key=lambda p: -p.per_game)
        for i, p in enumerate(ranked, 1):
            p.ros_rank = i
    return error


def _schedule(ctx: Context, row: LeagueRow, teams_rows: list[TeamRow], week: int, pstart: int,
              started: bool, fetch, to_key: dict[str, str],
              status: dict[str, str | None]) -> None:  # fmt: skip
    """Remaining regular-season matchups from Sleeper, and this week's points so far."""
    by_roster = {t.platform_team_id: t.id for t in teams_rows}
    lid = row.id.split(":", 1)[1]
    for w in range(week, pstart):
        try:
            ms = fetch(lid, w)
        except Exception as exc:
            ctx.notes.append(f"Couldn't read week {w}'s matchups from Sleeper: {exc}")
            ctx.schedule.clear()
            return
        pairs: dict[int, list[str]] = {}
        for m in ms:
            if m.get("matchup_id") is not None and str(m.get("roster_id")) in by_roster:
                pairs.setdefault(m["matchup_id"], []).append(by_roster[str(m["roster_id"])])
        ctx.schedule[w] = [(a, b) for a, b in (v for v in pairs.values() if len(v) == 2)]
        if w == week and started:
            _current(ctx, ms, by_roster, week, to_key, status)


def _current(ctx: Context, ms: list[dict], by_roster: dict[str, str], week: int,
             to_key: dict[str, str], status: dict[str, str | None]) -> None:  # fmt: skip
    """This week under way: each team's set starters, final games at awarded points, the
    rest at their projection (same rule as my_team)."""
    lg = ctx.league
    for m in ms:
        tid = by_roster.get(str(m.get("roster_id")))
        if tid is None:
            continue
        points = m.get("players_points") or {}
        mu = var = 0.0
        for sid in m.get("starters") or []:
            if sid in (None, "0"):
                continue
            p = lg.players.get(to_key.get(sid, ""))
            final = p is not None and status.get(p.nfl_team) == gamestate.FINAL
            if p is None or final:
                mu += float(points.get(sid) or 0.0)
                continue
            e, sd = p.weekly.get(week, (0.0, 0.0))
            mu += e
            var += sd**2
        ctx.current[tid] = (mu, var)
    ctx.current_week = week
