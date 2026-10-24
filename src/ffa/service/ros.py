"""Rest-of-season projections: every remaining week, from what is known today.

As of week `c` (its lines and injury report posted, its games not yet played):

- Week c runs exactly like the weekly projection (lines, injury report, Sleeper).
- Each later week reuses the same model with two changes. Its betting lines, where not
  yet posted, come from team ratings fitted on the lines so far (ffa.model.ratings), and
  each player's chance to play is how often players in his spot today played their
  team's k-th game from now (ffa.model.horizon): recovery, reserve-list stays and new
  injuries included.
- Player shares, efficiency and matchup ratings are today's, held flat.

Everything is cut at week c first (games, rosters, reports), so a backtest from a past
week sees only what was known then. Output: one row per (league scoring, player, week)
with expected points, points if he plays, chance to play and the spread of outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffa.model import horizon, ratings
from ffa.model.inputs import Models, build
from ffa.model.sim import simulate
from ffa.scoring import score
from ffa.scoring.rules import from_sleeper
from ffa.service import ifactive

N_FUTURE = 500  # simulations per future week
DATASET = "ros_projections"
DEFAULT_LAST_WEEK = 17  # fantasy championship week when a league's settings are unknown


@dataclass
class Frames:
    team_games: pl.DataFrame
    player_games: pl.DataFrame
    rosters: pl.DataFrame
    injuries: pl.DataFrame
    statuses: pl.DataFrame | None = None  # Sleeper statuses for week c (live only)


def as_of(f: Frames, season: int, week: int, *, mask_future_lines: bool) -> Frames:
    """The data as it stood before week `week`'s games: no results from it on, no
    rosters or reports after it, and (backtests) no lines posted after it."""
    tg = f.team_games.filter(pl.col("season") <= season)
    later = (pl.col("season") == season) & (pl.col("week") >= week)
    outcome_cols = [c for c in tg.columns if c not in (
        "game_id", "season", "week", "game_type", "team", "opponent", "is_home", "spread_for",
        "total_line", "implied", "opp_implied", "roof", "_ingested_at",
    )]  # fmt: skip
    tg = tg.with_columns(pl.when(later).then(None).otherwise(pl.col(c)).alias(c)
                         for c in outcome_cols)  # fmt: skip
    if mask_future_lines:
        future = (pl.col("season") == season) & (pl.col("week") > week)
        tg = tg.with_columns(
            pl.when(future).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("spread_for", "total_line", "implied", "opp_implied")
        )
    before = (pl.col("season") < season) | (pl.col("week") < week)
    upto = (pl.col("season") < season) | (pl.col("week") <= week)
    return Frames(
        tg, f.player_games.filter(before), f.rosters.filter(upto), f.injuries.filter(upto),
        f.statuses,
    )  # fmt: skip


def standing(f: Frames, season: int, week: int) -> pl.DataFrame:
    """key -> (cat, role) as of week `week`: latest roster status, this week's report."""
    st = horizon._standing(
        f.rosters.filter(pl.col("season") == season),
        f.injuries.filter(pl.col("season") == season),
        f.player_games.filter(pl.col("season") == season),
    ).filter(pl.col("week") <= week)
    return st.sort("week").group_by("key").last().select("key", "team", "cat", "role")


def _with_current(st: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Refine today's category with this week's inputs: the combined designation
    (Sleeper included) and, before any designation, this week's chance to play (a player
    who missed practice after being out is not healthy for the weeks after)."""
    cur = players.select(
        "key", pl.col("injury_status").alias("_st"), pl.col("p_play").alias("_p"),
        pl.col("roster_status").alias("_rs") if "roster_status" in players.columns
        else pl.lit(None, pl.Utf8).alias("_rs"),
    )  # fmt: skip
    cat = (
        pl.when(pl.col("_rs") == "RES").then(pl.lit("reserve"))
        .when(pl.col("_st").is_in(["Out", "Doubtful", "Questionable"]))
        .then(pl.col("_st").str.to_lowercase())
        .when(pl.col("_p") < 0.15).then(pl.lit("out"))
        .when(pl.col("_p") < 0.5).then(pl.lit("doubtful"))
        .when(pl.col("_p") < 0.95).then(pl.lit("questionable"))
        .otherwise(pl.col("cat"))
    )  # fmt: skip
    return (
        st.join(cur, on="key", how="left")
        .with_columns(
            pl.when(pl.col("_p").is_not_null()).then(cat).otherwise(pl.col("cat")).alias("cat")
        )
        .drop("_st", "_p", "_rs")
    )


def team_game_index(team_games: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """(team, week) -> k: the team's k-th game after week `week` (byes skipped)."""
    later = (pl.col("season") == season) & (pl.col("week") > week)
    g = team_games.filter(later & (pl.col("game_type") == "REG")).select("team", "week").unique()
    return g.with_columns(pl.col("week").rank("ordinal").over("team").cast(pl.Int32).alias("k"))


def _summaries(lines: pl.DataFrame, leagues: dict[str, dict], week: int) -> pl.DataFrame:
    sims = lines.with_columns(ifactive.active_flag(lines).alias("active"))
    out = []
    for league, scoring in leagues.items():
        pts = score(sims, from_sleeper(scoring)).select("key", "sim", "fpts", "active")
        s = pts.group_by("key").agg(
            pl.col("fpts").mean().alias("expected_points"),
            pl.col("fpts").filter(pl.col("active")).mean().alias("proj_if_active"),
            pl.col("fpts").std().alias("sd"),
            pl.col("fpts").filter(pl.col("active")).quantile(0.1).alias("p10_if_active"),
            pl.col("fpts").filter(pl.col("active")).quantile(0.9).alias("p90_if_active"),
        )
        out.append(s.with_columns(pl.lit(league).alias("league"), pl.lit(week).cast(pl.Int32)
                                  .alias("target_week")))  # fmt: skip
    return pl.concat(out)


def project(frames: Frames, models: Models, table: dict[str, float], season: int, week: int,
            last_week: int, leagues: dict[str, dict], *, n: int = N_FUTURE,
            mask_future_lines: bool = False, seed: int = 0) -> pl.DataFrame:  # fmt: skip
    """Rest-of-season rows: league, key, name, position, team, target_week, opponent,
    play_probability, expected_points, proj_if_active, sd, p10/p90_if_active,
    lines_from_ratings."""
    f = as_of(frames, season, week, mask_future_lines=mask_future_lines)
    r = ratings.fit(f.team_games, season, week)
    tg = ratings.fill_lines(f.team_games, r, season, week)
    st = standing(f, season, week)
    kidx = team_game_index(tg, season, week)
    out = []
    for w in range(week, last_week + 1):
        current = w == week
        inp = build(
            season=season, week=w, models=models, team_games=tg, player_games=f.player_games,
            rosters=f.rosters, injuries=f.injuries,
            statuses=f.statuses if current else None, keep_reserve=not current,
            estimate_week=week,  # today's estimates, held flat (not staler each week)
        )  # fmt: skip
        if inp.teams.height == 0:
            continue
        missing = inp.teams.filter(pl.col("spread_for").is_null() | pl.col("total_line").is_null()
                                   | pl.col("mu_plays").is_null())  # fmt: skip
        if missing.height:  # never simulate a game without lines: volume would collapse
            raise ValueError(f"Week {w}: no lines for {sorted(missing['team'].to_list())}")
        players = inp.players
        if current:  # where each player stands today, from this week's full picture
            st = _with_current(st, players)
        if not current:  # chance to play from where each player stands today
            k = kidx.filter(pl.col("week") == w).select("team", "k")
            players = players.join(st.select("key", "cat", "role"), on="key", how="left").join(
                k, on="team", how="left").with_columns(
                pl.struct("position", "cat", "role", "k").map_elements(
                    lambda x: horizon.p_play(table, x["position"], x["cat"] or "healthy",
                                             x["role"] or "depth", x["k"] or 1),
                    return_dtype=pl.Float64,
                ).alias("p_play")
            ).drop("cat", "role", "k")  # fmt: skip
            inp.players = players
        lines = simulate(inp, models.sim, n=n, seed=seed + season * 100 + w)
        s = _summaries(lines, leagues, w)
        meta = players.select("key", "name", "position", "team", "opponent",
                              pl.col("p_play").alias("play_probability"))  # fmt: skip
        dst = lines.filter(pl.col("key").str.starts_with("DST_")).select(
            "key", "name", "team", "position").unique("key")  # fmt: skip
        filled = tg.filter((pl.col("season") == season) & (pl.col("week") == w)).select(
            "team", pl.col("lines_from_ratings").fill_null(False)).unique("team")  # fmt: skip
        meta = pl.concat([meta, dst.with_columns(pl.lit(None, pl.Utf8).alias("opponent"),
                                                 pl.lit(1.0).alias("play_probability"))
                          .select(meta.columns)], how="vertical_relaxed")  # fmt: skip
        out.append(
            s.join(meta.unique("key"), on="key", how="left").join(filled, on="team", how="left")
        )
    if not out:
        return pl.DataFrame()
    df = pl.concat(out, how="diagonal_relaxed")
    return df.with_columns(pl.lit(season).cast(pl.Int32).alias("season"),
                           pl.lit(week).cast(pl.Int32).alias("as_of_week"))  # fmt: skip


def trim_to_leagues(df: pl.DataFrame, last: dict[str, int],
                    settings: dict[str, dict]) -> pl.DataFrame:  # fmt: skip
    """Each league's rows through its own championship week, with `playoff` marking its
    fantasy playoff weeks."""
    if df.height == 0:
        return df
    end = pl.col("league").replace_strict(last, default=DEFAULT_LAST_WEEK, return_dtype=pl.Int32)
    start = {lg: int(s.get("playoff_week_start") or 99) for lg, s in settings.items()}
    pstart = pl.col("league").replace_strict(start, default=99, return_dtype=pl.Int32)
    return df.filter(pl.col("target_week") <= end).with_columns(
        (pl.col("target_week") >= pstart).alias("playoff")
    )


def totals(ros: pl.DataFrame) -> pl.DataFrame:
    """Per (league, player): rest-of-season expected points and weeks projected."""
    return ros.group_by("league", "key", "name", "position").agg(
        pl.col("expected_points").sum().alias("ros_points"),
        pl.col("target_week").n_unique().alias("weeks"),
        (pl.col("sd") ** 2).sum().sqrt().alias("ros_sd"),
    )  # fmt: skip


def latest(con, season: int, week: int) -> pl.DataFrame:
    """The newest rest-of-season snapshot as of `week` (or the week before, e.g. Monday
    morning before the new week's run). Empty if none."""
    from ffa.warehouse import duck

    for w in (week, week - 1):
        if w < 1:
            continue
        try:
            t = duck.week(con, DATASET, season, w)
        except duck.MissingPartition:
            continue
        df = con.sql(f"select * from {t}").pl()
        if df.height and "_ingested_at" in df.columns:
            df = df.filter(pl.col("_ingested_at") == pl.col("_ingested_at").max())
        if df.height:
            return df.filter(pl.col("target_week") >= week)
    return pl.DataFrame()


def summary(snap: pl.DataFrame, keys: list[str], league: str | None) -> dict[str, dict]:
    """Per player: rest-of-season total, playoff-week total and each week, in `league`'s
    scoring when the snapshot has it (else standard half-PPR, said so)."""
    if snap.height == 0:
        return {}
    have = set(snap["league"].unique().to_list())
    lg = league if league in have else "default"
    s = snap.filter((pl.col("league") == lg) & pl.col("key").is_in(keys)).sort("target_week")
    as_of = s["_ingested_at"].max() if "_ingested_at" in s.columns and s.height else None
    out = {}
    for (key,), g in s.group_by("key"):
        playoff = g.filter(pl.col("playoff")) if "playoff" in g.columns else g.head(0)
        first = g.row(0, named=True)
        out[key] = {
            "key": key, "name": first.get("name"), "position": first.get("position"),
            "team": first.get("team"),
            "total": round(float(g["expected_points"].sum()), 1),
            # Points per game if he plays every remaining game (availability left out).
            "per_game_if_active": round(float(g["proj_if_active"].mean()), 1)
            if "proj_if_active" in g.columns and g["proj_if_active"].is_not_null().any()
            else None,
            "expected_games": round(float(g["play_probability"].fill_null(0).sum()), 1),
            "playoff_weeks_total": round(float(playoff["expected_points"].sum()), 1)
            if playoff.height else None,
            "weeks": g.height,
            "scoring": lg,
            "as_of": as_of.isoformat() if as_of is not None else None,
            "by_week": [
                {"week": int(r["target_week"]), "opponent": r["opponent"],
                 "expected_points": round(float(r["expected_points"]), 2),
                 "proj_if_active": None if r.get("proj_if_active") is None
                 else round(float(r["proj_if_active"]), 2),
                 "play_probability": round(float(r["play_probability"] or 0), 3),
                 "playoff": bool(r.get("playoff") or False)}
                for r in g.iter_rows(named=True)
            ],
        }  # fmt: skip
    return out
