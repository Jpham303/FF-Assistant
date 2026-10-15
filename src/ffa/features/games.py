"""Team-game and player-game logs from play-by-play, weekly stats, snaps and injuries."""

from __future__ import annotations

import polars as pl

from ffa.ingest.ids import TEAM_ALIASES
from ffa.scoring.defense import normalize_teams
from ffa.scoring.stats import player_stat_lines

SKILL = ["QB", "RB", "WR", "TE", "K"]
POSITION_MAP = {"FB": "RB", "HB": "RB"}

# Play-by-play columns the logs read. Load through `load_pbp` (all 372 columns for many
# seasons exhausts memory).
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
    "qb_dropback", "qb_scramble", "pass_attempt", "rush_attempt", "sack", "interception",
    "fumble_lost", "two_point_attempt", "receiver_player_id", "rusher_player_id",
    "passer_player_id", "air_yards", "yardline_100", "xpass", "epa", "success",
    "pass_touchdown", "rush_touchdown", "field_goal_attempt", "field_goal_result",
    "kick_distance", "extra_point_attempt", "extra_point_result", "kicker_player_id",
]  # fmt: skip


def load_pbp(con, season: int) -> pl.DataFrame:
    return con.sql(f"select {', '.join(PBP_COLUMNS)} from pbp where season = {int(season)}").pl()


def _f(c: str) -> pl.Expr:
    return pl.col(c).fill_null(0).cast(pl.Float64)


def _offense_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Pass and run plays that can produce fantasy stats (no kneels, spikes, conversions)."""
    return normalize_teams(pbp).filter(
        pl.col("play_type").is_in(["pass", "run"]) & (_f("two_point_attempt") == 0)
    )


# ---------------------------------------------------------------- team games


def _vegas(schedules: pl.DataFrame) -> pl.DataFrame:
    """Two rows per game (one per team) with lines from that team's point of view.

    nflverse spread_line is positive when the home team is favored, so a team's implied
    total is (total + spread_for) / 2.
    """
    s = normalize_teams(schedules).select(
        "game_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        "game_type", "home_team", "away_team", "home_score", "away_score",
        "spread_line", "total_line", "roof",
    )  # fmt: skip
    sides = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        sides.append(
            s.select(
                "game_id",
                "season",
                "week",
                "game_type",
                pl.col(f"{side}_team").alias("team"),
                pl.col(f"{other}_team").alias("opponent"),
                pl.lit(side == "home").alias("is_home"),
                (sign * pl.col("spread_line")).alias("spread_for"),
                pl.col("total_line"),
                pl.col(f"{side}_score").cast(pl.Float64).alias("points"),
                pl.col(f"{other}_score").cast(pl.Float64).alias("opp_points"),
                pl.col("roof"),
            )  # fmt: skip
        )
    return pl.concat(sides).with_columns(
        ((pl.col("total_line") + pl.col("spread_for")) / 2).alias("implied"),
        ((pl.col("total_line") - pl.col("spread_for")) / 2).alias("opp_implied"),
    )


def _team_offense(pbp: pl.DataFrame) -> pl.DataFrame:
    plays = _offense_plays(pbp)
    designed = (pl.col("play_type") == "run") & (_f("qb_scramble") == 0)
    dropback = _f("qb_dropback") == 1
    off = plays.group_by("game_id", pl.col("posteam").alias("team")).agg(
        pl.len().cast(pl.Float64).alias("plays"),
        _f("qb_dropback").sum().alias("dropbacks"),
        designed.cast(pl.Float64).sum().alias("designed_runs"),
        _f("qb_scramble").sum().alias("scrambles"),
        _f("sack").sum().alias("sacks_taken"),
        (_f("pass_attempt") - _f("sack")).clip(lower_bound=0).sum().alias("pass_att"),
        pl.col("receiver_player_id").is_not_null().cast(pl.Float64).sum().alias("targets"),
        pl.when(pl.col("receiver_player_id").is_not_null()).then(_f("air_yards")).otherwise(0)
        .sum().alias("air_yards"),
        _f("interception").sum().alias("ints_thrown"),
        _f("fumble_lost").sum().alias("fumbles_lost"),
        _f("pass_touchdown").sum().alias("pass_td"),
        _f("rush_touchdown").sum().alias("rush_td"),
        (pl.col("yardline_100") <= 20).cast(pl.Float64).sum().alias("rz_plays"),
        pl.col("xpass").mean().alias("xpass"),
        pl.col("epa").filter(dropback).mean().alias("epa_pass"),
        pl.col("epa").filter(designed).mean().alias("epa_rush"),
        pl.col("success").filter(dropback).mean().alias("success_pass"),
        pl.col("success").filter(designed).mean().alias("success_rush"),
    )  # fmt: skip
    return off.with_columns(
        (pl.col("dropbacks") / pl.col("plays")).alias("pass_rate"),
        (pl.col("dropbacks") / pl.col("plays") - pl.col("xpass")).alias("pass_rate_oe"),
    )


def _team_kicking(pbp: pl.DataFrame) -> pl.DataFrame:
    k = normalize_teams(pbp).filter(
        (_f("field_goal_attempt") == 1) | (_f("extra_point_attempt") == 1)
    )
    fg = _f("field_goal_attempt") == 1
    return k.group_by("game_id", pl.col("posteam").alias("team")).agg(
        fg.cast(pl.Float64).sum().alias("fg_att"),
        (pl.col("field_goal_result") == "made").cast(pl.Float64).sum().alias("fg_made"),
        pl.col("kick_distance").filter(fg).mean().alias("fg_avg_distance"),
        _f("extra_point_attempt").sum().alias("xp_att"),
        (pl.col("extra_point_result") == "good").cast(pl.Float64).sum().alias("xp_made"),
    )


def team_game_log(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team per game: lines, own offense, and the opponent's offense (which is
    what this team's defense allowed). Future games have lines and no outcomes."""
    base = _vegas(schedules)
    off = _team_offense(pbp).join(_team_kicking(pbp), on=["game_id", "team"], how="left")
    stat_cols = [c for c in off.columns if c not in ("game_id", "team")]
    opp = off.rename({c: f"opp_{c}" for c in stat_cols} | {"team": "opponent"})
    return (
        base.join(off, on=["game_id", "team"], how="left")
        .join(opp, on=["game_id", "opponent"], how="left")
        .sort("season", "week", "team")
    )


# ---------------------------------------------------------------- player games


def _player_opportunity(pbp: pl.DataFrame) -> pl.DataFrame:
    plays = _offense_plays(pbp)
    rz = pl.col("yardline_100") <= 20
    targeted = plays.filter(pl.col("receiver_player_id").is_not_null()).group_by(
        "game_id", pl.col("posteam").alias("team"), pl.col("receiver_player_id").alias("key")
    ).agg(
        pl.len().cast(pl.Float64).alias("targets"),
        rz.cast(pl.Float64).sum().alias("rz_targets"),
        (_f("air_yards") >= pl.col("yardline_100")).cast(pl.Float64).sum().alias("ez_targets"),
        _f("air_yards").sum().alias("air_yards"),
    )  # fmt: skip
    rushes = plays.filter(pl.col("rusher_player_id").is_not_null()).group_by(
        "game_id", pl.col("posteam").alias("team"), pl.col("rusher_player_id").alias("key")
    ).agg(
        (_f("qb_scramble") == 0).cast(pl.Float64).sum().alias("designed_carries"),
        _f("qb_scramble").sum().alias("scrambles"),
        (rz & (_f("qb_scramble") == 0)).cast(pl.Float64).sum().alias("rz_carries"),
        ((pl.col("yardline_100") <= 5) & (_f("qb_scramble") == 0)).cast(pl.Float64).sum()
        .alias("gl_carries"),
    )  # fmt: skip
    dropbacks = (
        plays.filter(_f("qb_dropback") == 1)
        .with_columns(pl.coalesce("passer_player_id", "rusher_player_id").alias("key"))
        .group_by("game_id", pl.col("posteam").alias("team"), "key")
        .agg(pl.len().cast(pl.Float64).alias("dropbacks"))
    )
    keys = ["game_id", "team", "key"]
    out = targeted.join(rushes, on=keys, how="full", coalesce=True)
    out = out.join(dropbacks, on=keys, how="full", coalesce=True)
    return out.filter(pl.col("key").is_not_null()).with_columns(
        pl.col(c).fill_null(0.0) for c in out.columns if c not in keys
    )


def _snaps(snap_counts: pl.DataFrame, player_ids: pl.DataFrame) -> pl.DataFrame:
    """Snap counts keyed by gsis_id. nflverse keys snaps by Pro Football Reference ID; the
    crosswalk maps 99.9% of them in every season (weekly rosters only ~75% before 2020)."""
    pfr = (
        player_ids.filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .select(pl.col("pfr_id").alias("pfr_player_id"), pl.col("gsis_id").alias("key"))
        .unique("pfr_player_id")
    )
    return (
        snap_counts.join(pfr, on="pfr_player_id", how="inner")
        .group_by("game_id", "key")
        .agg(
            _f("offense_snaps").sum().alias("snaps"),
            _f("offense_pct").max().alias("snap_pct"),
        )
    )


def _injury_status(injuries: pl.DataFrame) -> pl.DataFrame:
    """Final pre-game report status per player-week (None = not on the report)."""
    return (
        injuries.filter(pl.col("gsis_id").is_not_null())
        .select(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("gsis_id").alias("key"),
            pl.col("report_status").alias("injury_status"),
            pl.col("report_primary_injury").alias("injury"),
        )
        .unique(["season", "week", "key"], keep="last")
    )


def player_game_log(
    pbp: pl.DataFrame,
    player_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    player_ids: pl.DataFrame,
    injuries: pl.DataFrame,
    teams: pl.DataFrame,
    rosters: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per player per team-game: opportunities, canonical stat outcomes, snaps,
    injury status going in, and team totals for share denominators.

    `played` marks games the player recorded a stat in. With `rosters`, healthy
    active-roster players who did not play get a row too, with zero opportunities: a
    backup's expected share must count the weeks he sat. Weeks missed while on the injury
    report are left out (availability is modeled separately)."""
    stats = (
        player_stat_lines(player_stats)
        .with_columns(
            pl.col("team", "opponent").replace(TEAM_ALIASES),
            pl.col("position").replace(POSITION_MAP),
        )
        .filter(pl.col("position").is_in(SKILL))
    )
    games = teams.select("game_id", "season", "week", "team").unique()
    stats = stats.join(games, on=["season", "week", "team"], how="inner")

    opp = _player_opportunity(pbp)
    team_cols = teams.select(
        "game_id", "team", "opponent", "is_home", "implied", "opp_implied", "spread_for",
        "total_line", pl.col("plays").alias("team_plays"),
        pl.col("dropbacks").alias("team_dropbacks"),
        pl.col("designed_runs").alias("team_designed_runs"),
        pl.col("targets").alias("team_targets"), pl.col("air_yards").alias("team_air_yards"),
        pl.col("rz_plays").alias("team_rz_plays"),
        pl.col("pass_td").alias("team_pass_td"), pl.col("rush_td").alias("team_rush_td"),
    )  # fmt: skip
    out = (
        stats.drop("opponent")
        .join(opp.drop("team"), on=["game_id", "key"], how="left")
        .join(_snaps(snap_counts, player_ids), on=["game_id", "key"], how="left")
        .join(_injury_status(injuries), on=["season", "week", "key"], how="left")
        .join(team_cols, on=["game_id", "team"], how="left")
    )
    fill = ["targets", "rz_targets", "ez_targets", "air_yards", "designed_carries",
            "scrambles", "rz_carries", "gl_carries", "dropbacks", "snaps"]  # fmt: skip
    out = out.with_columns(pl.col(fill).fill_null(0.0), pl.lit(True).alias("played"))
    if rosters is not None:
        out = pl.concat([out, _unplayed(rosters, injuries, team_cols, out)],
                        how="diagonal_relaxed")  # fmt: skip
    return out.sort("season", "week", "key")


def _unplayed(
    rosters: pl.DataFrame, injuries: pl.DataFrame, team_cols: pl.DataFrame, played: pl.DataFrame
) -> pl.DataFrame:
    """Zero-opportunity rows for healthy active-roster players who didn't play."""
    ros = (
        rosters.filter(
            (pl.col("status") == "ACT")
            & pl.col("gsis_id").is_not_null()
            & pl.col("position").replace(POSITION_MAP).is_in(SKILL)
        )
        .select(
            pl.col("gsis_id").alias("key"),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("team").replace(TEAM_ALIASES),
            pl.col("position").replace(POSITION_MAP),
        )
        .unique(["key", "season", "week"])
    )
    games = team_cols.join(
        played.select("game_id", "season", "week").unique(), on="game_id", how="inner"
    )
    injured = _injury_status(injuries).filter(
        pl.col("injury_status").is_in(["Out", "Doubtful", "Questionable"])
    )
    zero = (
        ros.join(games, on=["season", "week", "team"], how="inner")
        .join(played.select("key", "season", "week"), on=["key", "season", "week"], how="anti")
        .join(injured.select("key", "season", "week"), on=["key", "season", "week"], how="anti")
    )
    counts = [c for c in played.columns if played.schema[c].is_numeric()
              and c not in team_cols.columns and c not in ("season", "week")]  # fmt: skip
    return zero.with_columns(*[pl.lit(0.0).alias(c) for c in counts], pl.lit(False).alias("played"))
