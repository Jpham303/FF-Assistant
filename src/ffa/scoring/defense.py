"""Team defense (D/ST) stat lines from play-by-play.

Built from plays rather than nflverse team totals because the platforms score things team
totals blur together, all confirmed against Sleeper's awarded points:

- special-teams fumble recoveries and forced fumbles score differently from defensive ones
- strip-sacks are sacks, even when no player is credited with the forced fumble
- blocked extra points are blocked kicks
- points allowed leaves out points the defense was not on the field for: return
  touchdowns by the opponent's defense or special teams, and opponent safeties
"""

from __future__ import annotations

import polars as pl

from ffa.ingest.ids import TEAM_ALIASES, dst_key

SCRIMMAGE = ["pass", "run", "qb_kneel", "qb_spike"]
SPECIAL = ["punt", "kickoff", "field_goal", "extra_point"]

DST_STAT_NAMES = [
    "sack", "sack_yd", "int", "ff", "fum_rec", "def_st_ff", "def_st_fum_rec", "safe",
    "def_td", "def_st_td", "blk_kick", "qb_hit", "tkl_loss", "def_pass_def",
    "def_kr_yd", "def_pr_yd", "pts_allow", "yds_allow",
]  # fmt: skip


TEAM_COLS = [
    "home_team", "away_team", "posteam", "defteam", "td_team",
    "fumble_recovery_1_team", "forced_fumble_player_1_team",
]  # fmt: skip


# The ~30 of play-by-play's 372 columns that D/ST scoring reads. Always load through
# `load_pbp`: a full multi-season pull of every column needs several GB of memory.
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "home_team", "away_team", "posteam", "defteam",
    "play_type", "sack", "interception", "fumble_forced", "fumble_lost",
    "forced_fumble_player_1_team", "fumble_recovery_1_team", "safety", "touchdown",
    "return_touchdown", "td_team", "punt_blocked", "field_goal_result", "extra_point_result",
    "two_point_conv_result", "defensive_two_point_conv", "two_point_attempt", "qb_hit",
    "tackled_for_loss", "pass_defense_1_player_id", "return_yards", "yards_gained",
    "posteam_score", "posteam_score_post",
]  # fmt: skip


def load_pbp(con, season: int | None = None) -> pl.DataFrame:
    """Play-by-play columns needed for D/ST scoring, from the DuckDB warehouse."""
    where = f"where season = {int(season)}" if season is not None else ""
    return con.sql(f"select {', '.join(PBP_COLUMNS)} from pbp {where}").pl()


def normalize_teams(df: pl.DataFrame) -> pl.DataFrame:
    """Relocated teams (STL/LA, SD/LAC, OAK/LV) are coded differently across nflverse
    datasets in older seasons; map every team column to current codes."""
    return df.with_columns(pl.col(c).replace(TEAM_ALIASES) for c in TEAM_COLS if c in df.columns)


def _f(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0).cast(pl.Float64)


def _is(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0) == 1


def _event(
    p: pl.DataFrame, cond: pl.Expr, scorer: pl.Expr, points: float, by_offense: pl.Expr, kind: str
) -> pl.DataFrame:
    return p.filter(cond).select(
        "game_id", "play_id", scorer.alias("scorer"), pl.lit(float(points)).alias("points"),
        by_offense.alias("by_offense"), pl.lit(kind).alias("kind"),
    )  # fmt: skip


def scoring_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Every scoring event: who scored, how many points, and whether the points count
    against the opposing defense (`against_defense`).

    Points come from the event itself (TD 6, FG 3, XP 1, two-point 2, safety 2) rather
    than from running score columns, which occasionally step backwards mid-game.
    """
    p = normalize_teams(pbp).sort("game_id", "play_id")
    scrim = pl.col("play_type").is_in(SCRIMMAGE)
    pos, dfn = pl.col("posteam"), pl.col("defteam")
    # Usually the defense scores a safety, but on a kick return the kicking team (posteam)
    # can; credit whichever side's score went up.
    safety_team = (
        pl.when((pl.col("posteam_score_post") - pl.col("posteam_score")) == 2)
        .then(pos)
        .otherwise(dfn)
    )
    events = pl.concat(
        [
            _event(p, _is("touchdown") & pl.col("td_team").is_not_null(), pl.col("td_team"),
                   6, scrim & (pl.col("td_team") == pos), "td"),
            _event(p, pl.col("field_goal_result") == "made", pos, 3, pl.lit(True), "fg"),
            _event(p, pl.col("extra_point_result") == "good", pos, 1, pl.lit(True), "conversion"),
            _event(p, pl.col("two_point_conv_result") == "success", pos, 2, pl.lit(True),
                   "conversion"),
            _event(p, _is("defensive_two_point_conv"), dfn, 2, pl.lit(False), "def_conversion"),
            _event(p, _is("safety"), safety_team, 2, pl.lit(False), "safety"),
        ]
    ).sort("game_id", "play_id")  # fmt: skip
    # The conversion after a return touchdown belongs to that return, not to the offense.
    prev = lambda c: pl.col(c).shift(1).over("game_id")  # noqa: E731
    after_return_td = (
        (pl.col("kind") == "conversion")
        & (prev("kind") == "td")
        & ~prev("by_offense").fill_null(True)
        & (prev("scorer") == pl.col("scorer"))
    )
    return events.with_columns(
        (pl.col("by_offense") & ~after_return_td).alias("against_defense")
    ).select("game_id", "play_id", "scorer", "points", "kind", "against_defense")


def _count(plays: pl.DataFrame, cond: pl.Expr, team: str, name: str, value: pl.Expr | None = None):
    agg = value.sum() if value is not None else pl.len()  # a literal would sum to 1 per group
    return (
        plays.filter(cond & pl.col(team).is_not_null())
        .group_by("game_id", pl.col(team).alias("team"))
        .agg(agg.cast(pl.Float64).alias(name))
    )


def dst_stat_lines(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game with canonical D/ST stats. Key = DST_<TEAM>."""
    schedules, pbp = normalize_teams(schedules), normalize_teams(pbp)
    games = schedules.filter(pl.col("home_score").is_not_null()).select(
        "game_id",
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    sides = pl.concat(
        [
            games.select("game_id", "season", "week", pl.col("home_team").alias("team"),
                         pl.col("away_team").alias("opponent"),
                         pl.col("away_score").cast(pl.Float64).alias("opp_score")),
            games.select("game_id", "season", "week", pl.col("away_team").alias("team"),
                         pl.col("home_team").alias("opponent"),
                         pl.col("home_score").cast(pl.Float64).alias("opp_score")),
        ]
    )  # fmt: skip

    p = pbp.filter(pl.col("game_id").is_in(games["game_id"].implode()))
    scrim = pl.col("play_type").is_in(SCRIMMAGE)
    # Any kicking play counts as special teams, including returns of blocked kicks.
    kick = pl.col("play_type").is_in(SPECIAL)
    pieces = [
        _count(p, _is("sack"), "defteam", "sack"),
        _count(p, _is("sack"), "defteam", "sack_yd", -_f("yards_gained")),
        _count(p, _is("interception"), "defteam", "int"),
        _count(p, scrim & _is("fumble_forced"), "forced_fumble_player_1_team", "ff"),
        _count(p, scrim & _is("fumble_lost"), "fumble_recovery_1_team", "fum_rec"),
        _count(p, kick & _is("fumble_forced"), "forced_fumble_player_1_team", "def_st_ff"),
        _count(p, kick & _is("fumble_lost"), "fumble_recovery_1_team", "def_st_fum_rec"),
        _count(
            p.with_columns(
                pl.when((pl.col("posteam_score_post") - pl.col("posteam_score")) == 2)
                .then(pl.col("posteam"))
                .otherwise(pl.col("defteam"))
                .alias("safety_team")
            ),
            _is("safety"),
            "safety_team",
            "safe",
        ),
        _count(p, scrim & _is("return_touchdown"), "td_team", "def_td"),
        _count(p, pl.col("play_type").is_in(SPECIAL) & _is("touchdown"), "td_team", "def_st_td"),
        _count(
            p,
            _is("punt_blocked")
            | (pl.col("field_goal_result") == "blocked")
            | (pl.col("extra_point_result") == "blocked"),
            "defteam",
            "blk_kick",
        ),
        _count(p, _is("qb_hit"), "defteam", "qb_hit"),
        _count(p, _is("tackled_for_loss"), "defteam", "tkl_loss"),
        _count(p, pl.col("pass_defense_1_player_id").is_not_null(), "defteam", "def_pass_def"),
        _count(p, pl.col("play_type") == "kickoff", "posteam", "def_kr_yd", _f("return_yards")),
        _count(p, pl.col("play_type") == "punt", "defteam", "def_pr_yd", _f("return_yards")),
        _count(p, scrim & ~_is("two_point_attempt"), "defteam", "yds_allow", _f("yards_gained")),
    ]
    not_against = (
        scoring_plays(p)
        .filter(~pl.col("against_defense"))
        .group_by("game_id", pl.col("scorer").alias("opponent"))
        .agg(pl.col("points").sum().cast(pl.Float64).alias("excluded"))
    )

    out = sides
    for piece in pieces:
        out = out.join(piece, on=["game_id", "team"], how="left")
    return (
        out.join(not_against, on=["game_id", "opponent"], how="left")
        .with_columns(
            (pl.col("opp_score") - pl.col("excluded").fill_null(0)).clip(lower_bound=0)
            .alias("pts_allow"),
            pl.col("team").map_elements(dst_key, return_dtype=pl.Utf8).alias("key"),
            pl.lit("DEF").alias("position"),
        )
        .with_columns(pl.col(DST_STAT_NAMES).fill_null(0.0))
        .select("key", "season", "week", "team", "opponent", "position", *DST_STAT_NAMES)
    )  # fmt: skip
