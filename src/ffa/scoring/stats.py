"""Canonical stat lines, built from nflverse.

Every scoring rule is written against these names, whatever the platform. They follow
Sleeper's vocabulary because it is the most granular public one; Yahoo stat IDs are mapped
onto them. One row per player-week (key = gsis_id) or defense-week (key = DST_<TEAM>).
"""

from __future__ import annotations

import polars as pl

from ffa.ingest.ids import dst_key

ID_COLS = ["key", "season", "week", "team", "opponent", "position"]


def _c(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0).cast(pl.Float64)


def _list_over(col: str, threshold: int) -> pl.Expr:
    """Sum over a ';'-separated distance list of max(d - threshold, 0)."""
    return (
        pl.col(col)
        .fill_null("")
        .str.split(";")
        .list.eval(pl.element().filter(pl.element() != "").cast(pl.Int64) - threshold)
        .list.eval(pl.element().clip(lower_bound=0))
        .list.sum()
        .fill_null(0)
        .cast(pl.Float64)
    )


# canonical name -> expression over nflverse player_stats columns
PLAYER_STATS: dict[str, pl.Expr] = {
    # passing (pass_yd is gross: sack yardage is not subtracted, matching the platforms)
    "pass_yd": _c("passing_yards"),
    "pass_td": _c("passing_tds"),
    "pass_int": _c("passing_interceptions"),
    "pass_2pt": _c("passing_2pt_conversions"),
    "pass_cmp": _c("completions"),
    "pass_att": _c("attempts"),
    "pass_inc": _c("attempts") - _c("completions"),
    "pass_sack": _c("sacks_suffered"),
    "pass_fd": _c("passing_first_downs"),
    # rushing
    "rush_yd": _c("rushing_yards"),
    "rush_att": _c("carries"),
    "rush_td": _c("rushing_tds"),
    "rush_2pt": _c("rushing_2pt_conversions"),
    "rush_fd": _c("rushing_first_downs"),
    # receiving
    "rec": _c("receptions"),
    "rec_tgt": _c("targets"),
    "rec_yd": _c("receiving_yards"),
    "rec_td": _c("receiving_tds"),
    "rec_2pt": _c("receiving_2pt_conversions"),
    "rec_fd": _c("receiving_first_downs"),
    # ball security and returns (fum_lost includes fumbles lost on returns)
    "fum": _c("fumbles_total"),
    "fum_lost": _c("fumbles_lost_total"),
    "st_td": _c("special_teams_tds"),
    "fum_rec_td": _c("fumble_recovery_tds"),
    "kr_yd": _c("kickoff_return_yards"),
    "pr_yd": _c("punt_return_yards"),
    # kicking (a blocked kick counts as a miss)
    "fgm": _c("fg_made"),
    "fgmiss": _c("fg_missed") + _c("fg_blocked"),
    "fgm_0_19": _c("fg_made_0_19"),
    "fgm_20_29": _c("fg_made_20_29"),
    "fgm_30_39": _c("fg_made_30_39"),
    "fgm_40_49": _c("fg_made_40_49"),
    "fgm_50_59": _c("fg_made_50_59"),
    "fgm_60p": _c("fg_made_60_"),
    "fgmiss_0_19": _c("fg_missed_0_19"),
    "fgmiss_20_29": _c("fg_missed_20_29"),
    "fgmiss_30_39": _c("fg_missed_30_39"),
    "fgmiss_40_49": _c("fg_missed_40_49"),
    "fgmiss_50_59": _c("fg_missed_50_59"),
    "fgmiss_60p": _c("fg_missed_60_"),
    "fgm_yds": _c("fg_made_distance"),
    "fgm_yds_over_30": _list_over("fg_made_list", 30),
    "xpm": _c("pat_made"),
    "xpmiss": _c("pat_missed") + _c("pat_blocked"),
}

# Derived from other canonical stats, for bonus rules.
DERIVED: dict[str, pl.Expr] = {
    "rush_rec_yd": pl.col("rush_yd") + pl.col("rec_yd"),
}

# canonical name -> expression over nflverse team_stats columns (the defense's own row)
DST_STATS: dict[str, pl.Expr] = {
    "sack": _c("def_sacks"),
    "sack_yd": _c("def_sack_yards"),
    "int": _c("def_interceptions"),
    "ff": _c("def_fumbles_forced"),
    # Includes recoveries on special teams; nflverse does not separate them.
    "fum_rec": _c("fumble_recovery_opp"),
    "safe": _c("def_safeties"),
    "def_td": _c("def_tds"),
    "def_st_td": _c("special_teams_tds"),
    "blk_kick": _c("def_fg_blocks") + _c("def_punt_blocks"),
    "qb_hit": _c("def_qb_hits"),
    "tkl_loss": _c("def_tackles_for_loss"),
    "def_pass_def": _c("def_pass_defended"),
    "def_kr_yd": _c("kickoff_return_yards"),
    "def_pr_yd": _c("punt_return_yards"),
}

PLAYER_STAT_NAMES = [*PLAYER_STATS, *DERIVED]
DST_STAT_NAMES = [*DST_STATS, "pts_allow", "yds_allow"]


def player_stat_lines(raw: pl.DataFrame) -> pl.DataFrame:
    """nflverse player_stats rows -> canonical player-week stat lines."""
    return (
        raw.select(
            pl.col("player_id").alias("key"),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            "team",
            pl.col("opponent_team").alias("opponent"),
            "position",
            *[expr.alias(name) for name, expr in PLAYER_STATS.items()],
        )
        .with_columns(*[expr.alias(name) for name, expr in DERIVED.items()])
        .filter(pl.col("key").is_not_null())
    )


def dst_stat_lines(team_stats: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """nflverse team_stats + schedules -> canonical defense-week stat lines.

    pts_allow is every point the opponent scored, including return touchdowns.
    yds_allow is the opponent's net offensive yards (sack yardage subtracted).
    """
    own = team_stats.select(
        pl.col("team"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("opponent_team").alias("opponent"),
        *[expr.alias(name) for name, expr in DST_STATS.items()],
    )
    opp_yards = team_stats.select(
        pl.col("team").alias("opponent"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        (_c("passing_yards") - _c("sack_yards_lost") + _c("rushing_yards")).alias("yds_allow"),
    )
    games = schedules.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ).filter(pl.col("home_score").is_not_null())
    allowed = pl.concat(
        [
            games.select(
                "season", "week", pl.col("home_team").alias("team"),
                pl.col("away_score").cast(pl.Float64).alias("pts_allow"),
            ),
            games.select(
                "season", "week", pl.col("away_team").alias("team"),
                pl.col("home_score").cast(pl.Float64).alias("pts_allow"),
            ),
        ]
    )  # fmt: skip
    return (
        own.join(allowed, on=["season", "week", "team"], how="inner")
        .join(opp_yards, on=["season", "week", "opponent"], how="left")
        .with_columns(
            pl.col("team").map_elements(dst_key, return_dtype=pl.Utf8).alias("key"),
            pl.lit("DEF").alias("position"),
            pl.col("yds_allow").fill_null(0),
        )
        .select(*ID_COLS, *DST_STAT_NAMES)
    )


def stat_lines(player_stats: pl.DataFrame, team_stats: pl.DataFrame, schedules: pl.DataFrame):
    """All scoreable rows (players and defenses) in one frame; missing stats are 0."""
    return pl.concat(
        [player_stat_lines(player_stats), dst_stat_lines(team_stats, schedules)],
        how="diagonal_relaxed",
    ).with_columns(pl.col(PLAYER_STAT_NAMES + DST_STAT_NAMES).fill_null(0))
