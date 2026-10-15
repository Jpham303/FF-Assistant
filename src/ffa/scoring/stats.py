"""Canonical stat lines, built from nflverse.

Every scoring rule is written against these names, whatever the platform. They follow
Sleeper's vocabulary because it is the most granular public one; Yahoo stat IDs are mapped
onto them. One row per player-week (key = gsis_id) or defense-week (key = DST_<TEAM>).
"""

from __future__ import annotations

import polars as pl

from ffa.scoring.defense import DST_STAT_NAMES, dst_stat_lines

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

PLAYER_STAT_NAMES = [*PLAYER_STATS, *DERIVED]


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


def stat_lines(player_stats: pl.DataFrame, pbp: pl.DataFrame, schedules: pl.DataFrame):
    """All scoreable rows (players and defenses) in one frame; missing stats are 0."""
    return pl.concat(
        [player_stat_lines(player_stats), dst_stat_lines(pbp, schedules)],
        how="diagonal_relaxed",
    ).with_columns(pl.col(PLAYER_STAT_NAMES + DST_STAT_NAMES).fill_null(0))
