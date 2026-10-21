"""What happened: canonical stat lines, snaps and an active flag per player-week, from
nflverse after stat corrections. League points are computed at scoring time with the
rules stored in each snapshot."""

from __future__ import annotations

import polars as pl

from ffa.scoring import stat_lines
from ffa.scoring.defense import load_pbp

DATASET = "actuals"


def build(con, season: int, weeks: list[int]) -> pl.DataFrame:
    """Stat lines for completed `weeks`, with snaps, snap share and active (played)."""
    q = lambda t: con.sql(f"select * from {t} where season = {int(season)}").pl()  # noqa: E731
    lines = stat_lines(q("player_stats"), load_pbp(con, season), q("schedules"))
    lines = lines.filter(pl.col("week").is_in(weeks)).with_columns(
        pl.col("week").cast(pl.Int32), pl.lit(season).cast(pl.Int32).alias("season")
    )
    snaps = (
        con.sql(
            f"""select key, week, max(snaps) as snaps, max(snap_pct) as snap_pct,
                   bool_or(played) as played
            from player_games where season = {int(season)} group by 1, 2"""
        )
        .pl()
        .with_columns(pl.col("week").cast(pl.Int32))
        .filter(pl.col("week").is_in(weeks))
    )
    # Full join: a player who took snaps but recorded no stat (no targets, no carries) is
    # active with zero points, not missing.
    out = lines.join(snaps, on=["key", "week"], how="full", coalesce=True)
    num = [c for c, t in out.schema.items() if t.is_numeric() and c not in
           ("week", "season", "snaps", "snap_pct")]  # fmt: skip
    return out.with_columns(
        pl.col(num).fill_null(0),
        pl.lit(season).cast(pl.Int32).alias("season"),
        # Played = took a snap, or recorded a stat. Defenses: always (the team played).
        (pl.col("key").str.starts_with("DST_") | pl.col("played").fill_null(False)
         | pl.col("snaps").fill_null(0).gt(0) | pl.col("position").is_not_null()
         & pl.col("played").is_null()).alias("active"),
    ).drop("played")  # fmt: skip


def completed_weeks(con, season: int) -> list[int]:
    """Weeks whose games have all been played."""
    rows = con.sql(
        f"""select week from schedules where season = {int(season)} and game_type = 'REG'
            group by week having bool_and(home_score is not null) order by week"""
    ).fetchall()
    return [int(r[0]) for r in rows]
