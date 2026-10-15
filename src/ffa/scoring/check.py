"""Check our scoring engine against the points a platform actually awarded.

This is the Phase 0 gate and a standing regression check: if a platform changes how it
scores something, or a stat correction lands, this is where it shows up.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel

from ffa.ingest.ids import Resolver
from ffa.leagues.base import Platform, WeekResult
from ffa.scoring.engine import breakdown, score
from ffa.scoring.rules import ScoringRules

TOLERANCE = 0.011  # platforms round to 2 decimals


class CheckReport(BaseModel):
    compared: int
    matched: int
    mismatched: int
    unresolved: int  # platform players we couldn't map to a canonical key

    @property
    def match_rate(self) -> float:
        return self.matched / self.compared if self.compared else 1.0


def awarded_frame(weeks: dict[int, list[WeekResult]]) -> pl.DataFrame:
    """Platform-awarded points as rows of (platform_id, week, awarded)."""
    rows = [
        (pid, week, pts)
        for week, results in weeks.items()
        for r in results
        for pid, pts in r.player_points.items()
    ]
    return pl.DataFrame(
        rows, schema={"platform_id": pl.Utf8, "week": pl.Int32, "awarded": pl.Float64},
        orient="row",
    ).unique(["platform_id", "week"])  # fmt: skip


def compare(
    awarded: pl.DataFrame,
    lines: pl.DataFrame,
    rules: ScoringRules,
    resolver: Resolver,
    platform: Platform,
) -> tuple[pl.DataFrame, CheckReport]:
    """One row per awarded player-week with our points, the difference and a status.

    A player with no stat line that week (inactive, bye) scores 0 by definition.
    """

    def key_of(pid: str) -> str | None:
        pos = "DEF" if pid.isalpha() else None
        return resolver.resolve(platform.value, pid, position=pos, team=pid if pos else None).key

    ids = awarded.select("platform_id").unique()
    ids = ids.with_columns(
        pl.col("platform_id").map_elements(key_of, return_dtype=pl.Utf8).alias("key")
    )
    ours = score(lines, rules).select("key", pl.col("week").cast(pl.Int32), "position", "fpts")
    out = (
        awarded.join(ids, on="platform_id", how="left")
        .join(ours, on=["key", "week"], how="left")
        .with_columns(
            pl.when(pl.col("key").is_null())
            .then(None)
            .otherwise(pl.col("fpts").fill_null(0.0))
            .alias("ours")
        )
        .with_columns((pl.col("ours") - pl.col("awarded")).round(2).alias("diff"))
        .with_columns(
            pl.when(pl.col("key").is_null())
            .then(pl.lit("unresolved"))
            .when(pl.col("diff").abs() <= TOLERANCE)
            .then(pl.lit("match"))
            .otherwise(pl.lit("mismatch"))
            .alias("status")
        )
        .drop("fpts")
        .sort(["status", "week", "platform_id"])
    )
    counts = dict(out.group_by("status").len().iter_rows())
    report = CheckReport(
        compared=out.height - counts.get("unresolved", 0),
        matched=counts.get("match", 0),
        mismatched=counts.get("mismatch", 0),
        unresolved=counts.get("unresolved", 0),
    )
    return out, report


def explain(mismatches: pl.DataFrame, lines: pl.DataFrame, rules: ScoringRules) -> pl.DataFrame:
    """Our per-term breakdown for each mismatched player-week, to see which term is off."""
    keys = mismatches.select("key", pl.col("week").cast(pl.Int32)).drop_nulls()
    subset = lines.with_columns(pl.col("week").cast(pl.Int32)).join(keys, on=["key", "week"])
    return breakdown(subset, rules, ids=["key", "week"])
