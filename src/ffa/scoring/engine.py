"""Apply scoring rules to canonical stat lines.

Works on any frame with canonical stat columns: real weekly stats for checking against
the platforms, or simulated stat lines from the projection model (one row per draw).
"""

from __future__ import annotations

import polars as pl

from ffa.scoring.rules import INF, ScoringRules


def _terms(rules: ScoringRules) -> list[tuple[str, pl.Expr]]:
    """(label, points expression) for every rule term."""
    terms: list[tuple[str, pl.Expr]] = []
    for stat, pts in rules.linear.items():
        terms.append((stat, pl.col(stat) * pts))
    for b in rules.bands:
        cond = pl.col(b.stat) >= b.lo
        if b.hi != INF:
            cond = cond & (pl.col(b.stat) < b.hi)
        if b.positions is not None:
            cond = cond & pl.col("position").is_in(list(b.positions))
        hi = "+" if b.hi == INF else f"_{b.hi:g}"
        terms.append((f"{b.stat}[{b.lo:g}{hi}]", pl.when(cond).then(b.points).otherwise(0.0)))
    for t in rules.position_terms:
        expr = pl.when(pl.col("position") == t.position).then(pl.col(t.stat) * t.points)
        terms.append((f"{t.stat}@{t.position}", expr.otherwise(0.0)))
    return terms


def _ensure(stats: pl.DataFrame, rules: ScoringRules) -> pl.DataFrame:
    missing = [s for s in rules.stats_used() if s not in stats.columns]
    return stats.with_columns(pl.lit(0.0).alias(s) for s in missing) if missing else stats


def score(stats: pl.DataFrame, rules: ScoringRules, *, column: str = "fpts") -> pl.DataFrame:
    """Add a points column."""
    terms = [expr for _, expr in _terms(rules)]
    total = pl.sum_horizontal(terms) if terms else pl.lit(0.0)
    return _ensure(stats, rules).with_columns(total.round(4).alias(column))


def breakdown(stats: pl.DataFrame, rules: ScoringRules, ids: list[str] | None = None):
    """Long format: one row per (player-week, scoring term) with nonzero points.
    This is what explanations cite: "+6.0 from rec (12 receptions x 0.5)"."""
    ids = ids or [c for c in ("key", "season", "week") if c in stats.columns]
    terms = _terms(rules)
    wide = _ensure(stats, rules).select(*ids, *[expr.alias(label) for label, expr in terms])
    return (
        wide.unpivot(index=ids, variable_name="term", value_name="points")
        .filter(pl.col("points") != 0)
        .sort([*ids, "points"], descending=[False] * len(ids) + [True])
    )
