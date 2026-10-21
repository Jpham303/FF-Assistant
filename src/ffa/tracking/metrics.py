"""Weekly accuracy, by league scoring and position, against simple baselines.

Phase 1 metrics (spec): MAE when active (proj_if_active), MAE on expected points (zeros
included), bias, and p10-p90 / p25-p75 coverage. Baselines: the player's average over
his last 4 games played and his season average, both from games before the week, in the
same scoring. Each row carries n, so weeks can be pooled (rolling 4, season to date).
"""

from __future__ import annotations

import json

import polars as pl

from ffa.scoring import score
from ffa.scoring.rules import from_sleeper

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
RELEVANT = {"QB": 12, "TE": 12, "RB": 36, "WR": 36}  # top N by season average to date


def league_points(actuals: pl.DataFrame, scoring_json: str) -> pl.DataFrame:
    rules = from_sleeper(json.loads(scoring_json))
    return score(actuals, rules).select("key", "season", "week", "active", "fpts")


def baselines(points: pl.DataFrame, week: int) -> pl.DataFrame:
    """Last-4 and season average per player from active games before `week`."""
    prior = points.filter((pl.col("week") < week) & pl.col("active")).sort("week")
    return prior.group_by("key").agg(
        pl.col("fpts").tail(4).mean().alias("last4"),
        pl.col("fpts").mean().alias("season_avg"),
        pl.len().alias("games"),
    )


def week_rows(snap: pl.DataFrame, actuals: pl.DataFrame, week: int) -> pl.DataFrame:
    """Per (league, player) for one week: frozen projection, actual points, baselines."""
    out = []
    for (_league, scoring), s in snap.group_by("league", "scoring"):
        pts = league_points(actuals, scoring)
        cur = pts.filter(pl.col("week") == week).select(
            "key", pl.col("active").alias("was_active"), pl.col("fpts").alias("actual")
        )
        b = baselines(pts, week)
        out.append(
            s.join(cur, on="key", how="left")
            .join(b, on="key", how="left")
            .with_columns(pl.col("was_active").fill_null(False), pl.col("actual").fill_null(0.0))
        )
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def _relevant(rows: pl.DataFrame) -> pl.Expr:
    rank = pl.col("season_avg").rank("ordinal", descending=True).over("league", "position")
    limit = pl.col("position").replace_strict(RELEVANT, default=0)
    return pl.col("season_avg").is_not_null() & (rank <= limit)


def metrics(rows: pl.DataFrame) -> pl.DataFrame:
    """Long table: league, position, scope, source, metric, value, n. Scopes: all
    projected players, and fantasy-relevant ones (top 12 QB/TE, 36 RB/WR by season
    average to date, a ranking independent of the model)."""
    rows = rows.with_columns(_relevant(rows).alias("relevant"))
    recs: list[dict] = []
    for scope, frame in (("all", rows), ("relevant", rows.filter("relevant"))):
        for (league, pos), g in frame.group_by("league", "position"):

            def add(source: str, metric: str, value, n: int,
                    league=league, pos=pos, scope=scope) -> None:  # fmt: skip
                v = None if value is None else float(value)
                recs.append({"league": league, "position": pos, "scope": scope,
                             "source": source, "metric": metric, "value": v,
                             "n": int(n)})  # fmt: skip

            add("model", "mae_expected", (g["expected_points"] - g["actual"]).abs().mean(),
                g.height)  # fmt: skip
            act = g.filter("was_active")
            if not act.height:
                continue
            e = act["proj_if_active"] - act["actual"]
            add("model", "mae_if_active", e.abs().mean(), act.height)
            add("model", "bias", e.mean(), act.height)
            inside = lambda lo, hi, a=act: a.select(  # noqa: E731
                pl.col("actual").is_between(pl.col(lo), pl.col(hi)).mean()
            ).item()
            add("model", "coverage_80", inside("p10", "p90"), act.height)
            add("model", "coverage_50", inside("p25", "p75"), act.height)
            for base in ("last4", "season_avg"):
                bb = act.filter(pl.col(base).is_not_null())
                if bb.height:  # baseline and model on the same players: like for like
                    add(base, "mae_if_active", (bb[base] - bb["actual"]).abs().mean(), bb.height)
                    add(f"model_vs_{base}", "mae_if_active",
                        (bb["proj_if_active"] - bb["actual"]).abs().mean(), bb.height)  # fmt: skip
    return pl.DataFrame(recs, schema={
        "league": pl.Utf8, "position": pl.Utf8, "scope": pl.Utf8, "source": pl.Utf8,
        "metric": pl.Utf8, "value": pl.Float64, "n": pl.Int64,
    })  # fmt: skip


def pooled(results: pl.DataFrame) -> pl.DataFrame:
    """Pool weekly rows (any set of weeks): n-weighted mean of each metric."""
    return (
        results.filter(pl.col("value").is_not_null())
        .group_by("league", "position", "scope", "source", "metric")
        .agg(
            ((pl.col("value") * pl.col("n")).sum() / pl.col("n").sum()).alias("value"),
            pl.col("n").sum(),
            pl.col("week").n_unique().alias("weeks"),
        )  # fmt: skip
    )
