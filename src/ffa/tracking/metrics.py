"""Weekly accuracy, by league scoring and position, against baselines and Sleeper.

Phase 1 metrics (spec): MAE when active (proj_if_active), MAE on expected points (zeros
included), bias, and p10-p90 / p25-p75 coverage. Baselines: the player's average over
his last 4 games played and his season average, both from games before the week, in the
same scoring. Each row carries n, so weeks can be pooled (rolling 4, season to date).

Phase 2: every miss goes into one bucket. An availability miss (given 50%+ to play, took
no snap) counts against the injury inputs; an early exit (under 25% of his usual snap
share: hurt or benched) is reported on its own; everything else is a model miss, and only
model misses feed MAE if active, bias and coverage (`mae_if_active_all` keeps exits in).
Against Sleeper's projections, frozen at the same moments and scored in the same rules,
on the same players: MAE, rank correlation within the position, and start/sit accuracy
on close pairs. Availability: Brier score of play_probability by injury status.
"""

from __future__ import annotations

import json

import polars as pl

from ffa.scoring import score
from ffa.scoring.rules import from_sleeper

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
RELEVANT = {"QB": 12, "TE": 12, "RB": 36, "WR": 36}  # top N by season average to date
EARLY_EXIT = 0.25  # share of his usual snap share below which a game counts as an exit
AVAILABILITY_MISS = 0.5  # given at least this chance to play, but took no snap
# ...and projected for a role. A healthy backup (QB2, TE3) is dressed and never takes a
# snap; that's not an injury-input miss, and it barely moves accuracy.
ROLE_MIN = 3.0  # points if active
CLOSE_PAIR = 3.0  # start/sit pairs: season averages within this many points


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


def usual_snaps(actuals: pl.DataFrame, week: int) -> pl.DataFrame:
    """Per player: snap share this week and his average in active games before it."""
    if "snap_pct" not in actuals.columns:
        return pl.DataFrame(schema={"key": pl.Utf8, "snap_pct": pl.Float64,
                                    "usual_snap_pct": pl.Float64})  # fmt: skip
    prior = actuals.filter((pl.col("week") < week) & pl.col("active")
                           & pl.col("snap_pct").is_not_null())  # fmt: skip
    usual = prior.group_by("key").agg(pl.col("snap_pct").mean().alias("usual_snap_pct"))
    cur = actuals.filter(pl.col("week") == week).select("key", "snap_pct")
    return cur.join(usual, on="key", how="full", coalesce=True)


def week_rows(snap: pl.DataFrame, actuals: pl.DataFrame, week: int,
              bench: pl.DataFrame | None = None) -> pl.DataFrame:  # fmt: skip
    """Per (league, player) for one week: frozen projection, actual points, baselines,
    snap shares, and Sleeper's frozen projection (`sleeper`) when one was captured."""
    out = []
    snaps = usual_snaps(actuals, week)
    for (league, scoring), s in snap.group_by("league", "scoring"):
        pts = league_points(actuals, scoring)
        cur = pts.filter(pl.col("week") == week).select(
            "key", pl.col("active").alias("was_active"), pl.col("fpts").alias("actual")
        )
        b = baselines(pts, week)
        f = (
            s.join(cur, on="key", how="left")
            .join(b, on="key", how="left")
            .join(snaps, on="key", how="left")
            .with_columns(pl.col("was_active").fill_null(False), pl.col("actual").fill_null(0.0))
        )
        if bench is not None and bench.height:
            sl = bench.filter(pl.col("league") == league).select(
                "key", pl.col("points").alias("sleeper")
            )
            f = f.join(sl, on="key", how="left")
        else:
            f = f.with_columns(pl.lit(None, pl.Float64).alias("sleeper"))
        out.append(f)
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def buckets(rows: pl.DataFrame) -> pl.Expr:
    """availability_miss / early_exit / model / not_expected (sat: as projected, or a
    backup with no role)."""
    p = pl.col("play_probability").fill_null(1.0)
    role = pl.col("proj_if_active").fill_null(0) >= ROLE_MIN
    exit_ = (pl.col("usual_snap_pct").fill_null(0) > 0) & (
        pl.col("snap_pct").fill_null(1.0) < EARLY_EXIT * pl.col("usual_snap_pct")
    )
    return (
        pl.when(~pl.col("was_active") & (p >= AVAILABILITY_MISS) & role)
        .then(pl.lit("availability_miss"))
        .when(~pl.col("was_active")).then(pl.lit("not_expected"))
        .when(exit_ & ~pl.col("position").is_in(["K", "DEF"])).then(pl.lit("early_exit"))
        .otherwise(pl.lit("model"))
    )  # fmt: skip


def _spearman(a: pl.Series, b: pl.Series) -> float | None:
    if a.len() < 3:
        return None
    ra, rb = a.rank("average"), b.rank("average")
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(pl.DataFrame({"a": ra, "b": rb}).select(pl.corr("a", "b")).item())


def _pairs(g: pl.DataFrame, col: str) -> tuple[float | None, int]:
    """Close same-position pairs (season averages within CLOSE_PAIR): share where the
    player `col` rates higher outscored the other. Ties (actual or projected) skipped."""
    d = g.filter(pl.col("season_avg").is_not_null() & pl.col(col).is_not_null()).select(
        "key", "season_avg", col, "actual"
    )
    if d.height < 2:
        return None, 0
    x = d.join(d, how="cross", suffix="_b").filter(
        (pl.col("key") < pl.col("key_b"))
        & ((pl.col("season_avg") - pl.col("season_avg_b")).abs() <= CLOSE_PAIR)
        & (pl.col(col) != pl.col(f"{col}_b")) & (pl.col("actual") != pl.col("actual_b"))
    )  # fmt: skip
    if x.height == 0:
        return None, 0
    right = (pl.col(col) > pl.col(f"{col}_b")) == (pl.col("actual") > pl.col("actual_b"))
    return float(x.select(right.mean()).item()), x.height


def _relevant(rows: pl.DataFrame) -> pl.Expr:
    rank = pl.col("season_avg").rank("ordinal", descending=True).over("league", "position")
    limit = pl.col("position").replace_strict(RELEVANT, default=0)
    return pl.col("season_avg").is_not_null() & (rank <= limit)


def metrics(rows: pl.DataFrame) -> pl.DataFrame:
    """Long table: league, position, scope, source, metric, value, n. Scopes: all
    projected players, and fantasy-relevant ones (top 12 QB/TE, 36 RB/WR by season
    average to date, a ranking independent of the model)."""
    if "play_probability" not in rows.columns:  # older snapshots: assume he was expected
        rows = rows.with_columns(pl.lit(1.0).alias("play_probability"))
    for c, t in (("sleeper", pl.Float64), ("snap_pct", pl.Float64),
                 ("usual_snap_pct", pl.Float64)):  # fmt: skip
        if c not in rows.columns:
            rows = rows.with_columns(pl.lit(None, t).alias(c))
    rows = rows.with_columns(_relevant(rows).alias("relevant"), buckets(rows).alias("bucket"))
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
            for b in ("availability_miss", "early_exit", "model", "not_expected"):
                add("misses", b, None if g.height == 0 else (g["bucket"] == b).mean(), g.height)
            both = g.filter(pl.col("sleeper").is_not_null())
            if both.height:  # Sleeper on the same players, zeros included
                add(
                    "model_vs_sleeper",
                    "mae_expected",
                    (both["expected_points"] - both["actual"]).abs().mean(),
                    both.height,
                )
                add(
                    "sleeper",
                    "mae_expected",
                    (both["sleeper"] - both["actual"]).abs().mean(),
                    both.height,
                )
                add(
                    "model_vs_sleeper",
                    "spearman",
                    _spearman(both["expected_points"], both["actual"]),
                    both.height,
                )
                add("sleeper", "spearman", _spearman(both["sleeper"], both["actual"]), both.height)
                for src, col in (("model_vs_sleeper", "expected_points"), ("sleeper", "sleeper")):
                    acc, n = _pairs(both, col)
                    if n:
                        add(src, "start_sit", acc, n)
            if pos not in ("K", "DEF"):
                pp = g["play_probability"].fill_null(1.0)
                add(
                    "model",
                    "availability_brier",
                    ((pp - g["was_active"].cast(pl.Float64)) ** 2).mean(),
                    g.height,
                )
            alle = g.filter("was_active")
            if alle.height:
                add(
                    "model",
                    "mae_if_active_all",
                    (alle["proj_if_active"] - alle["actual"]).abs().mean(),
                    alle.height,
                )
            act = g.filter(pl.col("bucket") == "model")
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
            sl = act.filter(pl.col("sleeper").is_not_null())
            if sl.height:  # Sleeper has no separate if-active number: its projection as is
                add(
                    "sleeper",
                    "mae_if_active",
                    (sl["sleeper"] - sl["actual"]).abs().mean(),
                    sl.height,
                )
                add(
                    "model_vs_sleeper",
                    "mae_if_active",
                    (sl["proj_if_active"] - sl["actual"]).abs().mean(),
                    sl.height,
                )
    # Availability calibration by injury status (skill players projected for a role, all
    # positions together): said vs happened and the Brier score, per league.
    skill = rows.filter(~pl.col("position").is_in(["K", "DEF"])
                        & (pl.col("proj_if_active").fill_null(0) >= ROLE_MIN))  # fmt: skip
    if "injury_status" in skill.columns and skill.height:
        st = pl.col("injury_status").fill_null("None").replace("", "None")
        for (league, status), g in skill.group_by("league", st.alias("st")):
            pp = g["play_probability"].fill_null(1.0)
            y = g["was_active"].cast(pl.Float64)
            for metric, v in (("said", pp.mean()), ("happened", y.mean()),
                              ("brier", ((pp - y) ** 2).mean())):  # fmt: skip
                recs.append({"league": league, "position": "ALL", "scope": "all",
                             "source": f"availability:{status}", "metric": metric,
                             "value": float(v), "n": g.height})  # fmt: skip
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
