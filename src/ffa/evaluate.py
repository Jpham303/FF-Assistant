"""Out-of-sample evaluation of weekly projections against baselines.

For each week, projections use only data from before that week. Two baselines are built
the same way, from the player's earlier games only:

    season_avg    his average fantasy points so far this season (last season's if none)
    opportunity   his average ffopportunity *expected* points so far this season: points
                  his opportunities were worth at league-average efficiency, scored with
                  the league's own rules. (The same week's expected points are not a
                  projection: they are computed from that game's actual opportunities.)

Metrics: MAE by position on player-weeks where the player played; share of outcomes
inside the 10th-90th percentile range; and start/sit accuracy on same-position pairs of
realistically startable players, with week-level bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ffa.model.inputs import Models, build
from ffa.model.sim import points, simulate
from ffa.scoring import score, stat_lines
from ffa.scoring.defense import load_pbp
from ffa.scoring.rules import ScoringRules

# ffopportunity expected-stat columns -> canonical stats
XFP_MAP = {
    "pass_yd": "pass_yards_gained_exp",
    "pass_td": "pass_touchdown_exp",
    "pass_int": "pass_interception_exp",
    "pass_2pt": "pass_two_point_conv_exp",
    "rec": "receptions_exp",
    "rec_yd": "rec_yards_gained_exp",
    "rec_td": "rec_touchdown_exp",
    "rec_2pt": "rec_two_point_conv_exp",
    "rush_yd": "rush_yards_gained_exp",
    "rush_td": "rush_touchdown_exp",
    "rush_2pt": "rush_two_point_conv_exp",
}
POOL = {"QB": 24, "RB": 36, "WR": 48, "TE": 24, "K": 20, "DEF": 20}  # startable per week


def actual_points(con, season: int, rules: ScoringRules) -> pl.DataFrame:
    q = lambda t: con.sql(f"select * from {t} where season = {season}").pl()  # noqa: E731
    lines = stat_lines(q("player_stats"), load_pbp(con, season), q("schedules"))
    return score(lines, rules).select(
        "key", pl.col("week").cast(pl.Int32), "position", pl.col("fpts").alias("actual")
    )


def xfp_points(con, season: int, rules: ScoringRules) -> pl.DataFrame:
    ep = con.sql(f"select * from expected_points where season = {season}").pl()
    lines = ep.select(
        pl.col("player_id").alias("key"),
        pl.col("week").cast(pl.Int32),
        "position",
        *[pl.col(src).fill_null(0).cast(pl.Float64).alias(dst) for dst, src in XFP_MAP.items()],
        (pl.col("rec_fumble_lost").fill_null(0) + pl.col("rush_fumble_lost").fill_null(0))
        .cast(pl.Float64)
        .alias("fum_lost"),
    )
    return score(lines, rules).select("key", "week", pl.col("fpts").alias("xfp"))


def _prior_mean(df: pl.DataFrame, col: str, last_season: pl.DataFrame | None) -> pl.DataFrame:
    """For every (key, week) in 1..22: mean of `col` over earlier weeks this season, or
    last season's mean when there are none."""
    weeks = pl.DataFrame({"week": list(range(1, 23))}, schema={"week": pl.Int32})
    grid = df.select("key").unique().join(weeks, how="cross")
    if last_season is not None:
        grid = pl.concat(
            [grid, last_season.select("key").unique().join(weeks, how="cross")]
        ).unique()
    hist = grid.join(df.select("key", pl.col("week").alias("w"), col), on="key", how="left")
    cur = (
        hist.filter(pl.col("w") < pl.col("week"))
        .group_by("key", "week")
        .agg(pl.col(col).mean().alias("cur"))
    )
    out = grid.join(cur, on=["key", "week"], how="left")
    if last_season is not None:
        prev = last_season.group_by("key").agg(pl.col(col).mean().alias("prev"))
        out = out.join(prev, on="key", how="left").with_columns(
            pl.coalesce("cur", "prev").alias("cur")
        )
    return out.select("key", "week", pl.col("cur").alias(f"base_{col}"))


@dataclass
class Holdout:
    rows: pl.DataFrame  # key, week, position, proj, p10, p90, actual, played, baselines
    draws: pl.DataFrame  # key, week, sim, fpts


def run(
    con,
    season: int,
    weeks: list[int],
    models: Models,
    rules: ScoringRules,
    n_sims: int = 1000,
    frames: dict[str, pl.DataFrame] | None = None,
) -> Holdout:
    q = lambda t: con.sql(f"select * from {t} where season <= {season}").pl()  # noqa: E731
    f = frames or {
        "team_games": q("team_games"),
        "player_games": q("player_games"),
        "rosters": con.sql(f"select * from rosters where season = {season}").pl(),
        "injuries": con.sql(f"select * from injuries where season = {season}").pl(),
    }
    projections, draws = [], []
    for week in weeks:
        inp = build(season=season, week=week, models=models, **f)
        pts = points(simulate(inp, models.sim, n=n_sims, seed=season * 100 + week), rules)
        draws.append(pts.select("key", pl.lit(week).cast(pl.Int32).alias("week"), "sim", "fpts"))
        projections.append(
            pts.group_by("key", "name", "position")
            .agg(
                pl.col("fpts").mean().alias("proj"),
                pl.col("fpts").quantile(0.1).alias("p10"),
                pl.col("fpts").quantile(0.9).alias("p90"),
            )
            .with_columns(pl.lit(week).cast(pl.Int32).alias("week"))
        )
    proj = pl.concat(projections)

    act = actual_points(con, season, rules)
    prev_act = actual_points(con, season - 1, rules)
    xfp, prev_xfp = xfp_points(con, season, rules), xfp_points(con, season - 1, rules)
    rows = (
        proj.join(act.drop("position"), on=["key", "week"], how="left")
        .with_columns(
            pl.col("actual").is_not_null().alias("played"), pl.col("actual").fill_null(0.0)
        )
        .join(
            _prior_mean(act, "actual", prev_act).rename({"base_actual": "base_season_avg"}),
            on=["key", "week"],
            how="left",
        )
        .join(
            _prior_mean(xfp, "xfp", prev_xfp).rename({"base_xfp": "base_opportunity"}),
            on=["key", "week"],
            how="left",
        )
    )
    return Holdout(rows, pl.concat(draws))


# ---------------------------------------------------------------- metrics


def _boot(weeks: np.ndarray, values: np.ndarray, stat, n: int = 2000, seed: int = 0):
    """95% CI of stat(values) resampling whole weeks (rows within a week are dependent)."""
    rng = np.random.default_rng(seed)
    uw = np.unique(weeks)
    idx = {w: np.nonzero(weeks == w)[0] for w in uw}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[w] for w in rng.choice(uw, len(uw))])
        out.append(stat(values[pick]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def accuracy(h: Holdout) -> pl.DataFrame:
    """MAE by position on player-weeks where the player played and every method has a
    projection; the model's edge over each baseline with a week-bootstrap 95% CI."""
    out = []
    played = h.rows.filter(pl.col("played"))
    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        d = played.filter(pl.col("position") == pos)
        bases = ["base_season_avg"] + (
            ["base_opportunity"] if pos in ("QB", "RB", "WR", "TE") else []
        )
        d = d.drop_nulls(bases)
        if d.height == 0:
            continue
        a = d["actual"].to_numpy()
        rec = {
            "position": pos,
            "n": d.height,
            "mae_model": float(np.mean(np.abs(d["proj"].to_numpy() - a))),
        }
        for b in bases:
            gain = np.abs(d[b].to_numpy() - a) - np.abs(d["proj"].to_numpy() - a)
            rec[f"mae_{b[5:]}"] = float(np.mean(np.abs(d[b].to_numpy() - a)))
            lo, hi = _boot(d["week"].to_numpy(), gain, np.mean)
            rec[f"edge_vs_{b[5:]}"] = float(gain.mean())
            rec[f"edge_vs_{b[5:]}_ci"] = f"[{lo:+.2f}, {hi:+.2f}]"
        out.append(rec)
    return pl.DataFrame(out)


def coverage(h: Holdout, min_proj: float = 0.0) -> pl.DataFrame:
    d = h.rows.filter(pl.col("proj") >= min_proj)
    inside = (pl.col("actual") >= pl.col("p10")) & (pl.col("actual") <= pl.col("p90"))
    return d.group_by("position").agg(
        pl.len().alias("n"),
        inside.mean().alias("in_10_90"),
        (pl.col("actual") < pl.col("p10")).mean().alias("below"),
        (pl.col("actual") > pl.col("p90")).mean().alias("above"),
    )


def start_sit_pairs(h: Holdout) -> pl.DataFrame:
    """Same-position pairs, same week, both in the startable pool (top POOL by season-to-date
    average, which keeps the pool independent of the model). Only pairs where both played
    and their actual scores differ."""
    pool = (
        h.rows.filter(pl.col("played") & pl.col("base_season_avg").is_not_null())
        .with_columns(
            pl.col("base_season_avg")
            .rank("ordinal", descending=True)
            .over("week", "position")
            .alias("rk")
        )
        .filter(pl.col("rk") <= pl.col("position").replace_strict(POOL, default=0))
    )
    a = pool.select(pl.all().name.prefix("a_"))
    b = pool.select(pl.all().name.prefix("b_"))
    pairs = a.join(b, left_on=["a_week", "a_position"], right_on=["b_week", "b_position"])
    return pairs.filter(
        (pl.col("a_key") < pl.col("b_key")) & (pl.col("a_actual") != pl.col("b_actual"))
    )


def pair_probabilities(draws: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """P(A outscores B) for each pair from the joint simulations (ties count half).
    Computed week by week on arrays: a row-wise join of pairs x simulations does not fit
    in memory (48k pairs x 1,000 sims)."""
    out = []
    for (week,), wd in draws.group_by("week"):
        wp = pairs.filter(pl.col("a_week") == week)
        if wp.height == 0:
            continue
        mat = wd.pivot(on="sim", index="key", values="fpts")
        keys = {k: i for i, k in enumerate(mat["key"].to_list())}
        arr = mat.drop("key").to_numpy()
        ok = [(a in keys) and (b in keys) for a, b in zip(wp["a_key"], wp["b_key"], strict=True)]
        wp = wp.filter(pl.Series(ok))
        ia = np.array([keys[k] for k in wp["a_key"]])
        ib = np.array([keys[k] for k in wp["b_key"]])
        fa, fb = arr[ia], arr[ib]
        p = ((fa > fb) + 0.5 * (fa == fb)).mean(axis=1)
        out.append(wp.select("a_key", "b_key", "a_week").with_columns(pl.Series("p_a", p)))
    return pl.concat(out)


def start_sit(h: Holdout) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pair accuracy per method (overall and on close calls), plus calibration of the
    model's P(A outscores B) computed from the joint simulations."""
    pairs = start_sit_pairs(h)
    pairs = pairs.join(pair_probabilities(h.draws, pairs), on=["a_key", "b_key", "a_week"])
    a_won = pl.col("a_actual") > pl.col("b_actual")
    methods = {
        "model": ("a_proj", "b_proj"),
        "season_avg": ("a_base_season_avg", "b_base_season_avg"),
        "opportunity": ("a_base_opportunity", "b_base_opportunity"),
    }
    rows = []
    close = (pl.col("a_base_season_avg") - pl.col("b_base_season_avg")).abs() <= 3
    for subset, cond in (("all pairs", pl.lit(True)), ("close calls (avg within 3 pts)", close)):
        sub = pairs.filter(cond)
        for m, (ca, cb) in methods.items():
            s = sub.drop_nulls([ca, cb]).filter(pl.col(ca) != pl.col(cb))
            if m == "opportunity":
                s = s.filter(pl.col("a_position").is_in(["QB", "RB", "WR", "TE"]))
            right = ((pl.col(ca) > pl.col(cb)) == a_won).cast(pl.Float64)
            vals = s.select(right)[:, 0].to_numpy()
            lo, hi = _boot(s["a_week"].to_numpy(), vals, np.mean)
            rows.append(
                {
                    "subset": subset,
                    "method": m,
                    "pairs": s.height,
                    "accuracy": float(vals.mean()),
                    "ci95": f"[{lo:.3f}, {hi:.3f}]",
                }
            )
    # Paired comparison: same pairs judged by both methods; resample weeks.
    for subset, cond in (("all pairs", pl.lit(True)), ("close calls (avg within 3 pts)", close)):
        for base, (ba, bb) in list(methods.items())[1:]:
            s = (
                pairs.filter(cond)
                .drop_nulls([ba, bb])
                .filter((pl.col(ba) != pl.col(bb)) & (pl.col("a_proj") != pl.col("b_proj")))
            )
            if base == "opportunity":
                s = s.filter(pl.col("a_position").is_in(["QB", "RB", "WR", "TE"]))
            diff = s.select(
                ((pl.col("a_proj") > pl.col("b_proj")) == a_won).cast(pl.Float64)
                - ((pl.col(ba) > pl.col(bb)) == a_won).cast(pl.Float64)
            )[:, 0].to_numpy()
            lo, hi = _boot(s["a_week"].to_numpy(), diff, np.mean)
            rows.append(
                {
                    "subset": subset,
                    "method": f"model minus {base}",
                    "pairs": s.height,
                    "accuracy": float(diff.mean()),
                    "ci95": f"[{lo:+.3f}, {hi:+.3f}]",
                }
            )
    calib = (
        pairs.with_columns(
            pl.max_horizontal("p_a", 1 - pl.col("p_a")).alias("p_fav"),
            pl.when(pl.col("p_a") >= 0.5).then(a_won).otherwise(~a_won).alias("fav_won"),
        )
        .with_columns(pl.col("p_fav").cut([0.55, 0.6, 0.65, 0.7, 0.8, 0.9]).alias("bucket"))
        .group_by("bucket")
        .agg(
            pl.len().alias("pairs"),
            pl.col("p_fav").mean().alias("said"),
            pl.col("fav_won").mean().alias("happened"),
        )
        .sort("bucket")
    )
    return pl.DataFrame(rows), calib
