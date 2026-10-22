"""One definition of the if-active projection, shared by the API and the snapshots.

- proj_if_active and its range: points over the simulations where the player is active
  (takes a snap). When fewer than MIN_ACTIVE such simulations exist (low chance to
  play), they come from an extra batch with him forced active, teammates as usual.
- play_probability: the model's chance he plays (the share of simulations he is active
  in, up to sampling noise).
- expected_points = play_probability x proj_if_active: what the lineup is ranked by.
- Not playing (play_probability 0: Out, IR, PUP, suspended, bye): if-active fields are
  null and expected points are 0.
"""

from __future__ import annotations

import polars as pl

MIN_ACTIVE = 500
QUANTILES = [round(i / 20, 2) for i in range(21)]  # 0, 0.05, ..., 1
RANGE = {"p10": 0.1, "p25": 0.25, "median": 0.5, "p75": 0.75, "p90": 0.9}


def active_flag(sims: pl.DataFrame) -> pl.Expr:
    """The simulator marks `active` for skill players; kickers and defenses it simulates
    are playing."""
    if "active" in sims.columns:
        return pl.col("active").fill_null(True)
    return pl.lit(True)


def summarize(points: pl.DataFrame, play: dict[str, float],
              forced: pl.DataFrame | None = None) -> pl.DataFrame:  # fmt: skip
    """points: key, sim, fpts, active (main batch). forced: key, sim, fpts (forced-active
    batches). play: key -> play probability (missing = 1.0). One row per key."""
    act = points.filter(pl.col("active")).select("key", "fpts")
    if forced is not None and forced.height:
        # where a forced batch exists, it replaces the (too few) active simulations
        act = pl.concat([act.join(forced.select("key"), on="key", how="anti"),
                         forced.select("key", "fpts")])  # fmt: skip
        forced_keys = set(forced["key"].unique().to_list())
    else:
        forced_keys = set()
    cond = act.group_by("key").agg(
        pl.len().alias("active_sims"),
        pl.col("fpts").mean().alias("proj_if_active"),
        *[pl.col("fpts").quantile(q, "linear").alias(f"{n}_if_active") for n, q in RANGE.items()],
        pl.concat_list([pl.col("fpts").quantile(q, "linear") for q in QUANTILES])
        .alias("quantiles_if_active"),
    )  # fmt: skip
    keys = points.select("key").unique()
    out = keys.join(cond, on="key", how="left").with_columns(
        pl.col("key").replace_strict(play, default=1.0, return_dtype=pl.Float64)
        .alias("play_probability"),
        pl.col("key").is_in(list(forced_keys)).alias("forced_active_batch"),
        pl.col("active_sims").fill_null(0),
    )  # fmt: skip
    if_cols = ["proj_if_active", *[f"{n}_if_active" for n in RANGE], "quantiles_if_active"]
    not_playing = pl.col("play_probability") <= 0
    return out.with_columns(
        *[pl.when(not_playing).then(None).otherwise(pl.col(c)).alias(c) for c in if_cols],
    ).with_columns(
        (pl.col("play_probability") * pl.col("proj_if_active").fill_null(0.0)).alias(
            "expected_points"
        ),
    )
