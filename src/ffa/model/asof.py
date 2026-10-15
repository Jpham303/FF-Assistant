"""Recency-weighted history as of each game, using strictly earlier games only.

Every projection input goes through here, which is what keeps the model point-in-time
correct: a row's estimate never sees its own game or anything after it.

Time runs on a game index where an offseason counts as `offseason_gap` extra weeks, so
last season still informs this one but fades faster than the recent weeks do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Decay:
    half_life: float = 6.0  # games until a past game's weight halves
    offseason_gap: float = 8.0  # extra "weeks" of decay between seasons

    def time(self) -> pl.Expr:
        return pl.col("season") * (22 + self.offseason_gap) + pl.col("week")


def decayed(
    df: pl.DataFrame,
    by: str,
    ratios: dict[str, tuple[str, str | None]],
    decay: Decay,
) -> pl.DataFrame:
    """Add, for each name in `ratios`, the decayed ratio of earlier games and its weight.

    ratios: name -> (numerator column, denominator column or None for a plain mean).
    Adds `{name}` (the ratio, null when there is no history) and `{name}_w` (effective
    amount of evidence: summed decay weights times the denominator).
    Games whose numerator is null (not yet played) contribute nothing.
    """
    df = df.with_columns(decay.time().cast(pl.Float64).alias("_t")).with_row_index("_row")
    num_cols = [n for n, _ in ratios.values()]
    den_cols = [d for _, d in ratios.values()]
    parts = []
    for _, g in df.sort("_t").group_by(by, maintain_order=True):
        t = g["_t"].to_numpy()
        # w[j, i] = weight of game i in game j's estimate: earlier games only.
        lag = t[:, None] - t[None, :]
        w = np.where(lag > 0, 0.5 ** (lag / decay.half_life), 0.0)
        out = {"_row": g["_row"].to_numpy()}
        for name, num, den in zip(ratios, num_cols, den_cols, strict=True):
            x = g[num].to_numpy().astype(float)
            played = ~np.isnan(x)
            d = np.ones_like(x) if den is None else g[den].to_numpy().astype(float)
            d = np.where(played & ~np.isnan(d), d, 0.0)
            x = np.where(played, x, 0.0)
            wd = w @ d
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = np.where(wd > 0, (w @ x) / wd, np.nan)
            out[f"{name}_w"] = wd
        parts.append(pl.DataFrame(out))
    hist = pl.concat(parts) if parts else pl.DataFrame({"_row": []})
    return (
        df.join(hist, on="_row", how="left")
        .sort("_row")
        .drop("_row", "_t")
        .with_columns(pl.col(n).fill_nan(None) for n in ratios)
    )


def shrink(value: str, weight: str, prior: pl.Expr | float, k: float) -> pl.Expr:
    """Empirical-Bayes style blend: (value * weight + prior * k) / (weight + k).
    `k` is how much evidence the prior is worth. No history -> the prior."""
    p = prior if isinstance(prior, pl.Expr) else pl.lit(prior)
    v = pl.col(value).fill_null(0.0)
    w = pl.col(weight).fill_null(0.0)
    return (v * w + p * k) / (w + k)
