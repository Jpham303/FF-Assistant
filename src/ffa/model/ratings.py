"""Team ratings from betting lines, for weeks whose lines aren't posted yet.

The betting market prices each team's offense and defense; every posted line is a
measurement of them. Each team-game's implied total is modelled as

    implied(team vs opp) = mu + home * is_home + off[team] + def[opp]

(`def` = points the defense allows above average), fitted by weighted ridge regression on
the lines posted so far: this season's, weighted toward the latest, plus last season's at
a discount. Applied to the future schedule, the ratings give each future game an implied
total for both teams, hence a spread and a total, which is what the team-volume model
needs. Ratings move only when new lines are posted; they don't anticipate injuries.

Hyperparameters were chosen on 2024 (see docs/evaluation_2025.md, Rest of season).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

DECAY = 0.6  # weight per week back, this season
PRIOR_SEASON = 0.0  # last season adds nothing once this season has 3+ weeks of lines
RIDGE = 0.3  # shrinkage of team effects toward average, in team-game rows


@dataclass
class Ratings:
    mu: float
    home: float
    off: dict[str, float]
    defense: dict[str, float]

    def implied(self, team: str, opp: str, is_home: bool) -> float:
        return (self.mu + self.home * float(bool(is_home)) + self.off.get(team, 0.0)
                + self.defense.get(opp, 0.0))  # fmt: skip

    def expr(self, team: str, opp: str, is_home: str) -> pl.Expr:
        """The implied total as a column expression (missing home flag = neutral site)."""
        return (
            pl.lit(self.mu) + self.home * pl.col(is_home).cast(pl.Float64).fill_null(0)
            + pl.col(team).replace_strict(self.off, default=0.0, return_dtype=pl.Float64)
            + pl.col(opp).replace_strict(self.defense, default=0.0, return_dtype=pl.Float64)
        )  # fmt: skip


def fit(team_games: pl.DataFrame, season: int, week: int, *, decay: float = DECAY,
        prior_season: float = PRIOR_SEASON, ridge: float = RIDGE) -> Ratings:  # fmt: skip
    """Ratings from lines posted for games up to and including `week` of `season` (the
    current week's lines are posted before its games), plus last season's."""
    tg = team_games.filter(
        (pl.col("game_type") == "REG") & pl.col("implied").is_not_null()
        & (((pl.col("season") == season) & (pl.col("week") <= week))
           | (pl.col("season") == season - 1))
    ).select("season", "week", "team", "opponent", "is_home", "implied")  # fmt: skip
    if tg.height == 0:
        raise ValueError(f"No posted lines to rate teams on before {season} week {week}")
    teams = sorted(set(tg["team"].to_list()) | set(tg["opponent"].to_list()))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = tg.height, len(teams)
    X = np.zeros((n, 2 + 2 * k))
    X[:, 0] = 1.0
    X[:, 1] = tg["is_home"].cast(pl.Float64).to_numpy()
    for r, (t, o) in enumerate(zip(tg["team"].to_list(), tg["opponent"].to_list(), strict=True)):
        X[r, 2 + ix[t]] = 1.0
        X[r, 2 + k + ix[o]] = 1.0
    cur = tg["season"].to_numpy() == season
    wk = tg["week"].to_numpy().astype(float)
    w = np.where(cur, decay ** (week - wk), prior_season * decay ** (18 - wk))
    y = tg["implied"].to_numpy()
    sw = np.sqrt(w)
    pen = np.zeros(2 + 2 * k)
    pen[2:] = ridge
    A = np.vstack([X * sw[:, None], np.diag(np.sqrt(pen))])
    b = np.concatenate([y * sw, np.zeros(2 + 2 * k)])
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return Ratings(
        mu=float(beta[0]), home=float(beta[1]),
        off={t: float(beta[2 + i]) for t, i in ix.items()},
        defense={t: float(beta[2 + k + i]) for t, i in ix.items()},
    )  # fmt: skip


def fill_lines(team_games: pl.DataFrame, ratings: Ratings, season: int,
               from_week: int) -> pl.DataFrame:  # fmt: skip
    """`team_games` with missing lines for `season` games from `from_week` on filled from
    the ratings (spread and total from the two teams' implied totals, so both rows of a
    game agree). Lines already posted are kept. Filled rows carry
    `lines_from_ratings = True`."""
    fill = (
        (pl.col("season") == season) & (pl.col("week") >= from_week)
        & (pl.col("game_type") == "REG") & pl.col("implied").is_null()
    )  # fmt: skip
    tg = team_games.with_columns((~pl.col("is_home").fill_null(True)).alias("_away"))
    neutral = pl.col("is_home").is_null()
    imp = ratings.expr("team", "opponent", "is_home")
    opp = ratings.expr("opponent", "team", "_away") - pl.when(neutral).then(ratings.home).otherwise(
        0
    )
    out = tg.with_columns(
        pl.when(fill).then(imp).otherwise(pl.col("implied")).alias("implied"),
        pl.when(fill).then(opp).otherwise(pl.col("opp_implied")).alias("opp_implied"),
        fill.alias("lines_from_ratings"),
    )
    return out.with_columns(
        # `fill` can't be re-tested here (implied is filled now): use the flag.
        pl.when(pl.col("lines_from_ratings")).then(pl.col("implied") - pl.col("opp_implied"))
        .otherwise(pl.col("spread_for")).alias("spread_for"),
        pl.when(pl.col("lines_from_ratings")).then(pl.col("implied") + pl.col("opp_implied"))
        .otherwise(pl.col("total_line")).alias("total_line"),
    ).drop("_away")  # fmt: skip
