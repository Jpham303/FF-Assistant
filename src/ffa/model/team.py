"""Team volume model: how many plays, how pass-heavy, how many touchdowns and field-goal
tries a team should have in a game.

Inputs are the betting lines (which already price most of what matters) plus each team's
own tendencies and its opponent's defensive tendencies, all as of before the game and
shrunk toward the league average when the evidence is thin.

    plays            linear       spread_for, total_line, pace, opp pace allowed
    pass rate        logistic     + pass rate over expected, opp's allowed
    offensive TDs    Poisson      log implied total
    pass TD share    logistic     implied total, pass rate over expected
    FG attempts      Poisson      log implied total, team FG tendency

Coefficients are stored as plain numbers (JSON), so prediction needs no model library
and every term can be cited in an explanation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from ffa.model.asof import Decay, decayed, shrink

# Tuned on 2024 after fitting on 2016-2023 (2025 is the untouched holdout). Validation
# loss was nearly flat across the grid (half-life 4-16, offseason gap 4-12, k 4-12), so
# these matter little; the betting lines carry most of the signal.
DEFAULT_DECAY = Decay(half_life=16, offseason_gap=4)
DEFAULT_K = 12.0
TRAIN_SEASONS = list(range(2016, 2024))
VALID_SEASONS = [2024]

TENDENCIES = {
    "pace": ("plays", None),
    "pace_allowed": ("opp_plays", None),
    "proe": ("pass_rate_oe", None),
    "proe_allowed": ("opp_pass_rate_oe", None),
    "fga_rate": ("fg_att", None),
}

FEATURES = {
    "plays": ["spread_for", "total_line", "pace_s", "opp_pace_allowed_s"],
    "pass_rate": ["spread_for", "total_line", "proe_s", "opp_proe_allowed_s"],
    "off_td": ["log_implied"],
    "pass_td_share": ["implied", "proe_s"],
    "fg_att": ["log_implied", "fga_rate_s"],
}
LINES_ONLY = {
    "plays": ["spread_for", "total_line"],
    "pass_rate": ["spread_for", "total_line"],
    "off_td": ["log_implied"],
    "pass_td_share": ["implied"],
    "fg_att": ["log_implied"],
}


@dataclass
class TeamModel:
    decay: Decay
    k: dict[str, float]  # prior strength (in games) per tendency
    league: dict[str, float]  # league means the tendencies shrink toward
    coef: dict[str, dict[str, float]] = field(default_factory=dict)
    plays_sd: float = 0.0
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------ features

    def features(self, team_games: pl.DataFrame) -> pl.DataFrame:
        """Add shrunk as-of tendencies for the team and its opponent."""
        tg = team_games.with_columns(
            (pl.col("pass_td") + pl.col("rush_td")).alias("off_td"),
        )
        hist = decayed(tg, "team", TENDENCIES, self.decay)
        shr = hist.select(
            "game_id",
            "team",
            *[shrink(n, f"{n}_w", self.league[n], self.k[n]).alias(f"{n}_s") for n in TENDENCIES],
        )
        opp = shr.select(
            "game_id",
            pl.col("team").alias("opponent"),
            pl.col("pace_allowed_s").alias("opp_pace_allowed_s"),
            pl.col("proe_allowed_s").alias("opp_proe_allowed_s"),
        )
        return (
            tg.join(shr, on=["game_id", "team"], how="left")
            .join(opp, on=["game_id", "opponent"], how="left")
            .with_columns(pl.col("implied").clip(lower_bound=3).log().alias("log_implied"))
        )

    # ------------------------------------------------------------ prediction

    def _lin(self, df: pl.DataFrame, target: str) -> pl.Expr:
        c = self.coef[target]
        return pl.lit(c["const"]) + pl.sum_horizontal([pl.col(f) * c[f] for f in FEATURES[target]])

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        logistic = lambda x: 1 / (1 + (-x).exp())  # noqa: E731
        return feats.with_columns(
            self._lin(feats, "plays").alias("mu_plays"),
            pl.lit(self.plays_sd).alias("sd_plays"),
            logistic(self._lin(feats, "pass_rate")).alias("p_pass"),
            self._lin(feats, "off_td").exp().alias("mu_off_td"),
            logistic(self._lin(feats, "pass_td_share")).alias("p_pass_td"),
            self._lin(feats, "fg_att").exp().alias("mu_fg_att"),
        )

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TeamModel:
        d = json.loads(Path(path).read_text())
        return cls(decay=Decay(**d.pop("decay")), **d)


# ---------------------------------------------------------------- fitting


def _design(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return sm.add_constant(df.select(cols).to_numpy().astype(float), has_constant="add")


def _fit_one(train: pl.DataFrame, target: str, cols: list[str]):
    X = _design(train, cols)
    if target == "plays":
        return sm.OLS(train["plays"].to_numpy(), X).fit()
    if target == "pass_rate":
        y = np.column_stack([train["dropbacks"], train["plays"] - train["dropbacks"]])
        return sm.GLM(y, X, family=sm.families.Binomial()).fit()
    if target == "pass_td_share":
        y = np.column_stack([train["pass_td"], train["rush_td"]])
        return sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return sm.GLM(train[target].to_numpy(), X, family=sm.families.Poisson()).fit()


def _score(res, df: pl.DataFrame, target: str, cols: list[str]) -> float:
    """Validation loss: RMSE for plays, mean deviance otherwise (lower is better)."""
    X = _design(df, cols)
    mu = res.predict(X)
    if target == "plays":
        return float(np.sqrt(np.mean((df["plays"].to_numpy() - mu) ** 2)))
    if target in ("pass_rate", "pass_td_share"):
        n = (df["plays"] if target == "pass_rate" else df["off_td"]).to_numpy()
        k = (df["dropbacks"] if target == "pass_rate" else df["pass_td"]).to_numpy()
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        ok = n > 0
        return float(-np.sum(k[ok] * np.log(mu[ok]) + (n[ok] - k[ok]) * np.log(1 - mu[ok]))
                     / n[ok].sum())  # fmt: skip
    y = df[target].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2 * np.mean(term))


def _rows(feats: pl.DataFrame, seasons: list[int], target: str) -> pl.DataFrame:
    cols = FEATURES[target]
    df = feats.filter(
        pl.col("season").is_in(seasons)
        & (pl.col("game_type") == "REG")
        & pl.col("plays").is_not_null()
    ).drop_nulls(cols)
    if target == "pass_td_share":
        df = df.filter(pl.col("off_td") > 0)
    return df


def fit(
    team_games: pl.DataFrame,
    train: list[int],
    valid: list[int],
    decay: Decay,
    k: dict[str, float],
) -> TeamModel:
    played = team_games.filter(pl.col("season").is_in(train) & pl.col("plays").is_not_null())
    league = {
        "pace": float(played["plays"].mean()),
        "pace_allowed": float(played["plays"].mean()),
        "proe": 0.0,
        "proe_allowed": 0.0,
        "fga_rate": float(played["fg_att"].fill_null(0).mean()),
    }
    model = TeamModel(decay=decay, k=k, league=league)
    feats = model.features(team_games.with_columns(pl.col("fg_att").fill_null(0)))
    for target, cols in FEATURES.items():
        tr, va = _rows(feats, train, target), _rows(feats, valid, target)
        res = _fit_one(tr, target, cols)
        model.coef[target] = {"const": float(res.params[0])} | {
            c: float(v) for c, v in zip(cols, res.params[1:], strict=True)
        }
        base = _fit_one(tr, target, LINES_ONLY[target])
        model.metrics[target] = {
            "valid_loss": _score(res, va, target, cols),
            "lines_only_loss": _score(base, va, target, LINES_ONLY[target]),
            "train_rows": tr.height,
            "valid_rows": va.height,
        }
        if target == "plays":
            model.plays_sd = float(np.sqrt(res.scale))
        if target in ("off_td", "fg_att"):
            model.metrics[target]["dispersion"] = float(res.pearson_chi2 / res.df_resid)
    return model
