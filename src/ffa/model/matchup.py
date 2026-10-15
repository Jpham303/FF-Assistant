"""Matchup ratings: where a defense gives up production, relative to expectation.

Betting lines already price how good a defense is overall (that flows through implied
totals into team volume and touchdowns), so these ratings do not re-penalize "a good
defense". They capture how production is distributed against it:

    tgt_{pos}    targets to that position vs what those players usually draw (funnels)
    catch_{pos}  catches per target vs those players' usual catch rate
    ypr_{pos}    yards per catch vs those players' usual yards per catch
    ypc_{pos}    yards per carry vs those players' usual yards per carry

Each rating is actual / expected over the defense's earlier games, where "expected" comes
from each opposing player's own as-of estimates, so ratings are opponent-adjusted by
construction. Ratings shrink toward 1.0 (no effect); a rating that does not improve
validation predictions ends up with a huge k and effectively switches itself off.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ffa.model.asof import Decay, decayed, shrink

RECEIVING = ["WR", "TE", "RB"]
RUSHING = ["RB", "QB"]

# rating -> (position, actual column, expected column, player estimate it multiplies,
#            numerator and denominator used to score it on validation)
RATINGS: dict[str, dict[str, str]] = {}
for _p in RECEIVING:
    RATINGS[f"tgt_{_p}"] = {"pos": _p, "act": "targets", "exp": "exp_targets"}
    RATINGS[f"catch_{_p}"] = {"pos": _p, "act": "rec", "exp": "exp_rec"}
    RATINGS[f"ypr_{_p}"] = {"pos": _p, "act": "rec_yd", "exp": "exp_rec_yd"}
for _p in RUSHING:
    RATINGS[f"ypc_{_p}"] = {"pos": _p, "act": "rush_yd", "exp": "exp_rush_yd"}

K_GRID = [10, 30, 100, 300, 1000, 3000, 1e9]  # in units of the expected quantity
DECAY_GRID = [Decay(h, g) for h in (4, 8, 16, 32) for g in (4, 12, 24)]


def expectations(est: pl.DataFrame) -> pl.DataFrame:
    """Per player-game: what the player would have produced at his own usual rates."""
    return est.with_columns(
        (pl.col("tgt_share_s") * pl.col("team_targets")).alias("exp_targets"),
        (pl.col("targets") * pl.col("catch_rate_s")).alias("exp_rec"),
        (pl.col("rec") * pl.col("ypr_s")).alias("exp_rec_yd"),
        (pl.col("rush_att") * pl.col("ypc_s")).alias("exp_rush_yd"),
    )


def defense_games(exp: pl.DataFrame) -> pl.DataFrame:
    """One row per defense per game with actual and expected totals by position.
    Only players who actually played count (zero rows for scratches would skew it)."""
    if "played" in exp.columns:
        exp = exp.filter(pl.col("played"))
    aggs = []
    for name, r in RATINGS.items():
        in_pos = pl.col("position") == r["pos"]
        ok = in_pos & pl.col(r["exp"]).is_not_null()
        aggs += [
            pl.col(r["act"]).filter(ok).sum().alias(f"{name}_act"),
            pl.col(r["exp"]).filter(ok).sum().alias(f"{name}_exp"),
        ]
    return exp.group_by("game_id", "season", "week", pl.col("opponent").alias("defense")).agg(aggs)


@dataclass
class MatchupModel:
    decay: dict[str, Decay]
    k: dict[str, float]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def ratings(self, defense_games: pl.DataFrame) -> pl.DataFrame:
        """As-of ratings for every defense-game: `{rating}` (1.0 = neutral, 1.10 = allows
        10% more than expected) and `{rating}_w` (evidence behind it)."""
        out = defense_games.select("game_id", "season", "week", "defense")
        for decay in set(self.decay.values()):
            names = [n for n, d in self.decay.items() if d == decay]
            ratios = {n: (f"{n}_act", f"{n}_exp") for n in names}
            h = decayed(defense_games, "defense", ratios, decay)  # keeps row order
            out = out.with_columns(h.select([c for n in names for c in (n, f"{n}_w")]))
        return out.with_columns(shrink(n, f"{n}_w", 1.0, self.k[n]).alias(n) for n in RATINGS)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> MatchupModel:
        d = json.loads(Path(path).read_text())
        return cls(decay={n: Decay(**v) for n, v in d.pop("decay").items()}, **d)


def _loss(rows: pl.DataFrame, r: dict[str, str], mult: np.ndarray) -> float:
    """Squared error of actual vs expected-times-rating, per unit of expectation."""
    act, exp = rows[r["act"]].to_numpy(), rows[r["exp"]].to_numpy()
    if r["act"] == "targets":  # counts: Poisson deviance
        mu = np.clip(exp * mult, 1e-6, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.mean(2 * (np.where(act > 0, act * np.log(act / mu), 0) - (act - mu))))
    # Expected yards can be ~0 or negative (backs who lose yards on catches); weight by at
    # least one unit so a near-zero expectation can't blow up the loss.
    w = np.clip(np.abs(exp), 1.0, None)
    return float(np.sum((act - exp * mult) ** 2 / w) / len(act))


def fit(exp_games: pl.DataFrame, valid: list[int]) -> MatchupModel:
    """Choose each rating's decay and k on `valid` seasons by predicting player-games from
    the defense's earlier games only. `exp_games` = expectations() over all seasons."""
    dg = defense_games(exp_games)
    va_players = exp_games.filter(pl.col("season").is_in(valid))
    model = MatchupModel({}, {})
    hists = {
        d: decayed(dg, "defense", {n: (f"{n}_act", f"{n}_exp") for n in RATINGS}, d)
        for d in DECAY_GRID
    }
    for name, r in RATINGS.items():
        rows = va_players.filter(
            (pl.col("position") == r["pos"]) & pl.col(r["exp"]).is_not_null()
            & (pl.col("played") if "played" in va_players.columns else pl.lit(True))
        ).select("game_id", "opponent", r["act"], r["exp"])  # fmt: skip
        trials = []
        for d, h in hists.items():
            rated = rows.join(
                h.select("game_id", pl.col("defense").alias("opponent"), name, f"{name}_w"),
                on=["game_id", "opponent"], how="left",
            )  # fmt: skip
            for k in K_GRID:
                mult = rated.select(shrink(name, f"{name}_w", 1.0, k))[:, 0].to_numpy()
                trials.append((_loss(rated, r, mult), d, k))
        loss, d, k = min(trials, key=lambda t: t[0])
        none = _loss(rows, r, np.ones(rows.height))
        model.decay[name], model.k[name] = d, k
        model.metrics[name] = {"valid_loss": loss, "no_matchup_loss": none, "rows": rows.height}
    return model
