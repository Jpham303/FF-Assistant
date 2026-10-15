"""Player opportunity shares and efficiency, as of each game.

Every quantity is a pooled ratio over the player's earlier games (recency-weighted), then
shrunk toward a prior in proportion to how little evidence there is:

    shares       targets, designed carries, dropbacks, red-zone looks, TDs
    efficiency   catch rate, yards per catch / carry, depth of target, INT, scramble,
                 fumble rates

Priors carry structure rather than a flat position average:
    TD share     -> the player's own target / carry share (TDs follow volume), blended
                    with red-zone share
    catch rate   -> position line in depth of target (deep targets are caught less)
    yards/catch  -> position line in depth of target (and gain more when they are)

`k` (how many units of evidence a prior is worth) and the recency decay are fitted by
predicting each validation game from the games before it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ffa.model.asof import Decay, decayed, shrink

TRAIN_SEASONS = list(range(2016, 2024))
VALID_SEASONS = [2024]  # 2025 is the untouched holdout
DECAY_GRID = [Decay(h, g) for h in (1, 2, 4, 8, 16, 32, 64, 128) for g in (4, 12, 24, 48)]

# name -> (numerator, denominator)
RATIOS: dict[str, tuple[str, str]] = {
    "tgt_share": ("targets", "team_targets"),
    "rz_tgt_share": ("rz_targets", "team_rz_targets"),
    "car_share": ("designed_carries", "team_designed_runs"),
    "rz_car_share": ("rz_carries", "team_rz_carries"),
    "db_share": ("dropbacks", "team_dropbacks"),
    "rec_td_share": ("rec_td", "team_pass_td"),
    "rush_td_share": ("rush_td", "team_rush_td"),
    "catch_rate": ("rec", "targets"),
    "ypr": ("rec_yd", "rec"),
    "ypc": ("rush_yd", "rush_att"),
    "adot": ("air_yards", "targets"),
    "int_rate": ("pass_int", "pass_att"),
    "scramble_rate": ("scrambles", "dropbacks"),
    "fum_rate": ("fum_lost", "touches"),
}
SHARES = ["tgt_share", "rz_tgt_share", "car_share", "rz_car_share", "db_share"]
TD_SHARES = {"rec_td_share": ("tgt_share", "rz_tgt_share"),
             "rush_td_share": ("car_share", "rz_car_share")}  # fmt: skip
DEPTH_PRIORS = ["catch_rate", "ypr"]  # prior = a + b * depth of target, by position
POSITION_PRIORS = ["adot", "ypc", "int_rate", "scramble_rate", "fum_rate"]

# Positions whose rows inform / are scored for each quantity.
POSITIONS = {
    "tgt_share": ["RB", "WR", "TE"], "rz_tgt_share": ["RB", "WR", "TE"],
    "car_share": ["QB", "RB", "WR"], "rz_car_share": ["QB", "RB", "WR"],
    "db_share": ["QB"], "rec_td_share": ["RB", "WR", "TE"],
    "rush_td_share": ["QB", "RB", "WR"], "catch_rate": ["RB", "WR", "TE"],
    "ypr": ["RB", "WR", "TE"], "ypc": ["QB", "RB", "WR"], "adot": ["RB", "WR", "TE"],
    "int_rate": ["QB"], "scramble_rate": ["QB"], "fum_rate": ["QB", "RB", "WR", "TE"],
}  # fmt: skip


def prepare(player_games: pl.DataFrame) -> pl.DataFrame:
    """Add team red-zone totals and touches, which the logs don't carry directly."""
    team_rz = player_games.group_by("game_id", "team").agg(
        pl.col("rz_targets").sum().alias("team_rz_targets"),
        pl.col("rz_carries").sum().alias("team_rz_carries"),
    )
    return player_games.join(team_rz, on=["game_id", "team"], how="left").with_columns(
        (pl.col("rec") + pl.col("rush_att")).alias("touches"),
        pl.col("position").replace({"FB": "RB"}),
    )


ORDER = [*SHARES, "adot", *TD_SHARES, *DEPTH_PRIORS,
         *[n for n in POSITION_PRIORS if n != "adot"]]  # fmt: skip


@dataclass
class PlayerModel:
    decay: dict[str, Decay]  # per quantity: roles and talent fade at different rates
    k: dict[str, float]
    td_rz_weight: dict[str, float]  # weight on red-zone share in the TD-share prior
    position_prior: dict[str, dict[str, float]] = field(default_factory=dict)
    depth_prior: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # [a, b]
    # Share prior for a rostered player with no games yet: what such players actually
    # get, counting the weeks they don't play (a backup QB is not a 95%-dropback starter).
    unproven_prior: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------ estimates

    def _prior(self, name: str) -> pl.Expr:
        pos = pl.col("position")
        if name in DEPTH_PRIORS:
            expr = pl.lit(None, pl.Float64)
            for p, (a, b) in self.depth_prior[name].items():
                expr = pl.when(pos == p).then(a + b * pl.col("adot_s")).otherwise(expr)
            return expr
        if name in TD_SHARES:
            share, rz = TD_SHARES[name]
            w = self.td_rz_weight[name]
            return (1 - w) * pl.col(f"{share}_s") + w * pl.col(f"{rz}_s")
        prior = pos.replace_strict(self.position_prior[name], default=None,
                                   return_dtype=pl.Float64)  # fmt: skip
        if name in SHARES and name in self.unproven_prior:
            unproven = pos.replace_strict(self.unproven_prior[name], default=0.0,
                                          return_dtype=pl.Float64)  # fmt: skip
            no_games = pl.col(f"{name}_w").fill_null(0) <= 0
            return pl.when(no_games).then(unproven).otherwise(prior)
        return prior

    def estimate(self, hist: pl.DataFrame, name: str) -> pl.DataFrame:
        return hist.with_columns(
            shrink(name, f"{name}_w", self._prior(name), self.k[name]).alias(f"{name}_s")
        )

    def estimates(self, hist: pl.DataFrame) -> pl.DataFrame:
        """Shrunk estimates `{name}_s` from the raw history columns `{name}`, `{name}_w`.
        Order matters: TD and efficiency priors use the shrunk shares and depth."""
        for name in ORDER:
            hist = self.estimate(hist, name)
        return hist

    def history(self, player_games: pl.DataFrame) -> pl.DataFrame:
        """Raw decayed history, each quantity with its own decay."""
        pg = prepare(player_games)
        out = pg
        for decay in set(self.decay.values()):
            names = {n: RATIOS[n] for n, d in self.decay.items() if d == decay}
            cols = [c for n in names for c in (n, f"{n}_w")]
            out = out.with_columns(decayed(pg, "key", names, decay).select(cols))
        return out

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PlayerModel:
        d = json.loads(Path(path).read_text())
        decay = {n: Decay(**v) for n, v in d.pop("decay").items()}
        return cls(decay=decay, **d)


# ---------------------------------------------------------------- fitting


def _pooled(df: pl.DataFrame, num: str, den: str) -> float:
    s = df.select(pl.col(num).sum(), pl.col(den).sum()).row(0)
    return float(s[0] / s[1]) if s[1] else 0.0


def _priors(train: pl.DataFrame) -> tuple[dict, dict]:
    """Position means, and depth-of-target lines for catch rate and yards per catch,
    fitted on training seasons at the player-season level (pooled, weighted)."""
    pos_prior: dict[str, dict[str, float]] = {}
    for name in [*SHARES, *POSITION_PRIORS]:
        num, den = RATIOS[name]
        pos_prior[name] = {
            p: _pooled(train.filter(pl.col("position") == p), num, den) for p in POSITIONS[name]
        }
    seasons = (
        train.filter(pl.col("position").is_in(["RB", "WR", "TE"]))
        .group_by("key", "season", "position")
        .agg(pl.col("air_yards", "targets", "rec", "rec_yd").sum())
        .filter(pl.col("targets") >= 20)
        .with_columns(
            (pl.col("air_yards") / pl.col("targets")).alias("adot"),
            (pl.col("rec") / pl.col("targets")).alias("catch_rate"),
            (pl.col("rec_yd") / pl.col("rec")).alias("ypr"),
        )
    )
    depth: dict[str, dict[str, list[float]]] = {"catch_rate": {}, "ypr": {}}
    for p in ["RB", "WR", "TE"]:
        s = seasons.filter(pl.col("position") == p)
        x = s["adot"].to_numpy()
        for name, wcol in (("catch_rate", "targets"), ("ypr", "rec")):
            w = s[wcol].to_numpy()
            b, a = np.polyfit(x, s[name].to_numpy(), 1, w=np.sqrt(w))
            depth[name][p] = [float(a), float(b)]
    return pos_prior, depth


def unproven_priors(
    rosters: pl.DataFrame, player_games: pl.DataFrame, team_games: pl.DataFrame
) -> dict[str, dict[str, float]]:
    """Average realized share of active-roster players in weeks before their first game
    (0 when they don't play). Pass training seasons only."""
    pg = prepare(player_games)
    played = pg.filter(pl.col("played")) if "played" in pg.columns else pg
    first = played.group_by("key").agg(
        (pl.col("season") * 100 + pl.col("week")).min().alias("first")
    )
    ros = (
        rosters.filter(pl.col("status") == "ACT")
        .select(
            pl.col("gsis_id").alias("key"),
            "season",
            "week",
            "team",
            pl.col("position").replace({"FB": "RB"}),
        )  # fmt: skip
        .unique(["key", "season", "week"])
        .join(first, on="key", how="left")
        .filter(
            pl.col("first").is_null() | (pl.col("first") >= pl.col("season") * 100 + pl.col("week"))
        )
    )
    team_tot = team_games.filter(pl.col("plays").is_not_null()).select(
        "season", "week", "team", "targets", "designed_runs", "dropbacks"
    )
    team_rz = pg.group_by("season", "week", "team").agg(
        pl.col("rz_targets").sum().alias("rz_t"), pl.col("rz_carries").sum().alias("rz_c")
    )
    mine = pg.select(
        "key", "season", "week",
        pl.col("targets").alias("my_t"), pl.col("designed_carries").alias("my_c"),
        pl.col("dropbacks").alias("my_d"), pl.col("rz_targets").alias("my_rzt"),
        pl.col("rz_carries").alias("my_rzc"),
    )  # fmt: skip
    d = (
        ros.join(team_tot, on=["season", "week", "team"], how="inner")
        .join(team_rz, on=["season", "week", "team"], how="left")
        .join(mine, on=["key", "season", "week"], how="left")
        .with_columns(pl.col("my_t", "my_c", "my_d", "my_rzt", "my_rzc").fill_null(0))
    )
    ratio = lambda a, b: (pl.col(a) / pl.col(b)).fill_nan(0).fill_null(0)  # noqa: E731
    g = d.group_by("position").agg(
        ratio("my_t", "targets").mean().alias("tgt_share"),
        ratio("my_rzt", "rz_t").mean().alias("rz_tgt_share"),
        ratio("my_c", "designed_runs").mean().alias("car_share"),
        ratio("my_rzc", "rz_c").mean().alias("rz_car_share"),
        ratio("my_d", "dropbacks").mean().alias("db_share"),
    )
    return {
        name: {r["position"]: float(r[name]) for r in g.iter_rows(named=True)} for name in SHARES
    }


def _loss(df: pl.DataFrame, name: str) -> float:
    """Loss for predicting this game's numerator from the estimate and actual denominator."""
    num, den = RATIOS[name]
    d = df.filter(
        pl.col("position").is_in(POSITIONS[name]) & (pl.col(den) > 0)
        & pl.col(f"{name}_s").is_not_null()
    )  # fmt: skip
    y, n = d[num].to_numpy(), d[den].to_numpy()
    p = d[f"{name}_s"].to_numpy()
    if name in ("catch_rate", "int_rate", "scramble_rate", "fum_rate"):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        y = np.clip(y, 0, n)
        return float(-np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / n.sum())
    if name in ("ypr", "ypc", "adot"):
        return float(np.sum(n * (y / n - p) ** 2) / n.sum())  # per-unit squared error
    mu = np.clip(p * n, 1e-6, None)  # shares and TD shares: Poisson deviance of counts
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
    return float(dev.mean())


K_GRID = {
    **{n: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 2, 4] for n in SHARES},  # in games of team evidence
    "rec_td_share": [2, 4, 8, 16, 32, 64, 128], "rush_td_share": [2, 4, 8, 16, 32, 64, 128],
    "catch_rate": [25, 50, 100, 200], "ypr": [50, 100, 200, 400, 800],
    "ypc": [6, 12, 25, 50, 100], "adot": [2.5, 5, 10, 25],
    "int_rate": [200, 400, 800, 1600, 3200], "scramble_rate": [12, 25, 50, 100, 200, 400],
    "fum_rate": [100, 200, 400, 800],
}  # fmt: skip


def fit(
    player_games: pl.DataFrame,
    train: list[int],
    valid: list[int],
    decays: list[Decay],
) -> PlayerModel:
    """For each quantity choose its decay, k (and TD red-zone weight) by validation loss.
    Priors come from `train` only; `valid` seasons are scored from earlier games only."""
    pg = prepare(player_games)
    pos_prior, depth = _priors(pg.filter(pl.col("season").is_in(train)))
    # Share k is in games of evidence, scaled by a typical team denominator.
    typical = {n: float(pg[RATIOS[n][1]].mean()) for n in [*SHARES, *TD_SHARES]}
    hists = {d: decayed(pg, "key", RATIOS, d).filter(pl.col("season").is_in(valid))
             for d in decays}  # fmt: skip
    model = PlayerModel({}, {}, {n: 0.0 for n in TD_SHARES}, pos_prior, depth)
    cur = next(iter(hists.values())).select(pl.exclude([c for n in RATIOS for c in (n, f"{n}_w")]))
    for name in ORDER:
        scale = typical.get(name, 1.0)
        rz_grid = [0.0, 0.25, 0.5, 0.75] if name in TD_SHARES else [None]
        trials = []
        for d, h in hists.items():
            frame = cur.with_columns(h.select(name, f"{name}_w"))
            for k in K_GRID[name]:
                for w in rz_grid:
                    model.k[name] = k * scale
                    if w is not None:
                        model.td_rz_weight[name] = w
                    trials.append((_loss(model.estimate(frame, name), name), d, k, w))
        loss, d, k, w = min(trials, key=lambda t: t[0])
        model.decay[name], model.k[name] = d, k * scale
        if w is not None:
            model.td_rz_weight[name] = w
        cur = model.estimate(cur.with_columns(hists[d].select(name, f"{name}_w")), name)
        # Reference: the same history with (almost) no shrinkage.
        k_keep, model.k[name] = model.k[name], 1e-6
        raw = _loss(model.estimate(cur, name), name)
        model.k[name] = k_keep
        grid = K_GRID[name]
        model.metrics[name] = {
            "valid_loss": loss,
            "no_shrink_loss": raw,
            "k_at_grid_edge": float(k in (grid[0], grid[-1])),
            "decay_at_grid_edge": float(
                d.half_life in (min(x.half_life for x in decays), max(x.half_life for x in decays))
            ),
        }
    return model
