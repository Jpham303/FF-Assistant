"""Calibration constants for the simulator, fitted on training seasons only.

availability   P(plays | injury status, final practice, role), from snap counts
dispersion     yards per catch (gamma shape), yards per carry (sd), and how much
               more targets/carries vary than a fixed share implies (Dirichlet
               concentration), all measured against as-of estimates so they include
               real-world estimation error
team rates     targets per dropback, pass attempts per target, league sack / INT /
               fumble rates, extra-point and two-point rates
kicking        field-goal distance distribution and make probability by distance
D/ST           league rates for return TDs, special-teams TDs, blocks, safeties
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

STARTER_SNAP_PCT = 0.30  # "role" threshold for availability


# Chosen on 2024 (2023 as a check) from {0, 0.5, 0.75, 1}: removes most of the ~1 point
# under-projection of players taking over a ruled-out starter's role and lowers RMSE; see
# docs/evaluation_2025.md "Changes after the holdout" and issue #1.
INHERIT_KEEP = 0.5

PENDING_PRACTICE = frozenset({
    "Did Not Participate In Practice", "Limited Participation in Practice",
})  # fmt: skip


CARRY_STATUSES = ("Out", "Doubtful", "Questionable")
# Used only if a fitted parameter file predates the carry-forward table: conservative
# values, replaced by fitted ones at the next sim_params run.
CARRY_FALLBACK = {"Out": 0.35, "Doubtful": 0.5, "Questionable": 0.85}


def is_rest(reason: str | None) -> bool:
    """Veterans' scheduled rest days appear as DNP; they say nothing about injury."""
    return bool(reason) and "rest" in reason.lower()


@dataclass
class SimParams:
    p_play: dict[str, float] = field(default_factory=dict)  # "status|practice|role"
    # Share of an absent player's work that stays with his position group (0 = spread
    # across all active teammates by their shares). Chosen by backtest; see issue #1.
    inherit_keep: float = INHERIT_KEEP
    rec_yd_shape: float = 1.0
    rush_yd_sd: float = 5.0
    target_concentration: float = 50.0
    carry_concentration: float = 50.0
    targets_per_dropback: float = 0.85
    pass_att_per_target: float = 1.05
    sack_rate: float = 0.065
    int_rate: float = 0.023
    fumble_rate: float = 0.006
    xp_rate: float = 0.94
    two_point_rate: float = 0.05
    td_dispersion: float = 0.71
    fg_dispersion: float = 0.84
    # Field goals keep getting longer (avg 37.7 yds in 2016, 41.6 in 2026) while make rates
    # hold, so kicking is fitted per season and a week uses the latest completed seasons.
    fg_distance: dict[str, dict[str, float]] = field(default_factory=dict)  # season -> pmf
    fg_make: dict[str, list[float]] = field(default_factory=dict)  # season -> [a, b] logit
    dst: dict[str, float] = field(default_factory=dict)
    # Quantiles of the share of team dropbacks the expected starter (the dressed QB with
    # the highest as-of dropback share) actually took. Mostly ~1; the low tail is in-game
    # injuries and benchings. Empty = parameter file predates it: QB_STARTER_FALLBACK.
    qb_starter_share: list[float] = field(default_factory=list)

    def starter_share_quantiles(self) -> np.ndarray:
        return np.asarray(self.qb_starter_share or QB_STARTER_FALLBACK, dtype=float)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> SimParams:
        return cls(**json.loads(Path(path).read_text()))

    def kicking(self, season: int) -> tuple[dict[str, float], list[float]]:
        """Distance distribution and make curve from the latest season before `season`."""
        prior = [int(s) for s in self.fg_distance if int(s) < season]
        key = str(max(prior)) if prior else min(self.fg_distance, key=int)
        return self.fg_distance[key], self.fg_make[key]

    def play_probability(
        self,
        status: str | None,
        practice: str | None,
        role: bool,
        prev_out: bool = False,
        final: bool = True,
        prev_status: str | None = None,
        published: bool = True,
    ) -> float:
        """P(plays).

        - A game designation this week decides it.
        - Before the team's first report of the week (`published=False`, usually until
          Wednesday), last week's designation is carried forward: a player Out last week
          is not assumed healthy just because no new report exists yet.
        - Before the final report (`final=False`), a missed or limited practice counts, as
          does having been Out the week before. On a final report, no designation means
          cleared to play.
        """
        r = "role" if role else "depth"
        if status == "Out":
            return 0.0
        if status in (None, "", "Probable", "Note"):
            if not published and prev_status in CARRY_STATUSES:
                for key in (f"carry|{prev_status}|{r}", f"carry|{prev_status}|any"):
                    if key in self.p_play:
                        return self.p_play[key]
                return CARRY_FALLBACK[prev_status]
            # On a final report no designation means cleared to play.
            if final or practice not in PENDING_PRACTICE:
                return 1.0
            prev = "prev_out" if prev_out else "prev_ok"
            for key in (f"pending|{practice}|{r}|{prev}", f"pending|{practice}|{r}|any",
                        f"pending|{practice}|any|any"):  # fmt: skip
                if key in self.p_play:
                    return self.p_play[key]
            return 1.0
        for key in (f"{status}|{practice}|{r}", f"{status}|any|{r}", f"{status}|any|any"):
            if key in self.p_play:
                return self.p_play[key]
        return 0.5


QB_QUANTILES = 41  # 0, 2.5%, ..., 100%
# Fitted on 2016-2023 (4,373 team games): under 50% of dropbacks in 2.8% of games, under
# 10% in 0.5%. Used only by parameter files fitted before this existed.
QB_STARTER_FALLBACK = (
    0.0, 0.4514, 0.6991, 0.875, 0.9348, 0.963, 0.9713, 0.975, 0.9787) + (1.0,) * 32


def _qb_starter_share(est_games: pl.DataFrame) -> list[float]:
    """Quantiles of the expected starter's realized share of team dropbacks, over every
    team game: the QB who played with the highest as-of `db_share_s` is the one the
    simulator would start. Only QBs who played (snaps > 0) count: the availability model
    defines playing the same way, so a game he sat out is already in P(plays)."""
    need = {"position", "season", "week", "team", "dropbacks", "team_dropbacks", "db_share_s",
            "snaps"}
    if not need <= set(est_games.columns):
        return []
    s = (
        est_games.filter(
            (pl.col("position") == "QB") & (pl.col("team_dropbacks") > 0)
            & pl.col("db_share_s").is_not_null() & (pl.col("snaps").fill_null(0) > 0)
        )  # fmt: skip
        .sort("db_share_s", descending=True)
        .unique(["season", "week", "team"], keep="first")
        .select((pl.col("dropbacks").fill_null(0) / pl.col("team_dropbacks")).clip(0, 1))
        .to_series()
        .to_numpy()
    )
    if len(s) < 200:
        return []
    return [round(float(q), 4) for q in np.quantile(s, np.linspace(0, 1, QB_QUANTILES))]


def _availability(injuries: pl.DataFrame, snaps: pl.DataFrame, player_games: pl.DataFrame,
                  team_games: pl.DataFrame | None = None):  # fmt: skip
    """P(plays) by status, final practice and role. Role = averaged at least 30% of
    offensive snaps in the player's earlier games that season."""
    role = (
        player_games.sort("season", "week")
        .with_columns(
            # Running average through each game; the as-of join below then takes the
            # value from the latest game strictly before the injury week.
            (pl.col("snap_pct").fill_null(0).cum_sum() / pl.col("snap_pct").cum_count())
            .over("key", "season")
            .alias("prior_snap_pct")
        )  # fmt: skip
        .select("key", "season", "week", "prior_snap_pct")
    )
    inj = (
        injuries.filter(pl.col("report_status").is_in(["Questionable", "Doubtful"]))
        .select(
            pl.col("gsis_id").alias("key"),
            "season",
            "week",
            pl.col("report_status").alias("status"),
            pl.col("practice_status").fill_null("any").alias("practice"),
        )  # fmt: skip
        .unique(["key", "season", "week"])
    )
    # A player's role going into a game = his prior average as of his latest earlier game.
    inj = inj.sort("week").join_asof(
        role.sort("week"), on="week", by=["key", "season"], strategy="backward",
        allow_exact_matches=False, check_sortedness=False,
    ).with_columns(
        pl.when(pl.col("prior_snap_pct") >= STARTER_SNAP_PCT).then(pl.lit("role"))
        .otherwise(pl.lit("depth")).alias("role")
    )  # fmt: skip
    played = (
        snaps.filter(pl.col("snaps") > 0)
        .select("key", "season", "week")
        .with_columns(pl.lit(True).alias("played"))
    )
    inj = inj.join(played, on=["key", "season", "week"], how="left").with_columns(
        pl.col("played").fill_null(False)
    )
    # "Pending": what a missed or limited practice says before a game status exists.
    # Fitted on every report row with that practice level, whatever status followed, and
    # split by whether the player was Out the week before.
    out_weeks = (
        injuries.filter(pl.col("report_status") == "Out")
        .select(pl.col("gsis_id").alias("key"), "season", (pl.col("week") + 1).alias("week"))
        .unique()
        .with_columns(pl.lit("prev_out").alias("prev"))
    )
    pend = (
        injuries.filter(
            pl.col("practice_status").is_in(list(PENDING_PRACTICE))
            & ~pl.col("practice_primary_injury")
            .fill_null("")
            .str.to_lowercase()
            .str.contains("rest")
        )
        .select(
            pl.col("gsis_id").alias("key"),
            "season",
            "week",
            pl.col("practice_status").alias("practice"),
        )  # fmt: skip
        .unique(["key", "season", "week"])
        .join(out_weeks, on=["key", "season", "week"], how="left")
        .with_columns(pl.col("prev").fill_null("prev_ok"))
    )
    pend = pend.sort("week").join_asof(
        role.sort("week"), on="week", by=["key", "season"], strategy="backward",
        allow_exact_matches=False, check_sortedness=False,
    ).with_columns(
        pl.when(pl.col("prior_snap_pct") >= STARTER_SNAP_PCT).then(pl.lit("role"))
        .otherwise(pl.lit("depth")).alias("role")
    ).join(played, on=["key", "season", "week"], how="left").with_columns(
        pl.col("played").fill_null(False)
    )  # fmt: skip
    out: dict[str, float] = {}
    for keys in (["status", "practice", "role"], ["status", "role"], ["status"]):
        g = inj.group_by(keys).agg(pl.len().alias("n"), pl.col("played").mean().alias("p"))
        for row in g.filter(pl.col("n") >= 30).iter_rows(named=True):
            key = "|".join([
                row["status"],
                row.get("practice", "any") if "practice" in keys else "any",
                row.get("role", "any") if "role" in keys else "any",
            ])  # fmt: skip
            out[key] = round(float(row["p"]), 4)
    # "Carry": a player designated Out / Doubtful / Questionable in his team's latest
    # report, before the team's next report exists. Outcome: played in the team's next
    # game (byes skipped, so a bye week isn't counted as a missed game).
    if team_games is not None and team_games.height:
        tg = team_games.select("team", "season", "week").unique().sort("week")
        prior = (
            injuries.filter(pl.col("report_status").is_in(list(CARRY_STATUSES)))
            .select(
                pl.col("gsis_id").alias("key"),
                "team",
                "season",
                "week",
                pl.col("report_status").alias("prev"),
            )  # fmt: skip
            .unique(["key", "season", "week"])
            .with_columns((pl.col("week") + 1).alias("from_week"))
            .sort("from_week")
        )
        nxt = (
            prior.join_asof(
                tg.rename({"week": "next_week"}),
                left_on="from_week",
                right_on="next_week",
                by=["team", "season"],
                strategy="forward",
                check_sortedness=False,
            )
            .filter(pl.col("next_week").is_not_null())
            .drop("week")
            .rename({"next_week": "week"})
        )
        carry = nxt.sort("week").join_asof(
            role.sort("week"), on="week", by=["key", "season"], strategy="backward",
            allow_exact_matches=False, check_sortedness=False,
        ).with_columns(
            pl.when(pl.col("prior_snap_pct") >= STARTER_SNAP_PCT).then(pl.lit("role"))
            .otherwise(pl.lit("depth")).alias("role")
        ).join(played, on=["key", "season", "week"], how="left").with_columns(
            pl.col("played").fill_null(False)
        )  # fmt: skip
        for keys in (["prev", "role"], ["prev"]):
            g = carry.group_by(keys).agg(pl.len().alias("n"), pl.col("played").mean().alias("p"))
            for row in g.filter(pl.col("n") >= 30).iter_rows(named=True):
                out["|".join(["carry", row["prev"], row.get("role", "any")])] = round(
                    float(row["p"]), 4
                )
    for keys in (["practice", "role", "prev"], ["practice", "role"], ["practice"]):
        g = pend.group_by(keys).agg(pl.len().alias("n"), pl.col("played").mean().alias("p"))
        for row in g.filter(pl.col("n") >= 30).iter_rows(named=True):
            key = "|".join(["pending", row["practice"], row.get("role", "any"),
                            row.get("prev", "any")])  # fmt: skip
            out[key] = round(float(row["p"]), 4)
    return out


def _concentration(rows: pl.DataFrame, y: str, p: str, n: str) -> float:
    """Dirichlet-multinomial concentration from the excess variance of counts around
    n * p: Var = n p (1-p) (n + c) / (1 + c)."""
    d = rows.filter(pl.col(p).is_not_null() & (pl.col(n) > 0)).with_columns(
        pl.col(p).clip(0.001, 0.999)
    )
    resid = ((d[y] - d[n] * d[p]) ** 2).sum()
    binom = (d[n] * d[p] * (1 - d[p])).sum()
    ratio = float(resid / binom)
    nbar = float(d[n].mean())
    return float(max((nbar - ratio) / max(ratio - 1, 1e-6), 1.0))


def fit(
    est_games: pl.DataFrame,
    team_games: pl.DataFrame,
    injuries: pl.DataFrame,
    snaps: pl.DataFrame,
    kicks: pl.DataFrame,
    dst_lines: pl.DataFrame,
    td_dispersion: float,
    fg_dispersion: float,
) -> SimParams:
    """All inputs restricted to training seasons by the caller.

    est_games: player_games with as-of estimates (PlayerModel.estimates(history)).
    kicks: play-by-play field-goal and extra-point attempts.
    dst_lines: ffa.scoring.defense.dst_stat_lines output.
    """
    sp = SimParams(td_dispersion=td_dispersion, fg_dispersion=fg_dispersion)
    sp.p_play = _availability(injuries, snaps, est_games, team_games)
    sp.qb_starter_share = _qb_starter_share(est_games)

    r = est_games.filter((pl.col("rec") > 0) & pl.col("ypr_s").is_not_null())
    sp.rec_yd_shape = float(
        (r["rec"] * r["ypr_s"] ** 2).sum() / ((r["rec_yd"] - r["rec"] * r["ypr_s"]) ** 2).sum()
    )
    c = est_games.filter((pl.col("rush_att") > 0) & pl.col("ypc_s").is_not_null())
    sp.rush_yd_sd = float(
        np.sqrt(((c["rush_yd"] - c["rush_att"] * c["ypc_s"]) ** 2).sum() / c["rush_att"].sum())
    )
    sp.target_concentration = _concentration(est_games, "targets", "tgt_share_s", "team_targets")
    sp.carry_concentration = _concentration(
        est_games, "designed_carries", "car_share_s", "team_designed_runs"
    )

    t = team_games.filter(pl.col("plays").is_not_null())
    cols = ["targets", "dropbacks", "pass_att", "sacks_taken", "ints_thrown", "fumbles_lost",
            "plays", "xp_att", "xp_made"]  # fmt: skip
    tot = t.select(pl.col(cols).sum()).row(0, named=True)
    sp.targets_per_dropback = tot["targets"] / tot["dropbacks"]
    sp.pass_att_per_target = tot["pass_att"] / tot["targets"]
    sp.sack_rate = tot["sacks_taken"] / tot["dropbacks"]
    sp.int_rate = tot["ints_thrown"] / tot["pass_att"]
    sp.fumble_rate = tot["fumbles_lost"] / tot["plays"]
    sp.xp_rate = tot["xp_made"] / tot["xp_att"]
    tds = float(t.select((pl.col("pass_td") + pl.col("rush_td")).sum()).item())
    sp.two_point_rate = float(max(0.0, 1 - tot["xp_att"] / tds))

    fg = kicks.filter(pl.col("field_goal_attempt") == 1).drop_nulls("kick_distance")
    for season in sorted(fg["season"].unique().to_list()):
        # Each season's entry pools it with the two before it (~3,000 kicks).
        recent = fg.filter(pl.col("season").is_between(season - 2, season))
        dist = recent["kick_distance"].cast(pl.Int64)
        vc = dist.value_counts(normalize=True).sort("kick_distance")
        sp.fg_distance[str(season)] = {str(d): float(p) for d, p in vc.iter_rows()}
        made = (recent["field_goal_result"] == "made").cast(pl.Float64).to_numpy()
        res = sm.GLM(made, sm.add_constant(dist.to_numpy().astype(float)),
                     family=sm.families.Binomial()).fit()  # fmt: skip
        sp.fg_make[str(season)] = [float(res.params[0]), float(res.params[1])]

    g = dst_lines.height
    dcols = ["def_td", "def_st_td", "blk_kick", "safe", "def_st_fum_rec", "def_st_ff", "ff",
             "int", "fum_rec"]  # fmt: skip
    tot_d = dst_lines.select(pl.col(dcols).sum()).row(0, named=True)
    sp.dst = {
        "def_td_per_takeaway": tot_d["def_td"] / (tot_d["int"] + tot_d["fum_rec"]),
        "def_st_td_per_game": tot_d["def_st_td"] / g,
        "blk_kick_per_game": tot_d["blk_kick"] / g,
        "safe_per_game": tot_d["safe"] / g,
        "def_st_fum_rec_per_game": tot_d["def_st_fum_rec"] / g,
        "def_st_ff_per_game": tot_d["def_st_ff"] / g,
        "ff_per_fum_rec": tot_d["ff"] / max(tot_d["fum_rec"], 1),
    }
    return sp
