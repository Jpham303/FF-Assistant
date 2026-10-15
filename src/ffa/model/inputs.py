"""Assemble one week's simulation inputs from the fitted models.

Everything is computed from games strictly before (season, week): team predictions from
the betting lines and as-of tendencies, player estimates for every active-roster player,
the opposing defense's matchup ratings, availability from the injury report, and team
pass-rush / takeaway tendencies for the defense model.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffa.ingest.ids import TEAM_ALIASES
from ffa.model.asof import Decay, decayed, shrink
from ffa.model.matchup import RATINGS, MatchupModel, defense_games, expectations
from ffa.model.player import PlayerModel
from ffa.model.sim_params import STARTER_SNAP_PCT, SimParams
from ffa.model.team import TeamModel

POSITIONS = ["QB", "RB", "WR", "TE", "K"]
TEAM_RATES = {  # offense allowed / defense forced, per opportunity
    "off_sack": ("sacks_taken", "dropbacks"),
    "off_int": ("ints_thrown", "pass_att"),
    "off_fum": ("fumbles_lost", "plays"),
    "def_sack": ("opp_sacks_taken", "opp_dropbacks"),
    "def_int": ("opp_ints_thrown", "opp_pass_att"),
    "def_fum": ("opp_fumbles_lost", "opp_plays"),
}
TEAM_RATE_K = {"sack": 200.0, "int": 400.0, "fum": 800.0}  # opportunities of evidence
TEAM_RATE_DECAY = Decay(half_life=16, offseason_gap=4)


@dataclass
class Models:
    team: TeamModel
    player: PlayerModel
    matchup: MatchupModel
    sim: SimParams


@dataclass
class WeekInputs:
    season: int
    week: int
    teams: pl.DataFrame  # one row per team playing this week
    players: pl.DataFrame  # one row per active-roster player on those teams


def _before(df: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return df.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def _teams(team_games: pl.DataFrame, models: Models, season: int, week: int) -> pl.DataFrame:
    tg = team_games.filter(pl.col("season") <= season).with_columns(pl.col("fg_att").fill_null(0))
    # Outcomes from this week on must not inform this week's predictions.
    later = ~((pl.col("season") < season) | (pl.col("week") < week))
    outcome_cols = [c for c in tg.columns if c not in (
        "game_id", "season", "week", "game_type", "team", "opponent", "is_home",
        "spread_for", "total_line", "implied", "opp_implied", "roof",
    )]  # fmt: skip
    tg = tg.with_columns(
        pl.when(later).then(None).otherwise(pl.col(c)).alias(c) for c in outcome_cols
    )
    pred = models.team.predict(models.team.features(tg))
    sim = models.sim
    league = {"sack": sim.sack_rate, "int": sim.int_rate, "fum": sim.fumble_rate}
    rates = decayed(tg, "team", TEAM_RATES, TEAM_RATE_DECAY)
    mults = []
    for n in TEAM_RATES:
        kind = n[4:]  # "off_sack" -> "sack"
        rate = shrink(n, f"{n}_w", league[kind], TEAM_RATE_K[kind])
        mults.append((rate / league[kind]).alias(f"{n}_mult"))
    rates = rates.select("game_id", "team", *mults)
    week_rows = pred.filter((pl.col("season") == season) & (pl.col("week") == week))
    out = week_rows.join(rates, on=["game_id", "team"], how="left")
    opp = out.select(
        "game_id", pl.col("team").alias("opponent"),
        pl.col("def_sack_mult").alias("opp_def_sack_mult"),
        pl.col("def_int_mult").alias("opp_def_int_mult"),
        pl.col("def_fum_mult").alias("opp_def_fum_mult"),
    )  # fmt: skip
    return out.join(opp, on=["game_id", "opponent"], how="left").select(
        "game_id", "team", "opponent", "is_home", "spread_for", "total_line", "implied",
        "opp_implied", "mu_plays", "sd_plays", "p_pass", "mu_off_td", "p_pass_td",
        "mu_fg_att", "pace_s", "proe_s", "off_sack_mult", "off_int_mult", "off_fum_mult",
        "def_sack_mult", "def_int_mult", "def_fum_mult", "opp_def_sack_mult",
        "opp_def_int_mult", "opp_def_fum_mult",
    )  # fmt: skip


def _roster(rosters: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    r = rosters.filter((pl.col("season") == season) & (pl.col("week") <= week))
    latest = r.filter(pl.col("week") == r["week"].max()) if r.height else r
    return (
        latest.filter(
            (pl.col("status") == "ACT")
            & pl.col("gsis_id").is_not_null()
            & pl.col("position").replace({"FB": "RB"}).is_in(POSITIONS)
        )
        .select(
            pl.col("gsis_id").alias("key"),
            pl.col("full_name").alias("name"),
            pl.col("team").replace(TEAM_ALIASES),
            pl.col("position").replace({"FB": "RB"}),
        )
        .unique("key")
    )


def build(
    *,
    season: int,
    week: int,
    team_games: pl.DataFrame,
    player_games: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    models: Models,
) -> WeekInputs:
    teams = _teams(team_games, models, season, week)
    roster = _roster(rosters, season, week).join(
        teams.select("game_id", "team", "opponent"), on="team", how="inner"
    )

    history = _before(player_games, season, week)
    to_predict = roster.with_columns(
        pl.lit(season).cast(history["season"].dtype).alias("season"),
        pl.lit(week).cast(history["week"].dtype).alias("week"),
    )
    pm = models.player
    est = pm.estimates(pm.history(pl.concat([history, to_predict], how="diagonal_relaxed")))
    est_hist = est.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    now = est.filter((pl.col("season") == season) & (pl.col("week") == week)).join(
        to_predict.select("key"), on="key", how="semi"
    )

    # Opposing defense's matchup ratings going into this week.
    dg = defense_games(expectations(est_hist))
    placeholders = teams.select(
        pl.col("game_id"), pl.lit(season).cast(dg["season"].dtype).alias("season"),
        pl.lit(week).cast(dg["week"].dtype).alias("week"), pl.col("opponent").alias("defense"),
    )  # fmt: skip
    ratings = models.matchup.ratings(pl.concat([dg, placeholders], how="diagonal_relaxed"))
    ratings = ratings.join(placeholders.select("game_id", "defense"), on=["game_id", "defense"],
                           how="semi").rename({"defense": "opponent"})  # fmt: skip

    players = now.join(ratings.select("game_id", "opponent", *RATINGS), on=["game_id", "opponent"],
                       how="left")  # fmt: skip
    pos = pl.col("position")

    def mult(prefix: str) -> pl.Expr:
        expr = pl.lit(1.0)
        for p in ("WR", "TE", "RB", "QB"):
            name = f"{prefix}_{p}"
            if name in RATINGS:
                expr = pl.when(pos == p).then(pl.col(name)).otherwise(expr)
        return expr.fill_null(1.0)

    players = players.with_columns(
        mult("tgt").alias("m_tgt"), mult("catch").alias("m_catch"),
        mult("ypr").alias("m_ypr"), mult("ypc").alias("m_ypc"),
    )  # fmt: skip

    # Availability: injury status, final practice, and role (prior snap share this season).
    role = (
        history.filter(pl.col("season") == season)
        .group_by("key")
        .agg(pl.col("snap_pct").fill_null(0).mean().alias("season_snap_pct"))
    )
    inj = (
        injuries.filter((pl.col("season") == season) & (pl.col("week") == week))
        .select(
            pl.col("gsis_id").alias("key"),
            pl.col("report_status").alias("injury_status"),
            pl.col("practice_status").alias("practice"),
            pl.col("report_primary_injury").alias("injury"),
        )
        .unique("key", keep="last")
    )
    players = (
        players.drop("injury_status", "injury", strict=False)
        .join(role, on="key", how="left").join(inj, on="key", how="left")
    )  # fmt: skip
    p_play = [
        models.sim.play_probability(
            r["injury_status"],
            r["practice"],
            (r["season_snap_pct"] or 0) >= STARTER_SNAP_PCT,
        )  # fmt: skip
        for r in players.select("injury_status", "practice", "season_snap_pct").iter_rows(
            named=True
        )
    ]
    keep = [
        "key", "name", "team", "opponent", "game_id", "position", "injury_status",
        "practice", "injury", "season_snap_pct", "m_tgt", "m_catch", "m_ypr", "m_ypc",
        *[f"{n}_s" for n in ("tgt_share", "rz_tgt_share", "car_share", "rz_car_share",
                             "db_share", "rec_td_share", "rush_td_share", "catch_rate",
                             "ypr", "ypc", "adot", "int_rate", "scramble_rate", "fum_rate")],
        "tgt_share_w", "car_share_w", "db_share_w",
    ]  # fmt: skip
    players = players.with_columns(pl.Series("p_play", p_play)).select(*keep, "p_play")
    return WeekInputs(season, week, teams, players)
