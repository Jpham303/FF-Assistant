"""Assemble one week's simulation inputs from the fitted models.

Everything is computed from games strictly before (season, week): team predictions from
the betting lines and as-of tendencies, player estimates for every active-roster player,
the opposing defense's matchup ratings, availability from the injury report, and team
pass-rush / takeaway tendencies for the defense model.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffa.ingest import status as status_src
from ffa.ingest.ids import TEAM_ALIASES
from ffa.model.asof import Decay, decayed, shrink
from ffa.model.matchup import RATINGS, MatchupModel, defense_games, expectations
from ffa.model.player import PlayerModel
from ffa.model.sim_params import STARTER_SNAP_PCT, SimParams, is_rest
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


RECENT_TEAM_GAMES = 2  # an absent player still holds a role if he played this recently


def _roster(rosters: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """This week's roster: active players, plus inactive / injured-reserve players (who
    will not play) so that a recently absent starter's role can pass to his teammates."""
    r = rosters.filter((pl.col("season") == season) & (pl.col("week") <= week))
    latest = r.filter(pl.col("week") == r["week"].max()) if r.height else r
    return (
        latest.filter(
            pl.col("status").is_in(["ACT", "INA", "RES"])
            & pl.col("gsis_id").is_not_null()
            & pl.col("position").replace({"FB": "RB"}).is_in(POSITIONS)
        )
        .select(
            pl.col("gsis_id").alias("key"),
            pl.col("full_name").alias("name"),
            pl.col("team").replace(TEAM_ALIASES),
            pl.col("position").replace({"FB": "RB"}),
            pl.col("status").alias("roster_status"),
        )
        .unique("key")
    )


def _recently_played(history: pl.DataFrame, roster: pl.DataFrame) -> pl.DataFrame:
    """Keys of inactive/IR players who played for their team in one of its last
    RECENT_TEAM_GAMES games. Older absences are already reflected in teammates' shares."""
    absent = roster.filter(pl.col("roster_status") != "ACT").select("key", "team")
    if absent.height == 0 or history.height == 0:
        return absent.select("key").head(0)
    games = (
        history.select("team", "season", "week").unique()
        .sort("season", "week", descending=True)
        .with_columns(pl.int_range(pl.len()).over("team").alias("rank"))
        .filter(pl.col("rank") < RECENT_TEAM_GAMES)
    )  # fmt: skip
    played = history.filter(pl.col("played")) if "played" in history.columns else history
    return (
        played.join(games, on=["team", "season", "week"], how="semi")
        .select("key", "team").unique()
        .join(absent, on=["key", "team"], how="semi").select("key")
    )  # fmt: skip


def build(
    *,
    season: int,
    week: int,
    team_games: pl.DataFrame,
    player_games: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    models: Models,
    statuses: pl.DataFrame | None = None,
) -> WeekInputs:
    """`statuses`: game statuses from a second source (see ffa.ingest.status), combined
    with the nflverse report, the more severe winning."""
    teams = _teams(team_games, models, season, week)
    roster = _roster(rosters, season, week).join(
        teams.select("game_id", "team", "opponent"), on="team", how="inner"
    )

    history = _before(player_games, season, week)
    recent = _recently_played(history, roster)
    roster = roster.filter(
        (pl.col("roster_status") == "ACT") | pl.col("key").is_in(recent["key"].implode())
    )
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
    status = _statuses(injuries, statuses, season, week)
    players = (
        players.drop("injury_status", "injury", strict=False)
        .join(role, on="key", how="left").join(status, on="key", how="left")
        .join(_final_reports(injuries, season, week), on="team", how="left")
        .with_columns(pl.col("prev_out").fill_null(False),
                      pl.col("final_report").fill_null(False))
    )  # fmt: skip
    p_play = [
        models.sim.play_probability(
            r["injury_status"],
            r["practice"],
            (r["season_snap_pct"] or 0) >= STARTER_SNAP_PCT,
            prev_out=r["prev_out"],
            final=r["final_report"],
        )  # fmt: skip
        for r in players.select(
            "injury_status", "practice", "season_snap_pct", "prev_out", "final_report"
        ).iter_rows(named=True)
    ]
    # Inactive / injured reserve this week: will not play, but his role is inherited.
    p_play = [
        0.0 if rs != "ACT" else p for p, rs in zip(p_play, players["roster_status"], strict=True)
    ]
    keep = [
        "key", "name", "team", "opponent", "game_id", "position", "roster_status",
        "injury_status",
        "practice", "injury", "injury_source", "injury_updated_at", "injury_fetched_at",
        "prev_out", "final_report", "season_snap_pct", "m_tgt", "m_catch", "m_ypr", "m_ypc",
        *[f"{n}_s" for n in ("tgt_share", "rz_tgt_share", "car_share", "rz_car_share",
                             "db_share", "rec_td_share", "rush_td_share", "catch_rate",
                             "ypr", "ypc", "adot", "int_rate", "scramble_rate", "fum_rate")],
        "tgt_share_w", "car_share_w", "db_share_w",
    ]  # fmt: skip
    players = players.with_columns(pl.Series("p_play", p_play)).select(*keep, "p_play")
    return WeekInputs(season, week, teams, players)


def _statuses(injuries: pl.DataFrame, statuses: pl.DataFrame | None, season: int,
              week: int) -> pl.DataFrame:  # fmt: skip
    """Per player this week: designation, practice, injury, which source said so and when,
    and whether he was Out the week before."""
    cur = injuries.filter((pl.col("season") == season) & (pl.col("week") == week))
    fetched = (
        pl.col("_ingested_at").dt.convert_time_zone("UTC")
        if "_ingested_at" in cur.columns else pl.lit(None, pl.Datetime("us", "UTC"))
    )  # fmt: skip
    reason = pl.coalesce(
        pl.col("practice_primary_injury") if "practice_primary_injury" in cur.columns
        else pl.lit(None, pl.Utf8), pl.lit(None, pl.Utf8),
    )  # fmt: skip
    nfl = cur.select(
        pl.col("gsis_id").alias("key"),
        pl.col("report_status").replace({"Probable": None, "Note": None}).alias("status"),
        # A scheduled rest day is not an injury signal.
        pl.when(reason.map_elements(is_rest, return_dtype=pl.Boolean).fill_null(False))
        .then(pl.lit("Full Participation in Practice"))
        .otherwise(pl.col("practice_status")).alias("practice"),
        pl.coalesce("report_primary_injury", reason).alias("injury"),
        pl.lit("nflverse").alias("source"),
        pl.lit(None, pl.Datetime("us", "UTC")).alias("updated_at"),
        fetched.alias("fetched_at"),
    ).filter(pl.col("key").is_not_null())  # fmt: skip
    nfl = nfl.unique("key", keep="last")
    combined = status_src.combine(nfl, statuses)
    prev = (
        injuries.filter(
            (pl.col("season") == season) & (pl.col("week") == week - 1)
            & (pl.col("report_status") == "Out")
        )
        .select(pl.col("gsis_id").alias("key")).unique()
        .with_columns(pl.lit(True).alias("prev_out"))
    )  # fmt: skip
    return combined.rename({
        "status": "injury_status", "source": "injury_source",
        "updated_at": "injury_updated_at", "fetched_at": "injury_fetched_at",
    }).join(prev, on="key", how="full", coalesce=True)  # fmt: skip


def _final_reports(injuries: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Whether each team's final injury report is in. Midweek reports carry practice
    participation only; game designations arrive with the final one (usually Friday), and
    in 2015-2025 all but 0.3% of final reports designated someone. Judged from the data,
    not the clock, because the feed can lag the official report."""
    cur = injuries.filter((pl.col("season") == season) & (pl.col("week") == week))
    if cur.height == 0 or "team" not in cur.columns:
        return pl.DataFrame(schema={"team": pl.Utf8, "final_report": pl.Boolean})
    return cur.group_by(pl.col("team").replace(TEAM_ALIASES)).agg(
        pl.col("report_status").is_in(["Out", "Questionable", "Doubtful"]).any()
        .alias("final_report")
    )  # fmt: skip
