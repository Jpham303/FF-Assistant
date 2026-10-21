"""Dagster definitions: nflverse ingestion assets and schedules.

No `from __future__ import annotations` here: Dagster inspects the real type of `context`.
"""

from pathlib import Path

import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetSelection,
    DefaultScheduleStatus,
    Definitions,
    MaterializeResult,
    RunRequest,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    define_asset_job,
    schedule,
)

from ffa import features
from ffa.config import settings
from ffa.db.session import session as db_session
from ffa.features.games import load_pbp as load_game_pbp
from ffa.ingest import ids, nflverse
from ffa.leagues import sync
from ffa.leagues.sleeper import SleeperAdapter
from ffa.model import matchup as matchup_mod
from ffa.model import player as player_model_mod
from ffa.model import sim_params as sim_params_mod
from ffa.model import team as team_model
from ffa.scoring.defense import dst_stat_lines
from ffa.scoring.defense import load_pbp as load_dst_pbp
from ffa.service import projections as projections_mod
from ffa.warehouse import duck, store

FIRST_SEASON = 2015
CURRENT = str(settings.current_season)
season_partitions = StaticPartitionsDefinition(
    [str(s) for s in range(FIRST_SEASON, settings.current_season + 1)]
)


def _make_asset(dataset: str):
    ds = nflverse.DATASETS[dataset]

    @asset(
        name=dataset,
        partitions_def=season_partitions,
        group_name="nflverse",
        description=f"nflverse {dataset}, one partition per season.",
    )
    def _ingest(context: AssetExecutionContext) -> MaterializeResult:
        season = int(context.partition_key)
        try:
            df = nflverse.fetch(dataset, season)
        except nflverse.NotPublishedError:
            context.log.warning(f"{dataset} {season}: not published yet")
            return MaterializeResult(metadata={"rows": 0, "partitions": 0})
        paths = store.write(df, settings.warehouse_dir, dataset, volatile=ds.volatile)
        weeks = sorted({int(p.parent.name.split("=")[1]) for p in paths})
        return MaterializeResult(
            metadata={"rows": df.height, "partitions": len(paths), "weeks": str(weeks)}
        )

    return _ingest


ingest_assets = [_make_asset(name) for name in nflverse.DATASETS]

nflverse_job = define_asset_job(
    "nflverse_ingest",
    selection=AssetSelection.groups("nflverse"),
)
injuries_job = define_asset_job(
    "injuries_refresh",
    selection=AssetSelection.assets("injuries", "rosters"),
)


@schedule(
    job=nflverse_job,
    cron_schedule="15 2 * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def nightly_current_season(context: ScheduleEvaluationContext):
    """All datasets for the current season, after nflverse's overnight builds."""
    return RunRequest(partition_key=CURRENT, run_key=f"nightly-{context.scheduled_execution_time}")


@schedule(
    job=injuries_job,
    cron_schedule="5 * * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def hourly_injuries(context: ScheduleEvaluationContext):
    """Injury reports and depth charts change through the week; snapshot them hourly."""
    return RunRequest(partition_key=CURRENT, run_key=f"hourly-{context.scheduled_execution_time}")


def _fetch_sleeper_status(log) -> dict:
    """Fetch Sleeper's player DB (once: Sleeper asks for at most one fetch a day) and store
    IDs/names plus this week's statuses. Returns run metadata."""
    from datetime import UTC, datetime

    from ffa.ingest import status as status_src
    from ffa.service.projections import upcoming_week

    now = datetime.now(UTC)
    players = ids.load_sleeper_players()
    store.write_dim(players, settings.warehouse_dir, "sleeper_players")
    con = duck.connect(settings.warehouse_dir)
    week = upcoming_week(con, settings.current_season)
    meta = {"players": players.height, "fetched_at": now.isoformat()}
    if week is not None:
        try:
            xw = con.sql("select sleeper_id, gsis_id from player_ids").pl()
        except Exception:
            xw = pl.DataFrame(schema={"sleeper_id": pl.Utf8, "gsis_id": pl.Utf8})
        st = status_src.sleeper_status(players, xw, now).with_columns(
            pl.lit(settings.current_season).cast(pl.Int32).alias("season"),
            pl.lit(week).cast(pl.Int32).alias("week"),
        )
        store.write(st, settings.warehouse_dir, "sleeper_status", volatile=True, ingested_at=now)
        meta |= {"week": week, "with_status": st.height,
                 "out": int((st["status"] == "Out").sum())}  # fmt: skip
    log.info(f"Sleeper player DB fetched: {meta}")
    return meta


def _sleeper_age_hours() -> float | None:
    """Hours since the Sleeper player DB was last fetched, or None if never."""
    from datetime import UTC, datetime

    f = Path(settings.warehouse_dir) / "dims" / "sleeper_players.parquet"
    if not f.exists():
        return None
    return (datetime.now(UTC).timestamp() - f.stat().st_mtime) / 3600


@asset(group_name="status")
def sleeper_players(context: AssetExecutionContext) -> MaterializeResult:
    """Sleeper's player DB, fetched once a day (Sleeper asks for no more): IDs and names
    for the crosswalk, and each player's current game status and practice participation,
    a fresher second source than nflverse's injury file."""
    return MaterializeResult(metadata=_fetch_sleeper_status(context.log))


status_job = define_asset_job(
    "status_refresh", selection=AssetSelection.assets("sleeper_players", "weekly_projections")
)


# Once a day: after practice reports on weekdays; Sunday after inactives for the noon games.
@schedule(
    job=status_job,
    cron_schedule="5 17 * * 1-6",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def daily_status(context: ScheduleEvaluationContext):
    return RunRequest(run_key=f"status-{context.scheduled_execution_time}")


@schedule(
    job=status_job,
    cron_schedule="45 10 * * 0",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def sunday_status(context: ScheduleEvaluationContext):
    return RunRequest(run_key=f"status-{context.scheduled_execution_time}")


@asset(group_name="dims", deps=["rosters", "sleeper_players"])
def player_ids(context: AssetExecutionContext) -> MaterializeResult:
    """Crosswalk from nflverse gsis_id to Sleeper and Yahoo IDs, with source and conflicts."""
    rosters = (
        duck.connect(settings.warehouse_dir)
        .sql("select gsis_id, sleeper_id, yahoo_id, season, week from rosters")
        .pl()
    )
    # The Sleeper DB is fetched once a day by `sleeper_players` (Sleeper's request); reuse it.
    try:
        sleeper = duck.connect(settings.warehouse_dir).sql("select * from sleeper_players").pl()
    except Exception:  # not fetched yet: the other sources still give a usable crosswalk
        context.log.warning("Sleeper player DB not fetched yet; crosswalk built without it")
        sleeper = None
    xw = ids.build_crosswalk(
        ids.load_nflverse_players(), rosters, ids.load_dynastyprocess(), sleeper
    )
    store.write_dim(xw, settings.warehouse_dir, "player_ids")

    active = xw.filter(
        pl.col("position").is_in(ids.FANTASY_POSITIONS)
        & (pl.col("last_season") >= settings.current_season - 1)
    )
    return MaterializeResult(
        metadata={
            "players": xw.height,
            "active_fantasy_players": active.height,
            "active_with_sleeper_id": int(active["sleeper_id"].is_not_null().sum()),
            "active_with_yahoo_id": int(active["yahoo_id"].is_not_null().sum()),
            "sleeper_id_conflicts": int(xw["sleeper_id_conflict"].sum()),
            "yahoo_id_conflicts": int(xw["yahoo_id_conflict"].sum()),
            "sleeper_db_used": sleeper is not None,
        }
    )


dims_job = define_asset_job("dims_refresh", selection=AssetSelection.groups("dims"))


@schedule(
    job=dims_job,
    cron_schedule="45 2 * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def nightly_dims(context: ScheduleEvaluationContext):
    """Rebuild the ID crosswalk after the nightly roster load."""
    return RunRequest(run_key=f"dims-{context.scheduled_execution_time}")


@asset(group_name="leagues", deps=["player_ids"])
def sleeper_league(context: AssetExecutionContext) -> MaterializeResult:
    """Your Sleeper league's settings, teams and rosters, resolved to canonical player keys
    and stored in Postgres. Skipped when SLEEPER_LEAGUE_ID is not set."""
    if not settings.sleeper_league_id:
        context.log.info("SLEEPER_LEAGUE_ID not set; skipping")
        return MaterializeResult(metadata={"skipped": True})

    resolver = ids.Resolver(
        duck.connect(settings.warehouse_dir).sql("select * from player_ids").pl()
    )
    adapter = SleeperAdapter(
        me=settings.sleeper_username or None,
        players=sync.load_sleeper_players(settings.warehouse_dir),
    )
    with db_session() as db:
        rep = sync.sync(adapter, settings.sleeper_league_id, resolver, db)
    for u in rep.unmatched:
        context.log.warning(f"Unmatched player on {u.team}: {u.platform_id} {u.name or ''}")
    return MaterializeResult(
        metadata={
            "league": rep.league,
            "teams": rep.teams,
            "players": rep.players,
            "match_rate": round(rep.match_rate, 4),
            "by_method": str(rep.by_method),
            "unmatched": str([u.platform_id for u in rep.unmatched]),
            "unsupported_slots": str(rep.unsupported_slots),
        }
    )


leagues_job = define_asset_job("leagues_refresh", selection=AssetSelection.groups("leagues"))


@schedule(
    job=leagues_job,
    cron_schedule="15 * * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def hourly_leagues(context: ScheduleEvaluationContext):
    """Rosters change with waivers and trades; resync hourly."""
    return RunRequest(run_key=f"leagues-{context.scheduled_execution_time}")


# ---------------------------------------------------------------- features


def _season_frames(season: int) -> dict[str, pl.DataFrame]:
    con = duck.connect(settings.warehouse_dir)

    def q(t: str) -> pl.DataFrame:
        return con.sql(f"select * exclude (_ingested_at) from {t} where season = {season}").pl()

    return {"con": con, "q": q}


@asset(
    partitions_def=season_partitions,
    group_name="features",
    deps=["pbp", "schedules"],
    description="One row per team-game: betting lines, offense, and what the defense allowed.",
)
def team_games(context: AssetExecutionContext) -> MaterializeResult:
    season = int(context.partition_key)
    f = _season_frames(season)
    df = features.team_game_log(load_game_pbp(f["con"], season), f["q"]("schedules"))
    paths = store.write(df, settings.warehouse_dir, "team_games")
    return MaterializeResult(metadata={"rows": df.height, "partitions": len(paths)})


@asset(
    partitions_def=season_partitions,
    group_name="features",
    deps=["team_games", "player_stats", "snap_counts", "injuries", "player_ids", "rosters"],
    description="One row per player-game: opportunities, outcomes, snaps, injury status.",
)
def player_games(context: AssetExecutionContext) -> MaterializeResult:
    season = int(context.partition_key)
    f = _season_frames(season)
    con, q = f["con"], f["q"]
    df = features.player_game_log(
        load_game_pbp(con, season),
        q("player_stats"),
        q("snap_counts"),
        con.sql("select * from player_ids").pl(),
        q("injuries"),
        q("team_games"),
        q("rosters"),
    )
    paths = store.write(df, settings.warehouse_dir, "player_games")
    return MaterializeResult(metadata={"rows": df.height, "partitions": len(paths)})


features_job = define_asset_job("features_refresh", selection=AssetSelection.groups("features"))


@schedule(
    job=features_job,
    cron_schedule="0 3 * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def nightly_features(context: ScheduleEvaluationContext):
    """Rebuild the current season's game logs after the nightly data and ID loads."""
    return RunRequest(partition_key=CURRENT, run_key=f"features-{context.scheduled_execution_time}")


# ---------------------------------------------------------------- models


@asset(group_name="models", deps=["team_games"])
def team_volume_model(context: AssetExecutionContext) -> MaterializeResult:
    """Fit the team volume model (plays, pass rate, TDs, FG tries) and save its parameters."""
    tg = duck.connect(settings.warehouse_dir).sql("select * from team_games").pl()
    m = team_model.fit(
        tg.with_columns(pl.col("fg_att").fill_null(0)),
        team_model.TRAIN_SEASONS,
        team_model.VALID_SEASONS,
        team_model.DEFAULT_DECAY,
        {n: team_model.DEFAULT_K for n in team_model.TENDENCIES},
    )
    m.save(Path(settings.models_dir) / "team_volume.json")
    return MaterializeResult(
        metadata={
            f"{t}_gain_vs_lines_only": round(1 - v["valid_loss"] / v["lines_only_loss"], 4)
            for t, v in m.metrics.items()
        }
    )


@asset(group_name="models", deps=["player_games"])
def player_model(context: AssetExecutionContext) -> MaterializeResult:
    """Fit player share and efficiency shrinkage (decay and prior strength per quantity)."""
    pg = (
        duck.connect(settings.warehouse_dir)
        .sql(f"select * from player_games where season <= {max(player_model_mod.VALID_SEASONS)}")
        .pl()
    )
    m = player_model_mod.fit(
        pg, player_model_mod.TRAIN_SEASONS, player_model_mod.VALID_SEASONS,
        player_model_mod.DECAY_GRID,
    )  # fmt: skip
    con = duck.connect(settings.warehouse_dir)
    first, last = min(player_model_mod.TRAIN_SEASONS), max(player_model_mod.TRAIN_SEASONS)
    train = f"season between {first} and {last}"
    m.unproven_prior = player_model_mod.unproven_priors(
        con.sql(f"select * from rosters where {train}").pl(),
        con.sql(f"select * from player_games where {train}").pl(),
        con.sql(f"select * from team_games where {train}").pl(),
    )
    m.save(Path(settings.models_dir) / "player.json")
    return MaterializeResult(
        metadata={
            f"{n}_gain_vs_no_shrink": round(1 - v["valid_loss"] / v["no_shrink_loss"], 4)
            for n, v in m.metrics.items()
        }
    )


@asset(group_name="models", deps=["player_model"])
def matchup_model(context: AssetExecutionContext) -> MaterializeResult:
    """Fit matchup ratings (where defenses give up production vs expectation)."""
    pg = (
        duck.connect(settings.warehouse_dir)
        .sql(f"select * from player_games where season <= {max(player_model_mod.VALID_SEASONS)}")
        .pl()
    )
    pm = player_model_mod.PlayerModel.load(Path(settings.models_dir) / "player.json")
    m = matchup_mod.fit(
        matchup_mod.expectations(pm.estimates(pm.history(pg))), player_model_mod.VALID_SEASONS
    )
    m.save(Path(settings.models_dir) / "matchup.json")
    return MaterializeResult(
        metadata={
            f"{n}_gain_vs_none": round(1 - v["valid_loss"] / v["no_matchup_loss"], 4)
            for n, v in m.metrics.items()
        }
    )


@asset(group_name="models", deps=["player_model", "team_volume_model", "pbp", "injuries"])
def sim_params(context: AssetExecutionContext) -> MaterializeResult:
    """Simulator calibration: availability, dispersion, team rates, kicking, D/ST rates."""
    con = duck.connect(settings.warehouse_dir)
    seasons = player_model_mod.TRAIN_SEASONS
    train = f"season between {min(seasons)} and {max(seasons)}"
    pm = player_model_mod.PlayerModel.load(Path(settings.models_dir) / "player.json")
    tm = team_model.TeamModel.load(Path(settings.models_dir) / "team_volume.json")
    pg = con.sql(f"select * from player_games where season <= {max(seasons)}").pl()
    est = pm.estimates(pm.history(pg)).filter(pl.col("season") >= min(seasons))
    snaps = con.sql(
        f"""select x.gsis_id as key, s.season, s.week, max(s.offense_snaps) as snaps
            from snap_counts s join player_ids x on x.pfr_id = s.pfr_player_id
            where s.{train} group by 1, 2, 3"""
    ).pl()
    # Kicking is fitted per season and each prediction uses the latest completed one, so
    # every season before the current one can be included without leakage.
    kicks = con.sql(
        f"""select season, field_goal_attempt, kick_distance, field_goal_result from pbp
            where field_goal_attempt = 1 and season < {settings.current_season}"""
    ).pl()
    dst = pl.concat(
        [
            dst_stat_lines(
                load_dst_pbp(con, s), con.sql(f"select * from schedules where season={s}").pl()
            )
            for s in seasons
        ]
    )
    sp = sim_params_mod.fit(
        est,
        con.sql(f"select * from team_games where {train} and game_type = 'REG'").pl(),
        con.sql(
            f"""select * from injuries where {train} and game_type = 'REG'
                and position in ('QB', 'RB', 'WR', 'TE', 'K')"""
        ).pl(),
        snaps, kicks, dst,
        tm.metrics["off_td"]["dispersion"], tm.metrics["fg_att"]["dispersion"],
    )  # fmt: skip
    sp.save(Path(settings.models_dir) / "sim_params.json")
    return MaterializeResult(
        metadata={
            "target_concentration": round(sp.target_concentration, 2),
            "carry_concentration": round(sp.carry_concentration, 2),
            "kicking_seasons": str(sorted(sp.fg_distance)),
        }
    )


@asset(
    group_name="projections",
    deps=["sim_params", "matchup_model", "team_games", "player_games", "injuries", "rosters",
          "schedules"],
)  # fmt: skip
def weekly_projections(context: AssetExecutionContext) -> MaterializeResult:
    """Simulate the upcoming week with the latest injury report, for the API to serve."""
    con = duck.connect(settings.warehouse_dir)
    season = settings.current_season
    week = projections_mod.upcoming_week(con, season)
    if week is None:
        context.log.info(f"No upcoming week in {season}; nothing to project")
        return MaterializeResult(metadata={"week": "none"})
    # The second injury source must not depend on someone switching a schedule on: if
    # Sleeper's DB is missing or over a day old, fetch it now (still at most daily).
    age = _sleeper_age_hours()
    if age is None or age > 23:
        context.log.info(f"Sleeper statuses {'never fetched' if age is None else f'{age:.0f}h old'}"
                         "; fetching before projecting")  # fmt: skip
        try:
            _fetch_sleeper_status(context.log)
        except Exception as exc:  # projections still run on nflverse alone; say so
            context.log.warning(f"Sleeper fetch failed, projecting without it: {exc}")
        con = duck.connect(settings.warehouse_dir)
    models = projections_mod.load_models(settings.models_dir)
    r = projections_mod.run_week(con, settings.warehouse_dir, models, season, week)
    return MaterializeResult(
        metadata={"season": r.season, "week": r.week, "players": r.players, "rows": r.rows}
    )


projections_job = define_asset_job(
    "projections_refresh", selection=AssetSelection.assets("weekly_projections")
)


# After the hourly injury snapshot (:05) lands, so status changes reach the projections.
@schedule(
    job=projections_job,
    cron_schedule="25 * * * *",
    execution_timezone="America/Chicago",
    default_status=DefaultScheduleStatus.RUNNING,
)
def hourly_projections(context: ScheduleEvaluationContext):
    return RunRequest(run_key=f"projections-{context.scheduled_execution_time}")


defs = Definitions(
    assets=[
        *ingest_assets,
        sleeper_players,
        player_ids,
        sleeper_league,
        team_games,
        player_games,
        team_volume_model,
        player_model,
        matchup_model,
        sim_params,
        weekly_projections,
    ],
    jobs=[
        nflverse_job,
        injuries_job,
        dims_job,
        leagues_job,
        features_job,
        projections_job,
        status_job,
    ],
    schedules=[
        nightly_current_season,
        hourly_injuries,
        nightly_dims,
        hourly_leagues,
        nightly_features,
        hourly_projections,
        daily_status,
        sunday_status,
    ],
)
