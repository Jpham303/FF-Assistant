"""Frozen projection snapshots: what the model said, before kickoff, never overwritten.

One row per (snapshot, league scoring, player). A league's scoring rules are stored in
the row, so a later change to the league's settings doesn't rewrite history. The
snapshot used to score a player is the last one taken before his game's kickoff.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ffa.ingest.ids import TEAM_ALIASES
from ffa.scoring import score
from ffa.scoring.rules import DEFAULT_SLEEPER_SCORING, from_sleeper
from ffa.service import ifactive
from ffa.warehouse import duck

DATASET = "projection_snapshots"
ET = ZoneInfo("America/New_York")  # nflverse schedule times are Eastern
STATS = ["pass_yd", "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td", "rec_tgt",
         "rec", "rec_yd", "rec_td"]  # fmt: skip


def model_version(models_dir: str | Path) -> str:
    """Content hash of the fitted model files: changes exactly when the model does."""
    h = hashlib.sha256()
    for f in sorted(Path(models_dir).glob("*.json")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def git_sha() -> str:
    return os.environ.get("FFA_GIT_SHA", "unknown")


def kickoffs(con, season: int, week: int) -> pl.DataFrame:
    """team -> game_id and kickoff (UTC) from the schedule (both teams of each game)."""
    try:  # this week's file only: other weeks' formats can't interfere
        t = duck.week(con, "schedules", season, week)
    except duck.MissingPartition:
        return pl.DataFrame(schema={"team": pl.Utf8, "game_id": pl.Utf8,
                                    "kickoff": pl.Datetime("us", "UTC")})  # fmt: skip
    rows = con.sql(f"select game_id, gameday, gametime, home_team, away_team from {t}").pl()
    out = []
    for g, day, t, home, away in rows.iter_rows():
        if day and t:
            ko = datetime.fromisoformat(f"{day}T{t}").replace(tzinfo=ET).astimezone(UTC)
            out += [{"team": home, "game_id": g, "kickoff": ko},
                    {"team": away, "game_id": g, "kickoff": ko}]  # fmt: skip
    schema = {"team": pl.Utf8, "game_id": pl.Utf8, "kickoff": pl.Datetime("us", "UTC")}
    df = pl.DataFrame(out, schema=schema)
    return df.with_columns(pl.col("team").replace(TEAM_ALIASES))


def build(con, season: int, week: int, leagues: dict[str, dict], *, taken_at: datetime,
          version: str, sha: str) -> pl.DataFrame:  # fmt: skip
    """Snapshot rows for this week's stored projections, one block per league scoring.
    `leagues`: league id -> Sleeper-style scoring settings (incl. 'default')."""
    sims = con.sql(f"select * from {duck.week(con, 'projection_sims', season, week)}").pl()
    # The simulator sets `active` only for skill players; kickers and defenses it
    # simulates are playing.
    sims = sims.with_columns(
        pl.col("active").fill_null(True)
        if "active" in sims.columns
        else pl.lit(True).alias("active")
    )
    inputs = con.sql(f"select * from {duck.week(con, 'projection_inputs', season, week)}").pl()
    # Inputs written by run_week always carry these (checked by INPUT_SCHEMA); take what
    # exists so a partial week still snapshots.
    wanted = ["injury_status", "practice", "injury_source", "injury_fetched_at",
              "prev_status", "report_published", "roster_status"]  # fmt: skip
    meta = inputs.select(
        "key", pl.col("p_play").alias("play_probability"),
        *[c for c in wanted if c in inputs.columns],
    )  # fmt: skip
    ko = kickoffs(con, season, week)
    try:
        forced = con.sql(
            f"select * from {duck.week(con, 'projection_sims_forced', season, week)}"
        ).pl()
    except duck.MissingPartition:
        forced = None
    play = dict(meta.select("key", "play_probability").iter_rows())
    snapshot_id = str(uuid.uuid4())
    blocks = []
    for league, scoring in leagues.items():
        rules = from_sleeper(scoring)
        pts = score(sims, rules).select("key", "sim", "fpts", "active")
        fpts = None
        if forced is not None and forced.height:
            fpts = score(forced, rules).select(pl.col("forced_key").alias("key"), "sim", "fpts")
        summ = ifactive.summarize(pts, play, fpts).rename({
            "p10_if_active": "p10", "p25_if_active": "p25", "median_if_active": "p50",
            "p75_if_active": "p75", "p90_if_active": "p90", "quantiles_if_active": "quantiles",
        }).drop("play_probability")  # fmt: skip
        stats = (
            sims.filter(pl.col("active"))
            .group_by("key")
            .agg(*[pl.col(c).mean().alias(f"exp_{c}") for c in STATS if c in sims.columns])
        )
        names = sims.select("key", "name", "team", "position").unique("key")
        block = (
            names.join(summ, on="key").join(stats, on="key", how="left")
            .join(meta, on="key", how="left")
            .with_columns(pl.col("play_probability").fill_null(1.0))
            .join(ko, on="team", how="left")
            .with_columns(
                pl.lit(league).alias("league"), pl.lit(json.dumps(scoring, sort_keys=True))
                .alias("scoring"),
            )
        )  # fmt: skip
        blocks.append(block)
    df = pl.concat(blocks, how="diagonal_relaxed")
    return df.with_columns(
        pl.lit(content_hash(df)).alias("content_hash"),
        pl.lit(snapshot_id).alias("snapshot_id"),
        pl.lit(taken_at.astimezone(UTC)).alias("taken_at"),
        pl.lit(version).alias("model_version"),
        pl.lit(sha).alias("git_sha"),
        pl.lit(season).cast(pl.Int32).alias("season"),
        pl.lit(week).cast(pl.Int32).alias("week"),
    )


def content_hash(df: pl.DataFrame) -> str:
    """What the snapshot says (projections, availability, scoring), ignoring when it was
    taken: equal hashes mean an identical forecast."""
    cols = ["league", "scoring", "key", "expected_points", "proj_if_active", "p10", "p90",
            "play_probability", "injury_status", "kickoff"]  # fmt: skip
    canon = df.select([c for c in cols if c in df.columns]).sort("league", "key")
    canon = canon.with_columns(pl.col(pl.Float64, pl.Float32).round(4))
    return hashlib.sha256(canon.write_csv().encode()).hexdigest()[:16]


def last_hash(root: str | Path, season: int, week: int) -> str | None:
    d = Path(root) / DATASET / f"season={season}" / f"week={week:02d}"
    files = sorted(d.glob("snap_*.parquet"))
    if not files:
        return None
    h = pl.read_parquet(files[-1], columns=["content_hash"])
    return h["content_hash"][0] if h.height else None


def write(df: pl.DataFrame, root: str | Path, taken_at: datetime) -> Path:
    """Write once: refuses to overwrite an existing snapshot file."""
    ts = taken_at.astimezone(UTC).replace(microsecond=0)
    season, week = int(df["season"][0]), int(df["week"][0])
    out = Path(root) / DATASET / f"season={season}" / f"week={week:02d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"snap_{ts:%Y%m%dT%H%M%SZ}.parquet"
    if path.exists():
        raise FileExistsError(f"snapshot already exists: {path}")
    df.drop("season", "week").with_columns(pl.lit(ts).alias("_ingested_at")).write_parquet(path)
    return path


def leagues_scoring(db) -> dict[str, dict]:
    """Scoring to snapshot: every synced Sleeper league, plus standard half-PPR."""
    out = {"default": dict(DEFAULT_SLEEPER_SCORING)}
    if db is None:
        return out
    from sqlalchemy import select

    from ffa.db.models import LeagueRow

    for row in db.scalars(select(LeagueRow)):
        if row.platform == "sleeper" and row.scoring:
            out[row.id] = dict(row.scoring)
    return out


def take(con, root: str | Path, models_dir: str | Path, season: int, week: int,
         leagues: dict[str, dict], now: datetime | None = None,
         ) -> tuple[Path | None, int]:  # fmt: skip
    """Freeze the stored projections for `week` now. Returns (file, rows); file is None
    when the forecast is identical to the last snapshot (the earlier one stands for it)."""
    now = now or datetime.now(UTC)
    df = build(con, season, week, leagues, taken_at=now, version=model_version(models_dir),
               sha=git_sha())  # fmt: skip
    if df.height and df["content_hash"][0] == last_hash(root, season, week):
        return None, df.height
    return write(df, root, now), df.height


def all_snapshots(root: str | Path, season: int, week: int) -> pl.DataFrame:
    """Every snapshot of a week, read from that week's files only."""
    d = Path(root) / DATASET / f"season={season}" / f"week={week:02d}"
    files = sorted(d.glob("snap_*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").with_columns(
        pl.lit(season).cast(pl.Int32).alias("season"), pl.lit(week).cast(pl.Int32).alias("week")
    )


def scoring_snapshot(snaps: pl.DataFrame) -> pl.DataFrame:
    """Per (league, player): the last snapshot taken before his kickoff. Players whose
    kickoff is unknown, or with no snapshot before it, are left out (never scored on a
    snapshot taken after the game started)."""
    if snaps.height == 0:
        return snaps
    return (
        snaps.filter(pl.col("kickoff").is_not_null() & (pl.col("taken_at") < pl.col("kickoff")))
        .sort("taken_at")
        .group_by("league", "key", maintain_order=True)
        .last()
    )
