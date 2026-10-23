"""Benchmark snapshots: what Sleeper projected, captured at the same moments as ours.

Sleeper's weekly projections (what a manager sees in the app when setting a lineup) come
from an unofficial, undocumented endpoint, so every raw response is stored as fetched:
if the format changes, history can be re-parsed. Its projected stats use the same keys
as Sleeper league scoring settings, so each league's points are the dot product of the
two, exactly as Sleeper scores them.

Like our snapshots, a benchmark is scored on the last capture before the player's kickoff
and never on one taken after it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from ffa.ingest.ids import TEAM_ALIASES

DATASET = "benchmark_snapshots"
RAW = "benchmark_raw"
SOURCE = "sleeper"
URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

SCHEMA = {
    "source": pl.Utf8, "league": pl.Utf8, "key": pl.Utf8, "sleeper_id": pl.Utf8,
    "name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8, "points": pl.Float64,
    "kickoff": pl.Datetime("us", "UTC"), "taken_at": pl.Datetime("us", "UTC"),
    "content_hash": pl.Utf8,
}  # fmt: skip


def fetch(season: int, week: int, client: httpx.Client | None = None) -> list[dict]:
    """Sleeper's projections for one week, all fantasy positions (one request)."""
    params = [("season_type", "regular"), *(("position[]", p) for p in POSITIONS)]
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        r = client.get(URL.format(season=season, week=week), params=params)
        r.raise_for_status()
        data = r.json()
    finally:
        if own:
            client.close()
    if not isinstance(data, list):
        raise ValueError(f"Sleeper projections: expected a list, got {type(data).__name__}")
    return data


def league_points(stats: dict, scoring: dict) -> float:
    """Sleeper projected stats scored with Sleeper scoring settings (same keys)."""
    return float(sum(float(v) * float(scoring[k]) for k, v in (stats or {}).items()
                     if k in scoring and isinstance(v, int | float)))  # fmt: skip


def build(raw: list[dict], crosswalk: pl.DataFrame, kickoffs: pl.DataFrame,
          leagues: dict[str, dict], taken_at: datetime) -> pl.DataFrame:  # fmt: skip
    """One row per (league, player) Sleeper projected. Players are keyed like ours:
    gsis_id via the Sleeper crosswalk, defenses as DST_<team>. Players that can't be
    matched are dropped (never guessed)."""
    xw = dict(crosswalk.select("sleeper_id", "gsis_id").drop_nulls().iter_rows())
    ko = dict(kickoffs.select("team", "kickoff").iter_rows()) if kickoffs.height else {}
    rows = []
    for e in raw:
        stats = e.get("stats") or {}
        if not stats:
            continue
        player = e.get("player") or {}
        pos = player.get("position") or ""
        team = e.get("team") or player.get("team")
        team = TEAM_ALIASES.get(team, team) if team else None
        sid = str(e.get("player_id") or "")
        if pos == "DEF":
            key = f"DST_{TEAM_ALIASES.get(sid, sid)}" if sid else None
        else:
            key = xw.get(sid)
        if key is None or pos not in POSITIONS:
            continue
        name = " ".join(x for x in (player.get("first_name"), player.get("last_name")) if x)
        for league, scoring in leagues.items():
            rows.append({
                "source": SOURCE, "league": league, "key": key, "sleeper_id": sid,
                "name": name or None, "position": pos, "team": team,
                "points": round(league_points(stats, scoring), 3),
                "kickoff": ko.get(team), "taken_at": taken_at.astimezone(UTC),
            })  # fmt: skip
    df = pl.DataFrame(rows, schema={k: v for k, v in SCHEMA.items() if k != "content_hash"})
    df = df.unique(["league", "key"], keep="first")
    return df.with_columns(pl.lit(content_hash(df)).alias("content_hash"))


def content_hash(df: pl.DataFrame) -> str:
    canon = df.select("league", "key", pl.col("points").round(3)).sort("league", "key")
    return hashlib.sha256(canon.write_csv().encode()).hexdigest()[:16]


def _dir(root: str | Path, dataset: str, season: int, week: int) -> Path:
    return Path(root) / dataset / f"season={season}" / f"week={week:02d}"


def last_hash(root: str | Path, season: int, week: int) -> str | None:
    files = sorted(_dir(root, DATASET, season, week).glob("snap_*.parquet"))
    if not files:
        return None
    h = pl.read_parquet(files[-1], columns=["content_hash"])
    return h["content_hash"][0] if h.height else None


def take(root: str | Path, season: int, week: int, crosswalk: pl.DataFrame,
         kickoffs: pl.DataFrame, leagues: dict[str, dict], now: datetime | None = None,
         raw: list[dict] | None = None) -> tuple[Path | None, int]:  # fmt: skip
    """Capture Sleeper's projections now. Returns (file, rows); file is None when they
    are unchanged since the last capture (that one stands for them). Write-once."""
    now = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    raw = fetch(season, week) if raw is None else raw
    df = build(raw, crosswalk, kickoffs, leagues, now)
    if df.height == 0:
        raise ValueError(f"Sleeper returned no matchable projections for {season} week {week}")
    if df["content_hash"][0] == last_hash(root, season, week):
        return None, df.height
    stamp = f"{now:%Y%m%dT%H%M%SZ}"
    raw_dir = _dir(root, RAW, season, week)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{SOURCE}_{stamp}.json.gz"
    out_dir = _dir(root, DATASET, season, week)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snap_{stamp}.parquet"
    if path.exists() or raw_path.exists():
        raise FileExistsError(f"benchmark snapshot already exists: {path}")
    raw_path.write_bytes(gzip.compress(json.dumps(raw).encode()))
    df.with_columns(pl.lit(now).alias("_ingested_at")).write_parquet(path)
    return path, df.height


def all_benchmarks(root: str | Path, season: int, week: int) -> pl.DataFrame:
    files = sorted(_dir(root, DATASET, season, week).glob("snap_*.parquet"))
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
