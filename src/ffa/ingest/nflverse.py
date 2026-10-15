"""Loaders for nflverse release data.

Each dataset is published as one file per season (schedules: one file for all seasons).
Loaders return polars DataFrames; writing to the warehouse is a separate step.

Includes ffopportunity (ffverse), whose expected stats per player-week are the
"opportunity-only" baseline the projection model has to beat.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import httpx
import polars as pl

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
FFOPPORTUNITY = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str  # may contain {season}
    per_season: bool = True
    # Volatile data changes during a week (injury reports, depth charts), so every fetch is
    # kept as a timestamped snapshot instead of overwriting. That keeps the backtest honest.
    volatile: bool = False
    fmt: str = "parquet"  # or "csv"


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset("player_stats", f"{RELEASES}/stats_player/stats_player_week_{{season}}.parquet"),
        Dataset("team_stats", f"{RELEASES}/stats_team/stats_team_week_{{season}}.parquet"),
        Dataset("pbp", f"{RELEASES}/pbp/play_by_play_{{season}}.parquet"),
        Dataset("snap_counts", f"{RELEASES}/snap_counts/snap_counts_{{season}}.parquet"),
        Dataset("schedules", f"{RELEASES}/schedules/games.parquet", per_season=False),
        Dataset("injuries", f"{RELEASES}/injuries/injuries_{{season}}.parquet", volatile=True),
        Dataset(
            "rosters",
            f"{RELEASES}/weekly_rosters/roster_weekly_{{season}}.parquet",
            volatile=True,
        ),
        Dataset("expected_points", f"{FFOPPORTUNITY}/ep_weekly_{{season}}.csv", fmt="csv"),
    ]
}

# CSV columns that look numeric but are identifiers.
_ID_COLUMNS = {"player_id": pl.Utf8, "game_id": pl.Utf8}


class NotPublishedError(Exception):
    """The requested season file does not exist (yet)."""


def fetch(dataset: str, season: int | None = None, *, client: httpx.Client | None = None):
    """Download one dataset (one season) and return it as a DataFrame."""
    ds = DATASETS[dataset]
    url = ds.url.format(season=season) if ds.per_season else ds.url
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        resp = client.get(url)
        if resp.status_code == 404:
            raise NotPublishedError(url)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        if ds.fmt == "csv":
            df = pl.read_csv(
                buf, infer_schema_length=None, null_values=["NA", ""], schema_overrides=_ID_COLUMNS
            )
        else:
            df = pl.read_parquet(buf)
    finally:
        if own:
            client.close()
    if not ds.per_season and season is not None:
        df = df.filter(pl.col("season") == season)
    return df
