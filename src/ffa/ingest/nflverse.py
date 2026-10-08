"""Loaders for nflverse release data.

Each dataset is published as one Parquet file per season (schedules: one file for all
seasons). Loaders return polars DataFrames; writing to the warehouse is a separate step.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import httpx
import polars as pl

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str  # may contain {season}
    per_season: bool = True
    # Volatile data changes during a week (injury reports, depth charts), so every fetch is
    # kept as a timestamped snapshot instead of overwriting. That keeps the backtest honest.
    volatile: bool = False


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset("player_stats", f"{RELEASES}/stats_player/stats_player_week_{{season}}.parquet"),
        Dataset("snap_counts", f"{RELEASES}/snap_counts/snap_counts_{{season}}.parquet"),
        Dataset("schedules", f"{RELEASES}/schedules/games.parquet", per_season=False),
        Dataset("injuries", f"{RELEASES}/injuries/injuries_{{season}}.parquet", volatile=True),
        Dataset(
            "rosters",
            f"{RELEASES}/weekly_rosters/roster_weekly_{{season}}.parquet",
            volatile=True,
        ),
    ]
}


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
        df = pl.read_parquet(io.BytesIO(resp.content))
    finally:
        if own:
            client.close()
    if not ds.per_season and season is not None:
        df = df.filter(pl.col("season") == season)
    return df
