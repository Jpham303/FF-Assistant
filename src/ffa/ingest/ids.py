"""Player ID crosswalk: nflverse gsis_id <-> Sleeper and Yahoo IDs.

The canonical player key everywhere in the warehouse is nflverse's `gsis_id`. Platform IDs
are mapped onto it from several sources, in priority order:

    sleeper_id: Sleeper player DB  > nflverse weekly rosters > DynastyProcess
    yahoo_id:   nflverse weekly rosters > Sleeper player DB > DynastyProcess

Sources disagree occasionally; the higher-priority one wins and the disagreement is
reported. Players still missing an ID (often recent rookies on Yahoo) are matched at
import time by normalized name + position, then team, via `Resolver`.

Team defenses have no gsis_id; they use the key `DST_<TEAM>` with nflverse team codes.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

import httpx
import polars as pl

DP_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K"]
DEFENSE_POSITIONS = {"DEF", "DST", "D/ST"}

# Every abbreviation seen across nflverse, DynastyProcess, Sleeper and Yahoo -> nflverse.
TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GBP": "GB", "GNB": "GB", "HST": "HOU",
    "JAC": "JAX", "KCC": "KC", "KAN": "KC", "LAR": "LA", "STL": "LA", "LVR": "LV",
    "OAK": "LV", "NEP": "NE", "NWE": "NE", "NOP": "NO", "NOR": "NO", "SDC": "LAC",
    "SD": "LAC", "SFO": "SF", "TBB": "TB", "TAM": "TB", "WSH": "WAS",
}  # fmt: skip

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_team(team: str | None) -> str | None:
    if not team:
        return None
    t = team.strip().upper()
    return TEAM_ALIASES.get(t, t)


def normalize_name(name: str | None) -> str:
    """'D.K. Metcalf Jr.' -> 'dkmetcalf'. Accents, punctuation and suffixes removed."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.'\-]", "", n)
    n = _SUFFIX.sub("", n)
    return re.sub(r"[^a-z]", "", n)


def dst_key(team: str) -> str:
    return f"DST_{normalize_team(team)}"


# ---------------------------------------------------------------- source loaders


def _get(url: str, client: httpx.Client | None) -> bytes:
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        r = client.get(url)
        r.raise_for_status()
        return r.content
    finally:
        if own:
            client.close()


def load_dynastyprocess(client: httpx.Client | None = None) -> pl.DataFrame:
    raw = pl.read_csv(
        io.BytesIO(_get(DP_URL, client)), infer_schema_length=0, null_values=["NA", ""]
    )
    return raw.select("gsis_id", "sleeper_id", "yahoo_id", "name", "position", "team")


def load_nflverse_players(client: httpx.Client | None = None) -> pl.DataFrame:
    raw = pl.read_parquet(io.BytesIO(_get(PLAYERS_URL, client)))
    return raw.select(
        "gsis_id",
        pl.col("display_name").alias("name"),
        "position",
        pl.col("latest_team").alias("team"),
        "last_season",
    )


def load_sleeper_players(client: httpx.Client | None = None) -> pl.DataFrame:
    """Sleeper's full player DB (~5 MB). Sleeper asks that it be fetched at most daily."""
    data = httpx.Response(200, content=_get(SLEEPER_PLAYERS_URL, client)).json()
    rows = [
        {
            "sleeper_id": str(pid),
            "gsis_id": (p.get("gsis_id") or "").strip() or None,
            "yahoo_id": str(p["yahoo_id"]) if p.get("yahoo_id") else None,
            "name": p.get("full_name"),
            "position": p.get("position"),
            "team": p.get("team"),
        }
        for pid, p in data.items()
    ]
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in rows[0]} if rows else None)


def roster_ids(rosters: pl.DataFrame) -> pl.DataFrame:
    """Latest sleeper_id / yahoo_id per gsis_id from nflverse weekly rosters."""
    return (
        rosters.filter(pl.col("gsis_id").is_not_null())
        .sort(["season", "week"])
        .group_by("gsis_id")
        .agg(
            pl.col("sleeper_id").drop_nulls().last().cast(pl.Utf8),
            pl.col("yahoo_id").drop_nulls().last().cast(pl.Utf8),
        )
    )


# ---------------------------------------------------------------- crosswalk


def _pick(frame: pl.DataFrame, col: str, order: list[str]) -> pl.DataFrame:
    """Coalesce `col` across sources in priority order; record winner and any conflict."""
    cols = [f"{col}__{s}" for s in order]
    present = [c for c in cols if c in frame.columns]
    winner = pl.coalesce(present)
    source = pl.coalesce(
        [pl.when(pl.col(c).is_not_null()).then(pl.lit(c.split("__")[1])) for c in present]
    )
    distinct = pl.concat_list(present).list.drop_nulls().list.unique().list.len()
    return frame.with_columns(
        winner.alias(col),
        source.alias(f"{col}_source"),
        (distinct > 1).alias(f"{col}_conflict"),
    )


def build_crosswalk(
    players: pl.DataFrame,
    rosters: pl.DataFrame,
    dp: pl.DataFrame,
    sleeper: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per gsis_id with platform IDs, the source of each, and conflict flags."""

    def ids(df: pl.DataFrame, source: str) -> pl.DataFrame:
        df = df.filter(pl.col("gsis_id").is_not_null()).unique("gsis_id", keep="first")
        return df.select(
            "gsis_id",
            pl.col("sleeper_id").cast(pl.Utf8).alias(f"sleeper_id__{source}"),
            pl.col("yahoo_id").cast(pl.Utf8).alias(f"yahoo_id__{source}"),
        )

    base = players.unique("gsis_id", keep="first").with_columns(
        pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8)
    )
    frame = base.join(ids(roster_ids(rosters), "rosters"), on="gsis_id", how="left")
    frame = frame.join(ids(dp, "dp"), on="gsis_id", how="left")
    if sleeper is not None:
        frame = frame.join(ids(sleeper, "sleeper"), on="gsis_id", how="left")

    frame = _pick(frame, "sleeper_id", ["sleeper", "rosters", "dp"])
    frame = _pick(frame, "yahoo_id", ["rosters", "sleeper", "dp"])
    keep = ["gsis_id", "name", "position", "team", "last_season"]
    out = frame.select(
        *[c for c in keep if c in frame.columns],
        "sleeper_id", "sleeper_id_source", "sleeper_id_conflict",
        "yahoo_id", "yahoo_id_source", "yahoo_id_conflict",
    )  # fmt: skip
    return out.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_key")
    )


# ---------------------------------------------------------------- resolver


@dataclass(frozen=True)
class Match:
    key: str | None  # gsis_id, DST_<TEAM>, or None
    method: str  # "id", "name", "name+team", "dst", "unmatched", "ambiguous"


class Resolver:
    """Maps a platform player to the canonical key. Exact ID first, then name matching."""

    def __init__(self, crosswalk: pl.DataFrame):
        self._by_id = {
            "sleeper": dict(crosswalk.select("sleeper_id", "gsis_id").drop_nulls().iter_rows()),
            "yahoo": dict(crosswalk.select("yahoo_id", "gsis_id").drop_nulls().iter_rows()),
        }
        self._by_name: dict[tuple[str, str], list[tuple[str, str | None, int]]] = {}
        for gsis, key, pos, team, last in crosswalk.select(
            "gsis_id", "name_key", "position", "team", "last_season"
        ).iter_rows():
            self._by_name.setdefault((key, pos), []).append((gsis, team, last or 0))

    def resolve(
        self,
        platform: str,
        platform_id: str | None,
        *,
        name: str | None = None,
        position: str | None = None,
        team: str | None = None,
    ) -> Match:
        pos = (position or "").upper()
        if pos in DEFENSE_POSITIONS:
            return Match(dst_key(team or platform_id or ""), "dst")
        if platform_id and (gsis := self._by_id[platform].get(str(platform_id))):
            return Match(gsis, "id")
        if not name or not pos:
            return Match(None, "unmatched")

        candidates = self._by_name.get((normalize_name(name), pos), [])
        if not candidates:
            return Match(None, "unmatched")
        if len(candidates) == 1:
            return Match(candidates[0][0], "name")
        t = normalize_team(team)
        on_team = [c for c in candidates if t and c[1] == t]
        if len(on_team) == 1:
            return Match(on_team[0][0], "name+team")
        # Same name, same position, no team signal: only the most recent player is plausible
        # if every other candidate is long retired.
        recent = sorted(candidates, key=lambda c: c[2], reverse=True)
        if recent[0][2] - recent[1][2] >= 3:
            return Match(recent[0][0], "name")
        return Match(None, "ambiguous")
