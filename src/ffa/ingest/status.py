"""Game statuses from a second source: Sleeper's player DB.

nflverse's injury file lags the official reports, and game designations (Out, Doubtful,
Questionable) only arrive with the final report, usually Friday, which is when start/sit
questions come in. Sleeper's player DB carries each player's current designation and
practice participation, with the time its news was last updated.

Sleeper asks that the player DB be fetched at most once a day, so it is fetched once
daily: late afternoon after practice reports, and on Sunday morning after inactives.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

SOURCE = "Sleeper"

# Sleeper designations -> the nflverse vocabulary the availability model uses.
STATUS = {
    "Questionable": "Questionable", "Doubtful": "Doubtful", "Out": "Out",
    "IR": "Out", "PUP": "Out", "Sus": "Out", "NA": "Out", "DNR": "Out", "COV": "Out",
}  # fmt: skip
SEVERITY = {"Questionable": 1, "Doubtful": 2, "Out": 3}  # anything else: 0


def practice(value: str | None) -> str | None:
    v = (value or "").strip().lower()
    if not v:
        return None
    if "dnp" in v or "did not" in v or v == "out":
        return "Did Not Participate In Practice"
    if "limited" in v or v == "lp":
        return "Limited Participation in Practice"
    if "full" in v or v == "fp":
        return "Full Participation in Practice"
    return None


def sleeper_status(players: pl.DataFrame, crosswalk: pl.DataFrame,
                   fetched_at: datetime) -> pl.DataFrame:  # fmt: skip
    """One row per player Sleeper lists with a designation or practice note, keyed by
    gsis_id: status, practice, injury, source, updated_at (Sleeper's news time),
    fetched_at."""
    p = players.filter(
        pl.col("injury_status").is_not_null() | pl.col("practice_participation").is_not_null()
    )
    if p.height == 0:
        return _empty()
    xw = crosswalk.select(pl.col("sleeper_id"), pl.col("gsis_id").alias("xw_gsis")).drop_nulls()
    p = p.join(xw, on="sleeper_id", how="left").with_columns(
        pl.coalesce("xw_gsis", "gsis_id").alias("key")
    )
    rows = []
    for r in p.filter(pl.col("key").is_not_null()).iter_rows(named=True):
        upd = None
        if r.get("news_updated"):
            try:
                upd = datetime.fromtimestamp(int(r["news_updated"]) / 1000, UTC)
            except ValueError:
                upd = None
        rows.append({
            "key": r["key"], "status": STATUS.get(r["injury_status"] or ""),
            "practice": practice(r["practice_participation"]), "injury": r["injury_body_part"],
            "source": SOURCE, "updated_at": upd, "fetched_at": fetched_at.astimezone(UTC),
            "designation": r["injury_status"],  # as Sleeper says it: IR, PUP, Out...
            "injury_since": (r.get("injury_start_date") or None),
        })  # fmt: skip
    return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


_SCHEMA = {
    "key": pl.Utf8, "status": pl.Utf8, "practice": pl.Utf8, "injury": pl.Utf8,
    "source": pl.Utf8, "updated_at": pl.Datetime("us", "UTC"),
    "fetched_at": pl.Datetime("us", "UTC"),
    "designation": pl.Utf8,  # the source's own word (Sleeper: IR, PUP, Sus...); status maps it
    "injury_since": pl.Utf8,  # ISO date the injury on file began, when the source says
}  # fmt: skip


def _conform(df: pl.DataFrame) -> pl.DataFrame:
    """Every _SCHEMA column (older stored fetches lack the newer ones: null)."""
    return df.with_columns(pl.lit(None, t).alias(c) for c, t in _SCHEMA.items()
                           if c not in df.columns).select(list(_SCHEMA)).cast(_SCHEMA)  # fmt: skip


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def combine(nflverse: pl.DataFrame, sleeper: pl.DataFrame | None) -> pl.DataFrame:
    """Per player, the more severe designation of the two sources (a missed ruling costs
    far more than a cautious one), with that source's timestamps. On equal severity the
    source with a practice note wins, then Sleeper (fresher).

    The practice note is taken separately: the official report's (nflverse) when it has
    one, else Sleeper's. Midweek, Sleeper often already lists a designation with no
    practice note while the official report has practice and no designation yet; taking
    the note with the designation would drop the practice signal."""
    frames = [_conform(nflverse)]
    if sleeper is not None and sleeper.height:
        frames.append(_conform(sleeper))
    allr = pl.concat(frames).with_columns(
        pl.col("status").replace_strict(SEVERITY, default=0).alias("_sev"),
        pl.col("practice").is_not_null().cast(pl.Int8).alias("_prac"),
        (pl.col("source") == SOURCE).cast(pl.Int8).alias("_fresh"),
    )
    best = (
        allr.sort(["key", "_sev", "_prac", "_fresh"], descending=[False, True, True, True])
        .unique("key", keep="first", maintain_order=True)
        .drop("_sev", "_prac", "_fresh")
    )
    note = (
        allr.filter(pl.col("practice").is_not_null())
        .sort(["key", "_fresh"])  # official report (nflverse) first
        .unique("key", keep="first", maintain_order=True)
        .select("key", pl.col("practice").alias("_note"))
    )
    return (
        best.join(note, on="key", how="left")
        .with_columns(pl.coalesce("_note", "practice").alias("practice"))
        .drop("_note")
        .select(list(_SCHEMA))
    )
