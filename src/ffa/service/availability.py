"""Why a rostered player isn't playing this week: bye, injured reserve, PUP, suspension,
ruled out... Each reason says where it came from, so a bye is never confused with a
season-ending injury, and players who may return stay visible."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from ffa.warehouse import duck

# NFL transaction status codes as published in nflverse weekly rosters
# (status_description_abbr), for players not on the active roster.
ROSTER_CODES = {
    "R01": ("injured_reserve", "Injured reserve"),
    "R48": ("injured_reserve", "Injured reserve, designated to return"),
    "R04": ("pup", "Physically unable to perform (PUP) list"),
    "R05": ("non_football_injury", "Non-football injury list"),
    "R02": ("retired", "Retired"),
    "W03": ("released", "Waived / released"),
    "I01": ("inactive", "Inactive"),
    "I02": ("inactive", "Inactive"),
}
ROSTER_STATUS = {  # fallback when the code is unknown
    "RES": ("reserve_list", "On a reserve list"),
    "INA": ("inactive", "Inactive"),
    "DEV": ("practice_squad", "On the practice squad"),
    "CUT": ("released", "Released"),
    "RET": ("retired", "Retired"),
    "EXE": ("exempt", "Exempt list"),
}
SLEEPER = {  # Sleeper injury_status values that mean he won't play
    "IR": ("injured_reserve", "Injured reserve"),
    "PUP": ("pup", "Physically unable to perform (PUP) list"),
    "Sus": ("suspended", "Suspended"),
    "NA": ("not_active", "Not active"),
    "DNR": ("not_active", "Did not report"),
    "COV": ("not_active", "Reserve/COVID"),
    "Out": ("out", "Out"),
}
MAY_RETURN = {"bye", "out", "injured_reserve", "pup", "non_football_injury", "suspended",
              "inactive", "not_active", "reserve_list"}  # fmt: skip


@dataclass
class NotPlaying:
    key: str | None
    name: str
    reason: str  # bye, out, injured_reserve, pup, suspended, released, ... or unknown
    detail: str
    source: str
    may_return: bool  # later this season (bye: next week)
    fantasy_slot: str | None = None  # e.g. "IR" if he sits in your league's IR slot

    def as_dict(self) -> dict:
        return asdict(self)


def _latest_roster(con, season: int, week: int) -> pl.DataFrame:
    for w in (week, week - 1, week - 2):
        if w < 1:
            break
        try:
            t = duck.week(con, "rosters", season, w)
        except duck.MissingPartition:
            continue
        return con.sql(
            f"""select gsis_id as key, team, status, status_description_abbr as code,
                       {w} as roster_week from {t}"""
        ).pl()
    return pl.DataFrame(schema={"key": pl.Utf8, "team": pl.Utf8, "status": pl.Utf8,
                                "code": pl.Utf8, "roster_week": pl.Int64})  # fmt: skip


def _teams_playing(con, season: int, week: int) -> set[str] | None:
    try:
        rows = con.sql(
            f"select home_team, away_team from schedules where season = {int(season)} "
            f"and week = {int(week)}"
        ).fetchall()
    except Exception:
        return None
    return {t for r in rows for t in r} if rows else None


def _sleeper(con, keys: list[str]) -> dict[str, str]:
    """gsis key -> Sleeper injury_status (current, fetched once a day)."""
    try:
        in_list = ", ".join("'" + k.replace("'", "''") + "'" for k in keys) or "''"
        rows = con.sql(
            f"""select x.gsis_id, s.injury_status from player_ids x
                join sleeper_players s on s.sleeper_id = x.sleeper_id
                where x.gsis_id in ({in_list}) and s.injury_status is not null"""
        ).fetchall()
    except Exception:
        return {}
    return dict(rows)


def _teams_of(con, keys: list[str]) -> dict[str, str]:
    """Fallback team per player from the ID crosswalk (when roster data is missing)."""
    try:
        in_list = ", ".join("'" + k.replace("'", "''") + "'" for k in keys) or "''"
        return dict(con.sql(
            f"select gsis_id, team from player_ids where gsis_id in ({in_list})"
        ).fetchall())  # fmt: skip
    except Exception:
        return {}


def reasons(con, season: int, week: int, players: dict[str, str]) -> list[NotPlaying]:
    """Reason for each player in `players` (key -> name) who has no projection this week.
    Precedence: roster reserve status, Sleeper status, bye, other roster status."""
    keys = list(players)
    rframe = _latest_roster(con, season, week)
    roster = {r["key"]: r for r in rframe.iter_rows(named=True)}
    have_rosters = rframe.height > 0
    team_of = _teams_of(con, keys)
    playing = _teams_playing(con, season, week)
    sleeper = _sleeper(con, keys)
    out = []
    for k in keys:
        r, name = roster.get(k), players[k]
        src_roster = f"nflverse roster, week {r['roster_week']}" if r else ""
        if r and r["status"] == "RES" and r["code"] in ROSTER_CODES:
            reason, detail = ROSTER_CODES[r["code"]]
            src = src_roster
        elif sleeper.get(k) in SLEEPER:
            reason, detail = SLEEPER[sleeper[k]]
            src = "Sleeper player status"
        elif (
            (r is None or r["status"] == "ACT") and playing is not None
            and (team := (r["team"] if r else team_of.get(k))) and team not in playing
        ):  # fmt: skip
            reason, detail, src = "bye", f"{team} has no game in week {week}", "schedule"
        elif r and r["status"] in ROSTER_STATUS:
            reason, detail = ROSTER_CODES.get(r["code"] or "", ROSTER_STATUS[r["status"]])
            src = src_roster
        elif r is None and have_rosters:
            reason, detail, src = ("not_on_nfl_roster", "Not on an NFL roster this week",
                                   "nflverse rosters")  # fmt: skip
        else:
            reason, detail, src = ("unknown", "No projection; reason not found in roster, "
                                   "status or schedule data", "")  # fmt: skip
        out.append(NotPlaying(k, name, reason, detail, src, reason in MAY_RETURN))
    return out
