"""Where each game stands this week, and what the user has set in Sleeper.

One rule for every view (lineup, start/sit, matchup odds), so they never disagree:

- upcoming     before kickoff: the projection applies, the slot can still change
- in_progress  from kickoff: the lineup slot is locked; live points are shown, the
               projection still stands in for the total
- final        from kickoff + 4 hours: locked, and the points Sleeper awarded replace the
               projection

Sleeper's matchups for the week (fetched when asked, cached a minute) supply each team's
set starters, slot by slot, and each player's points so far.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.leagues.sleeper import SleeperClient

GAME_LENGTH = timedelta(hours=4)  # after kickoff + this, a game's points are final
UPCOMING, IN_PROGRESS, FINAL = "upcoming", "in_progress", "final"
STARTED = (IN_PROGRESS, FINAL)
CACHE_SECONDS = 60

_cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}


def status(kickoff: datetime | None, now: datetime | None = None) -> str | None:
    if kickoff is None:
        return None
    now = now or datetime.now(UTC)
    if now >= kickoff + GAME_LENGTH:
        return FINAL
    return IN_PROGRESS if now >= kickoff else UPCOMING


def fetch_matchups(platform_league_id: str, week: int,
                   client: SleeperClient | None = None) -> list[dict]:  # fmt: skip
    """Sleeper's matchups for the week (one small request, cached a minute)."""
    k = (platform_league_id, week)
    hit = _cache.get(k)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    data = (client or SleeperClient()).get(f"league/{platform_league_id}/matchups/{week}") or []
    _cache[k] = (time.monotonic(), data)
    return data


@dataclass
class LeagueWeek:
    """The user's league this week, from Sleeper and the synced rosters."""

    available: bool
    reason: str | None = None
    me: TeamRow | None = None
    opponent: TeamRow | None = None
    my_entry: dict = field(default_factory=dict)
    opp_entry: dict = field(default_factory=dict)
    matchups: list[dict] = field(default_factory=list)
    to_key: dict[str, str] = field(default_factory=dict)  # Sleeper id -> player key
    my_roster: list[RosterEntryRow] = field(default_factory=list)
    opp_roster: list[RosterEntryRow] = field(default_factory=list)

    def points(self) -> dict[str, float]:
        """Player key -> points Sleeper has awarded so far this week (every team)."""
        out: dict[str, float] = {}
        for m in self.matchups:
            for sid, p in (m.get("players_points") or {}).items():
                if sid in self.to_key:
                    out[self.to_key[sid]] = float(p)
        return out

    def starters(self, entry: dict, n_slots: int) -> list[str | None]:
        """Set starters in slot order (Sleeper lists them in roster-position order;
        "0" is an empty slot), as player keys."""
        raw = list(entry.get("starters") or [])
        raw += ["0"] * max(0, n_slots - len(raw))
        return [None if s in (None, "0") else self.to_key.get(s) for s in raw[:n_slots]]


def league_week(db, league: LeagueRow, week: int,
                matchups: list[dict] | None = None) -> LeagueWeek:  # fmt: skip
    if league.platform != "sleeper":
        return LeagueWeek(False, f"Matchups for {league.platform} not supported yet")
    me = db.scalar(select(TeamRow).where(TeamRow.league_id == league.id, TeamRow.is_me))
    if me is None:
        return LeagueWeek(False, "Your team isn't identified in this league")
    try:
        ms = matchups if matchups is not None else fetch_matchups(
            league.id.split(":", 1)[1], week)  # fmt: skip
    except Exception as exc:  # network or Sleeper down: views still answer
        return LeagueWeek(False, f"Couldn't fetch this week's matchups: {exc}", me=me)

    def roster(team: TeamRow) -> list[RosterEntryRow]:
        return list(db.scalars(select(RosterEntryRow).where(RosterEntryRow.team_id == team.id)))

    teams = list(db.scalars(select(TeamRow).where(TeamRow.league_id == league.id)))
    entries = [e for t in teams for e in roster(t)]
    to_key = {e.platform_player_id: e.player_key for e in entries if e.player_key}
    lw = LeagueWeek(True, me=me, matchups=ms, to_key=to_key, my_roster=roster(me))
    mine = next((m for m in ms if str(m.get("roster_id")) == me.platform_team_id), None)
    if mine is None:
        return LeagueWeek(False, "No fantasy matchup for your team this week", me=me, matchups=ms,
                          to_key=to_key, my_roster=lw.my_roster)  # fmt: skip
    lw.my_entry = mine
    if mine.get("matchup_id") is not None:
        opp = next((m for m in ms if m.get("matchup_id") == mine["matchup_id"]
                    and str(m.get("roster_id")) != me.platform_team_id), None)  # fmt: skip
        if opp is not None:
            lw.opp_entry = opp
            lw.opponent = next((t for t in teams if t.platform_team_id == str(opp["roster_id"])),
                               None)  # fmt: skip
            if lw.opponent is not None:
                lw.opp_roster = roster(lw.opponent)
    return lw
