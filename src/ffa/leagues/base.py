"""Platform-neutral league model and the adapter interface every platform implements.

Adapters return platform IDs only. Mapping players onto the canonical key (nflverse
gsis_id, or DST_<TEAM>) happens afterwards in `ffa.leagues.sync`, so every platform goes
through the same resolver and the same unmatched-player report.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class Platform(StrEnum):
    SLEEPER = "sleeper"
    YAHOO = "yahoo"


class SlotType(StrEnum):
    STARTER = "starter"
    BENCH = "bench"
    IR = "ir"
    TAXI = "taxi"


# Canonical starting slots and the positions each one accepts.
SLOT_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "WRRB_FLEX": frozenset({"WR", "RB"}),
}
NON_STARTING = {"BN": SlotType.BENCH, "IR": SlotType.IR, "TAXI": SlotType.TAXI}


class PlayerInfo(BaseModel):
    """What a platform says about a player; used for name matching when IDs are missing."""

    name: str | None = None
    position: str | None = None
    team: str | None = None


class RosterPlayer(BaseModel):
    platform_id: str
    slot_type: SlotType
    lineup_slot: str | None = None  # canonical slot for starters, e.g. "FLEX"
    info: PlayerInfo = Field(default_factory=PlayerInfo)
    key: str | None = None  # filled by the resolver
    match: str | None = None  # how `key` was found: id, name, name+team, dst, unmatched…


class Team(BaseModel):
    team_id: str
    name: str
    owner_name: str | None = None
    is_me: bool = False
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    players: list[RosterPlayer]
    empty_lineup_slots: list[str] = Field(default_factory=list)


class League(BaseModel):
    platform: Platform
    league_id: str
    season: int
    name: str
    num_teams: int
    lineup_slots: list[str]  # canonical starting slots, in platform order
    bench_slots: int
    ir_slots: int
    taxi_slots: int
    scoring: dict[str, float]  # raw platform scoring keys; translated in ffa.scoring
    unsupported_slots: list[str] = Field(default_factory=list)  # e.g. IDP slots
    teams: list[Team]

    @property
    def uid(self) -> str:
        return f"{self.platform}:{self.league_id}"

    def my_team(self) -> Team | None:
        return next((t for t in self.teams if t.is_me), None)


class LeagueRef(BaseModel):
    platform: Platform
    league_id: str
    name: str
    season: int
    num_teams: int


class WeekResult(BaseModel):
    """One team's scored week, used to check our scoring engine against the platform's."""

    team_id: str
    matchup_id: int | None
    points: float
    starters: list[str]
    player_points: dict[str, float]  # platform_id -> points the platform awarded


class LeagueAdapter(Protocol):
    platform: Platform

    def find_leagues(self, user: str, season: int) -> list[LeagueRef]: ...

    def fetch_league(self, league_id: str) -> League: ...

    def fetch_week(self, league_id: str, week: int) -> list[WeekResult]: ...
