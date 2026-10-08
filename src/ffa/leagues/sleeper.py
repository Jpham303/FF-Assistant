"""Sleeper adapter. Sleeper's API is public and read-only: no key, no login.

Docs: https://docs.sleeper.com — Sleeper asks for fewer than 1,000 calls a minute, and that
the 5 MB player database be fetched at most once a day.
"""

from __future__ import annotations

from typing import Any

import httpx

from ffa.leagues.base import (
    NON_STARTING,
    SLOT_ELIGIBILITY,
    League,
    LeagueRef,
    Platform,
    PlayerInfo,
    RosterPlayer,
    SlotType,
    Team,
    WeekResult,
)

BASE_URL = "https://api.sleeper.app/v1"

# Sleeper roster_positions -> canonical slots. Anything else (IDP) is reported as unsupported.
SLOT_MAP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DEF",
    "FLEX": "FLEX", "SUPER_FLEX": "SUPER_FLEX", "REC_FLEX": "REC_FLEX",
    "WRRB_FLEX": "WRRB_FLEX", "BN": "BN", "IR": "IR", "TAXI": "TAXI",
}  # fmt: skip
EMPTY = "0"


class SleeperClient:
    def __init__(self, client: httpx.Client | None = None, base_url: str = BASE_URL):
        self._client = client or httpx.Client(timeout=30, follow_redirects=True)
        self._base = base_url.rstrip("/")

    def get(self, path: str) -> Any:
        resp = self._client.get(f"{self._base}/{path.lstrip('/')}")
        resp.raise_for_status()
        return resp.json()

    def user(self, username_or_id: str) -> dict:
        data = self.get(f"user/{username_or_id}")
        if not data:
            raise LookupError(f"Sleeper user not found: {username_or_id}")
        return data


def _is_defense(pid: str) -> bool:
    return pid.isalpha()


class SleeperAdapter:
    platform = Platform.SLEEPER

    def __init__(
        self,
        client: SleeperClient | None = None,
        *,
        me: str | None = None,
        players: dict[str, PlayerInfo] | None = None,
    ):
        """`me`: your Sleeper username or user ID, to mark your team.
        `players`: optional sleeper_id -> PlayerInfo (from the Sleeper player DB) so the
        resolver can fall back to name matching for IDs the crosswalk doesn't know yet."""
        self.api = client or SleeperClient()
        self._me = me
        self._me_id: str | None = None
        self._players = players or {}

    def _my_user_id(self) -> str | None:
        if self._me and self._me_id is None:
            self._me_id = self._me if self._me.isdigit() else self.api.user(self._me)["user_id"]
        return self._me_id

    def find_leagues(self, user: str, season: int) -> list[LeagueRef]:
        uid = user if user.isdigit() else self.api.user(user)["user_id"]
        return [
            LeagueRef(
                platform=self.platform,
                league_id=lg["league_id"],
                name=lg["name"],
                season=int(lg["season"]),
                num_teams=lg["total_rosters"],
            )
            for lg in self.api.get(f"user/{uid}/leagues/nfl/{season}") or []
        ]

    def current_week(self) -> int:
        return int(self.api.get("state/nfl")["week"])

    def _info(self, pid: str) -> PlayerInfo:
        if _is_defense(pid):
            return PlayerInfo(name=f"{pid} D/ST", position="DEF", team=pid)
        return self._players.get(pid, PlayerInfo())

    def fetch_league(self, league_id: str) -> League:
        lg = self.api.get(f"league/{league_id}")
        users = {u["user_id"]: u for u in self.api.get(f"league/{league_id}/users") or []}
        rosters = self.api.get(f"league/{league_id}/rosters") or []
        settings = lg.get("settings") or {}

        raw_slots = lg.get("roster_positions") or []
        unsupported = sorted({s for s in raw_slots if s not in SLOT_MAP})
        canonical = [SLOT_MAP.get(s, s) for s in raw_slots]
        lineup = [s for s in canonical if s not in NON_STARTING]
        bench = canonical.count("BN")
        ir = settings.get("reserve_slots", canonical.count("IR")) or 0
        taxi = settings.get("taxi_slots", canonical.count("TAXI")) or 0

        me = self._my_user_id()
        teams = []
        for r in rosters:
            owner_ids = [r.get("owner_id"), *(r.get("co_owners") or [])]
            # Team name lives on the owner's profile; if the primary owner has left the
            # league, use the first co-owner who is still in it.
            owner = next((users[o] for o in owner_ids if o in users), {})
            teams.append(self._team(r, owner, lineup, is_me=bool(me) and me in owner_ids))

        return League(
            platform=self.platform,
            league_id=str(lg["league_id"]),
            season=int(lg["season"]),
            name=lg["name"],
            num_teams=int(lg.get("total_rosters") or len(rosters)),
            lineup_slots=lineup,
            bench_slots=bench,
            ir_slots=int(ir),
            taxi_slots=int(taxi),
            scoring={k: float(v) for k, v in (lg.get("scoring_settings") or {}).items()},
            unsupported_slots=unsupported,
            teams=teams,
        )

    def _team(self, r: dict, owner: dict, lineup: list[str], *, is_me: bool) -> Team:
        starters = r.get("starters") or []
        reserve = set(r.get("reserve") or [])
        taxi = set(r.get("taxi") or [])
        players: list[RosterPlayer] = []
        empty: list[str] = []

        for slot, pid in zip(lineup, starters, strict=False):
            if pid == EMPTY or not pid:
                empty.append(slot)
                continue
            players.append(
                RosterPlayer(
                    platform_id=pid,
                    slot_type=SlotType.STARTER,
                    lineup_slot=slot,
                    info=self._info(pid),
                )
            )
        started = {p.platform_id for p in players}
        for pid in r.get("players") or []:
            if pid in started:
                continue
            slot_type = (
                SlotType.IR if pid in reserve else SlotType.TAXI if pid in taxi else SlotType.BENCH
            )
            players.append(RosterPlayer(platform_id=pid, slot_type=slot_type, info=self._info(pid)))

        s = r.get("settings") or {}
        meta = owner.get("metadata") or {}
        display = owner.get("display_name")
        return Team(
            team_id=str(r["roster_id"]),
            name=meta.get("team_name") or display or f"Team {r['roster_id']}",
            owner_name=display,
            is_me=is_me,
            wins=s.get("wins", 0),
            losses=s.get("losses", 0),
            ties=s.get("ties", 0),
            points_for=s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100,
            players=players,
            empty_lineup_slots=empty,
        )

    def fetch_week(self, league_id: str, week: int) -> list[WeekResult]:
        return [
            WeekResult(
                team_id=str(m["roster_id"]),
                matchup_id=m.get("matchup_id"),
                points=float(m.get("points") or 0),
                starters=[p for p in m.get("starters") or [] if p != EMPTY],
                player_points={k: float(v) for k, v in (m.get("players_points") or {}).items()},
            )
            for m in self.api.get(f"league/{league_id}/matchups/{week}") or []
        ]


assert set(SLOT_MAP.values()) <= set(SLOT_ELIGIBILITY) | set(NON_STARTING)
