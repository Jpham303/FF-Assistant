"""Resolve a fetched league onto canonical player keys and store it in Postgres.

A sync replaces the league's current state (teams and rosters). It is idempotent: running
it twice in a row leaves the database unchanged.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.ingest.ids import Resolver
from ffa.leagues.base import League, LeagueAdapter, PlayerInfo, RosterPlayer


class UnmatchedPlayer(BaseModel):
    team: str
    platform_id: str
    name: str | None
    position: str | None
    slot_type: str
    reason: str  # "unmatched" or "ambiguous"


class SyncReport(BaseModel):
    league: str
    teams: int
    players: int
    by_method: dict[str, int]
    unmatched: list[UnmatchedPlayer]
    unsupported_slots: list[str]

    @property
    def match_rate(self) -> float:
        return 1 - len(self.unmatched) / self.players if self.players else 1.0


def resolve(league: League, resolver: Resolver) -> League:
    """Fill `key` and `match` on every roster player."""
    teams = []
    for team in league.teams:
        players = []
        for p in team.players:
            m = resolver.resolve(
                league.platform.value,
                p.platform_id,
                name=p.info.name,
                position=p.info.position,
                team=p.info.team,
            )
            players.append(p.model_copy(update={"key": m.key, "match": m.method}))
        teams.append(team.model_copy(update={"players": players}))
    return league.model_copy(update={"teams": teams})


def report(league: League) -> SyncReport:
    everyone: list[tuple[str, RosterPlayer]] = [
        (t.name, p) for t in league.teams for p in t.players
    ]
    return SyncReport(
        league=league.uid,
        teams=len(league.teams),
        players=len(everyone),
        by_method=dict(Counter(p.match or "unresolved" for _, p in everyone)),
        unmatched=[
            UnmatchedPlayer(
                team=team,
                platform_id=p.platform_id,
                name=p.info.name,
                position=p.info.position,
                slot_type=p.slot_type.value,
                reason=p.match or "unmatched",
            )
            for team, p in everyone
            if p.key is None
        ],
        unsupported_slots=league.unsupported_slots,
    )


def store(league: League, db: Session, *, now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    # Explicit child-first deletes, so this works without relying on DB-level cascades.
    team_ids = select(TeamRow.id).where(TeamRow.league_id == league.uid)
    db.execute(delete(RosterEntryRow).where(RosterEntryRow.team_id.in_(team_ids)))
    db.execute(delete(TeamRow).where(TeamRow.league_id == league.uid))
    db.merge(
        LeagueRow(
            id=league.uid,
            platform=league.platform.value,
            platform_league_id=league.league_id,
            season=league.season,
            name=league.name,
            num_teams=league.num_teams,
            lineup_slots=league.lineup_slots,
            bench_slots=league.bench_slots,
            ir_slots=league.ir_slots,
            taxi_slots=league.taxi_slots,
            scoring=league.scoring,
            unsupported_slots=league.unsupported_slots,
            synced_at=now,
        )
    )
    for t in league.teams:
        team_uid = f"{league.uid}:{t.team_id}"
        db.add(
            TeamRow(
                id=team_uid,
                league_id=league.uid,
                platform_team_id=t.team_id,
                name=t.name,
                owner_name=t.owner_name,
                is_me=t.is_me,
                wins=t.wins,
                losses=t.losses,
                ties=t.ties,
                points_for=t.points_for,
                empty_lineup_slots=t.empty_lineup_slots,
                entries=[
                    RosterEntryRow(
                        platform_player_id=p.platform_id,
                        player_key=p.key,
                        match_method=p.match,
                        name=p.info.name,
                        position=p.info.position,
                        slot_type=p.slot_type.value,
                        lineup_slot=p.lineup_slot,
                    )
                    for p in t.players
                ],
            )
        )
    db.commit()


def sync(adapter: LeagueAdapter, league_id: str, resolver: Resolver, db: Session) -> SyncReport:
    league = resolve(adapter.fetch_league(league_id), resolver)
    store(league, db)
    return report(league)


def load_sleeper_players(warehouse_dir: str | Path) -> dict[str, PlayerInfo]:
    """sleeper_id -> PlayerInfo from the cached Sleeper player DB, if it has been built."""
    path = Path(warehouse_dir) / "dims" / "sleeper_players.parquet"
    if not path.exists():
        return {}
    return {
        sid: PlayerInfo(name=name, position=pos, team=team)
        for sid, name, pos, team in pl.read_parquet(path)
        .select("sleeper_id", "name", "position", "team")
        .iter_rows()
    }
