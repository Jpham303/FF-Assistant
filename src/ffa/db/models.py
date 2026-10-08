"""App database (Postgres): leagues, teams and current rosters.

Schema changes go through Alembic (`migrations/`); never edit tables by hand.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class LeagueRow(Base):
    __tablename__ = "leagues"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "sleeper:<league_id>"
    platform: Mapped[str] = mapped_column(String(16))
    platform_league_id: Mapped[str] = mapped_column(String)
    season: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String)
    num_teams: Mapped[int] = mapped_column(Integer)
    lineup_slots: Mapped[list[str]] = mapped_column(JSON)
    bench_slots: Mapped[int] = mapped_column(Integer)
    ir_slots: Mapped[int] = mapped_column(Integer)
    taxi_slots: Mapped[int] = mapped_column(Integer)
    scoring: Mapped[dict[str, float]] = mapped_column(JSON)
    unsupported_slots: Mapped[list[str]] = mapped_column(JSON)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    teams: Mapped[list[TeamRow]] = relationship(
        back_populates="league", cascade="all, delete-orphan"
    )


class TeamRow(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "sleeper:<league>:<team>"
    league_id: Mapped[str] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"))
    platform_team_id: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    owner_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_me: Mapped[bool] = mapped_column(default=False)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    ties: Mapped[int] = mapped_column(Integer, default=0)
    points_for: Mapped[float] = mapped_column(default=0.0)
    empty_lineup_slots: Mapped[list[str]] = mapped_column(JSON, default=list)

    league: Mapped[LeagueRow] = relationship(back_populates="teams")
    entries: Mapped[list[RosterEntryRow]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )


class RosterEntryRow(Base):
    __tablename__ = "roster_entries"
    __table_args__ = (Index("ix_roster_entries_player_key", "player_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    platform_player_id: Mapped[str] = mapped_column(String)
    player_key: Mapped[str | None] = mapped_column(String, nullable=True)  # gsis_id / DST_X
    match_method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    slot_type: Mapped[str] = mapped_column(String(8))
    lineup_slot: Mapped[str | None] = mapped_column(String(16), nullable=True)

    team: Mapped[TeamRow] = relationship(back_populates="entries")
