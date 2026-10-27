import json
from pathlib import Path

import httpx
import polars as pl
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from ffa.db.models import Base, LeagueRow, RosterEntryRow, TeamRow
from ffa.ingest.ids import Resolver
from ffa.leagues import sync
from ffa.leagues.base import PlayerInfo, SlotType
from ffa.leagues.sleeper import SleeperAdapter, SleeperClient

FIX = Path(__file__).parent / "fixtures" / "sleeper"
LEAGUE = "900000000000000001"
ROOT = Path(__file__).parents[1]


def _transport(overrides: dict[str, object] | None = None) -> httpx.MockTransport:
    routes = {
        f"/v1/league/{LEAGUE}": "league.json",
        f"/v1/league/{LEAGUE}/users": "users.json",
        f"/v1/league/{LEAGUE}/rosters": "rosters.json",
        f"/v1/league/{LEAGUE}/matchups/4": "matchups_4.json",
        "/v1/user/me_user": "user.json",
        "/v1/state/nfl": "state.json",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if overrides and path in overrides:
            return httpx.Response(200, json=overrides[path])
        if path not in routes:
            return httpx.Response(404, json=None)
        return httpx.Response(200, json=json.loads((FIX / routes[path]).read_text()))

    return httpx.MockTransport(handler)


def _adapter(**kw) -> SleeperAdapter:
    client = SleeperClient(httpx.Client(transport=_transport(kw.pop("overrides", None))))
    return SleeperAdapter(client, **kw)


@pytest.fixture
def league():
    return _adapter(me="me_user").fetch_league(LEAGUE)


def test_league_settings(league):
    assert league.uid == f"sleeper:{LEAGUE}"
    assert league.season == 2026 and league.num_teams == 12
    assert league.lineup_slots == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert (league.bench_slots, league.ir_slots, league.taxi_slots) == (6, 1, 0)
    assert league.scoring["rec"] == 0.5
    assert league.unsupported_slots == []


def test_my_team_found_via_co_owner(league):
    mine = league.my_team()
    assert mine is not None and mine.team_id == "8" and mine.name == "My Team"
    assert mine.points_for == pytest.approx(336.42)
    assert sum(t.is_me for t in league.teams) == 1


def test_roster_slots(league):
    mine = league.my_team()
    by_type = {t: [p for p in mine.players if p.slot_type == t] for t in SlotType}
    assert len(by_type[SlotType.STARTER]) == 9
    assert [p.platform_id for p in by_type[SlotType.IR]] == ["8142"]
    assert len(by_type[SlotType.BENCH]) == 5
    assert len(mine.players) == 15
    qb = by_type[SlotType.STARTER][0]
    assert (qb.platform_id, qb.lineup_slot) == ("11564", "QB")
    flex = next(p for p in mine.players if p.lineup_slot == "FLEX")
    assert flex.platform_id == "8148"
    dst = next(p for p in mine.players if p.platform_id == "CIN")
    assert dst.info.position == "DEF" and dst.lineup_slot == "DEF"


def test_empty_lineup_slot_recorded(league):
    team1 = next(t for t in league.teams if t.team_id == "1")
    assert team1.empty_lineup_slots == ["K"]
    assert len([p for p in team1.players if p.slot_type == SlotType.STARTER]) == 8
    assert len(team1.players) == 15  # empty slot does not drop anyone


def test_team_name_falls_back_to_display_name(league):
    assert next(t for t in league.teams if t.team_id == "4").name == "manager_four"


def test_unsupported_slots_reported():
    raw = json.loads((FIX / "league.json").read_text())
    raw["roster_positions"] = [*raw["roster_positions"], "IDP_FLEX"]
    lg = _adapter(overrides={f"/v1/league/{LEAGUE}": raw}).fetch_league(LEAGUE)
    assert lg.unsupported_slots == ["IDP_FLEX"]
    assert "IDP_FLEX" in lg.lineup_slots  # kept in order so starters still line up


def test_week_points_and_state():
    a = _adapter()
    week = {r.team_id: r for r in a.fetch_week(LEAGUE, 4)}
    assert week["8"].points == pytest.approx(104.72)
    assert week["8"].player_points["11564"] == pytest.approx(24.12)
    assert "0" not in week["1"].starters
    assert a.current_week() == 5


# ---------------------------------------------------------------- resolve + store


def _resolver(*, know_13545: bool = False) -> Resolver:
    sleeper_ids = [
        "11560", "11610", "12512", "12534", "13414", "3257", "4034", "4993", "5844", "5859",
        "7525", "7564", "8167", "9500", "11604", "11834", "12517", "13330", "5872", "6770",
        "8112", "8132", "8150", "8183", "8228", "8408", "9511", "11564", "11576", "1166",
        "11786", "12506", "13298", "3321", "4866", "7588", "8142", "8148", "9487", "9493",
        "9753",
    ]  # fmt: skip
    rows = [(f"G-{s}", s, f"Player {s}", "WR", "NYG", 2026) for s in sleeper_ids]
    if know_13545:
        rows.append(("G-NEWK", None, "Brand New Kicker", "K", "DAL", 2026))
    xw = pl.DataFrame(
        rows,
        schema=["gsis_id", "sleeper_id", "name", "position", "team", "last_season"],
        orient="row",
    ).with_columns(
        yahoo_id=pl.lit(None, pl.Utf8),
        name_key=pl.col("name").str.to_lowercase().str.replace_all(r"[^a-z]", ""),
    )
    return Resolver(xw)


def test_resolve_reports_unknown_player(league):
    rep = sync.report(sync.resolve(league, _resolver()))
    assert rep.players == 45
    assert rep.by_method == {"id": 41, "dst": 3, "unmatched": 1}
    assert [u.platform_id for u in rep.unmatched] == ["13545"]
    assert rep.unmatched[0].slot_type == "starter"
    assert rep.match_rate == pytest.approx(44 / 45)


def test_name_fallback_with_sleeper_player_db():
    names = {"13545": PlayerInfo(name="Brand New Kicker", position="K", team="DAL")}
    lg = _adapter(me="me_user", players=names).fetch_league(LEAGUE)
    rep = sync.report(sync.resolve(lg, _resolver(know_13545=True)))
    assert rep.unmatched == []
    assert rep.by_method["name"] == 1


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_store_is_idempotent_and_replaces_rosters(league, db):
    resolved = sync.resolve(league, _resolver())
    sync.store(resolved, db)
    sync.store(resolved, db)
    assert db.scalar(select(func.count()).select_from(LeagueRow)) == 1
    assert db.scalar(select(func.count()).select_from(TeamRow)) == 3
    assert db.scalar(select(func.count()).select_from(RosterEntryRow)) == 45

    # A trade: team 1 loses a bench player. The next sync must reflect it.
    team1 = next(t for t in resolved.teams if t.team_id == "1")
    smaller = team1.model_copy(update={"players": team1.players[:-1]})
    traded = resolved.model_copy(
        update={"teams": [smaller if t.team_id == "1" else t for t in resolved.teams]}
    )
    sync.store(traded, db)
    assert db.scalar(select(func.count()).select_from(RosterEntryRow)) == 44

    me = db.scalar(select(TeamRow).where(TeamRow.is_me))
    assert me.platform_team_id == "8"
    unmatched = db.scalars(select(RosterEntryRow).where(RosterEntryRow.player_key.is_(None)))
    assert [e.platform_player_id for e in unmatched] == ["13545"]


def test_migrations_match_models(tmp_path):
    """`alembic upgrade head` produces exactly the schema the models describe."""
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.cmd_opts = type("o", (), {"x": [f"url={url}"]})()
    command.upgrade(cfg, "head")
    with create_engine(url).connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"version_table": "ffa_alembic_version"})
        diff = compare_metadata(ctx, Base.metadata)
    assert diff == []


def test_season_settings_last_fantasy_week():
    from ffa.leagues.sleeper import season_settings

    assert season_settings({"playoff_week_start": 15, "playoff_teams": 6})["last_week"] == 17
    assert season_settings({"playoff_week_start": 15, "playoff_teams": 4})["last_week"] == 16
    two_week_final = {"playoff_week_start": 15, "playoff_teams": 4, "playoff_round_type": 1}
    assert season_settings(two_week_final)["last_week"] == 17
    assert "last_week" not in season_settings({})


def test_draft_positions_and_completed_trades_are_kept():
    lg = json.loads((FIX / "league.json").read_text()) | {"draft_id": "d1"}
    trade = {"type": "trade", "status": "complete", "transaction_id": "t1",
             "created": 1761000000000, "adds": {"4046": 2, "6794": 1},
             "drops": {"4046": 1, "6794": 2}, "roster_ids": [1, 2]}  # fmt: skip
    over = {
        f"/v1/league/{LEAGUE}": lg,
        "/v1/draft/d1/picks": [{"player_id": "4046", "pick_no": 3, "round": 1},
                               {"player_id": "6794", "pick_no": 30, "round": 3}],
        f"/v1/league/{LEAGUE}/transactions/2": [
            trade, {"type": "waiver", "status": "complete", "transaction_id": "w1"},
            {"type": "trade", "status": "failed", "transaction_id": "t2"}],
    }  # fmt: skip
    s = _adapter(me="me_user", overrides=over).fetch_league(LEAGUE).settings
    assert s["draft_picks"] == {"4046": [3, 1], "6794": [30, 3]}
    assert [t["id"] for t in s["trades"]] == ["t1"] and s["trades"][0]["week"] == 2
    assert s["trades"][0]["adds"] == {"4046": "2", "6794": "1"}
    assert "playoff_week_start" in s  # season settings still there
