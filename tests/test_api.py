"""Service and API tests on a small synthetic warehouse and a SQLite app database."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ffa.api import main
from ffa.db.models import Base, LeagueRow, RosterEntryRow, TeamRow
from ffa.scoring.rules import default_rules
from ffa.service import startsit
from ffa.warehouse import duck, store

SEASON, WEEK, N = 2026, 6, 400

# key, name, team, position, mean rushing yards, mean receptions
PLAYERS = [
    ("00-A", "Alpha Back", "DAL", "RB", 80, 3),
    ("00-B", "Bravo Back", "JAX", "RB", 50, 2),
    ("00-C", "Charlie Wide", "DAL", "WR", 0, 6),
    ("00-D", "Delta Wide", "JAX", "WR", 0, 4),
    ("00-Q", "Quinn Passer", "DAL", "QB", 10, 0),
]
BYE = ("00-E", "Echo Williams", "DET", "WR")  # rostered, on bye: no projection


def _sims() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for key, name, team, pos, rush, rec in PLAYERS:
        for s in range(N):
            r = rng.poisson(rec)
            rows.append({
                "key": key, "name": name, "team": team, "position": pos, "sim": s,
                "rush_att": float(rush // 4), "rush_yd": float(rng.poisson(rush)),
                "rush_td": float(rng.random() < 0.3), "rec_tgt": float(r + 1),
                "rec": float(r), "rec_yd": float(r * 11), "rec_td": float(rng.random() < 0.2),
                "pass_yd": float(rng.normal(250, 40)) if pos == "QB" else 0.0,
                "pass_td": float(rng.poisson(1.8)) if pos == "QB" else 0.0,
                "pass_int": float(rng.poisson(0.8)) if pos == "QB" else 0.0,
            })  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("sim").cast(pl.Int32))


@pytest.fixture
def warehouse(tmp_path):
    root = tmp_path / "wh"
    tag = [pl.lit(SEASON).cast(pl.Int32).alias("season"), pl.lit(WEEK).cast(pl.Int32).alias("week")]
    store.write(_sims().with_columns(*tag), root, "projection_sims")
    inputs = pl.DataFrame({
        "key": [p[0] for p in PLAYERS],
        "opponent": ["GB" if p[2] == "DAL" else "HOU" for p in PLAYERS],
        "p_play": [1.0, 0.8, 1.0, 1.0, 1.0],
        "injury_status": [None, "Questionable", None, None, None],
        "tgt_share_s": [0.10, 0.08, 0.22, 0.18, None],
        "rz_tgt_share_s": [0.10, 0.08, 0.2, 0.18, None],
        "car_share_s": [0.55, 0.4, 0.0, 0.0, 0.08],
        "rz_car_share_s": [0.6, 0.4, 0.0, 0.0, 0.1],
    })  # fmt: skip
    store.write(inputs.with_columns(*tag), root, "projection_inputs")
    teams = pl.DataFrame({
        "team": ["DAL", "JAX"], "opponent": ["GB", "HOU"], "implied": [26.0, 22.5],
        "spread_for": [-1.5, 2.5], "total_line": [53.5, 42.5], "mu_plays": [64.0, 61.0],
        "p_pass": [0.62, 0.6],
    })  # fmt: skip
    store.write(teams.with_columns(*tag), root, "projection_teams")
    sched = pl.DataFrame({"season": [SEASON] * 2, "week": [5, 6], "home_score": [24, None]})
    store.write(sched.with_columns(pl.col("season", "week").cast(pl.Int32)), root, "schedules")
    tg = pl.DataFrame({"season": [SEASON] * 3, "week": [5] * 3, "team": ["DAL", "JAX", "DET"]})
    store.write(tg.with_columns(pl.col("season", "week").cast(pl.Int32)), root, "team_games")
    ids = pl.DataFrame({
        "gsis_id": [p[0] for p in PLAYERS] + [BYE[0]],
        "name": [p[1] for p in PLAYERS] + [BYE[1]],
        "team": [p[2] for p in PLAYERS] + [BYE[2]],
        "position": [p[3] for p in PLAYERS] + [BYE[3]],
        "last_season": [SEASON] * (len(PLAYERS) + 1),
    })  # fmt: skip
    store.write_dim(ids, root, "player_ids")
    return root


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    s = sessionmaker(engine, expire_on_commit=False)()
    s.add(LeagueRow(
        id="sleeper:1", platform="sleeper", platform_league_id="1", season=SEASON,
        name="Test League", num_teams=2, lineup_slots=["QB", "RB", "WR", "FLEX"],
        bench_slots=5, ir_slots=1, taxi_slots=0, scoring={"rec": 1.0, "rec_yd": 0.1,
        "rush_yd": 0.1, "rush_td": 6, "rec_td": 6, "pass_yd": 0.04, "pass_td": 4},
        unsupported_slots=[], synced_at=datetime.now(UTC),
    ))  # fmt: skip
    s.add(TeamRow(id="sleeper:1:8", league_id="sleeper:1", platform_team_id="8", name="Mine",
                  is_me=True))  # fmt: skip
    for key, name, *_ in PLAYERS[:4]:
        s.add(RosterEntryRow(team_id="sleeper:1:8", platform_player_id=key, player_key=key,
                             name=name, slot_type="bench"))  # fmt: skip
    s.add(RosterEntryRow(team_id="sleeper:1:8", platform_player_id="E", player_key=BYE[0],
                         name=BYE[1], slot_type="bench"))  # fmt: skip
    s.commit()
    yield s
    s.close()


@pytest.fixture
def client(warehouse, db, monkeypatch):
    monkeypatch.setattr(main.settings, "current_season", SEASON)
    monkeypatch.setattr(main.settings, "sleeper_league_id", None)

    def con():
        c = duck.connect(warehouse)
        try:
            yield c
        finally:
            c.close()

    main.app.dependency_overrides[main.get_con] = con
    main.app.dependency_overrides[main.get_db] = lambda: db
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


# ---------------------------------------------------------------- service


def test_search_exact_name_never_swapped_for_similar(warehouse):
    con = duck.connect(warehouse)
    res = startsit.search(con, SEASON, WEEK, "Echo Williams")
    assert res[0]["key"] == BYE[0] and res[0]["match"] == 1.0
    assert res[0]["projected"] is False  # on bye: reported, not replaced
    assert startsit.search(con, SEASON, WEEK, "nobody atall") == []
    assert startsit.search(con, SEASON, WEEK, "JAX")[0]["key"] == "DST_JAX"
    assert startsit.search(con, SEASON, WEEK, "alpha bak")[0]["key"] == "00-A"  # typo


def test_compare_probabilities_consistent(warehouse):
    con = duck.connect(warehouse)
    w = startsit.load_week(con, SEASON, WEEK, ["00-A", "00-B", "00-C"])
    c = startsit.compare(w, default_rules())
    assert c.recommendation == c.players[0].key
    assert [v.mean for v in c.players] == sorted((v.mean for v in c.players), reverse=True)
    assert sum(v.p_best for v in c.players) == pytest.approx(1.0)
    p = {(x["a"], x["b"]): x["p_a_beats_b"] for x in c.pairwise}
    for (a, b), v in p.items():
        assert v + p[(b, a)] == pytest.approx(1.0, abs=1e-3)
    assert any("Bravo Back is Questionable" in n for n in c.notes)


def test_drivers_carry_numbers_and_size_matchups():
    row = {"team": "DAL", "car_share_s": 0.5, "rz_car_share_s": 0.6, "tgt_share_s": 0.1,
           "rz_tgt_share_s": 0.1, "m_ypc": 0.92, "opponent": "JAX"}  # fmt: skip
    team = {"implied": 26.0, "opponent": "JAX", "spread_for": -1.5, "total_line": 50.0,
            "mu_plays": 64.0, "p_pass": 0.6}  # fmt: skip
    d = startsit.drivers(row, team, "RB")
    assert d[1].startswith("Expected share of designed runs: 50.0%")  # rushing role first
    m = [x for x in d if x.startswith("Matchup")]
    assert m and "8% fewer yards per carry against JAX" in m[0]
    assert "small factor" in m[0]


def test_best_lineup_fills_restrictive_slots_first(warehouse):
    con = duck.connect(warehouse)
    w = startsit.load_week(con, SEASON, WEEK, [p[0] for p in PLAYERS])
    views = startsit.player_views(w, default_rules())
    lineup = startsit.best_lineup(views, ["QB", "RB", "WR", "FLEX", "TE"])
    slots = {s["slot"]: s["key"] for s in lineup}
    assert slots["QB"] == "00-Q"
    assert slots["TE"] is None  # no TE rostered: slot left empty, not filled by a WR
    assert len({s["key"] for s in lineup if s["key"]}) == 4


# ---------------------------------------------------------------- API


def test_week_and_search(client):
    assert client.get("/week").json() == {"season": SEASON, "week": WEEK,
                                          "projections_available": True}  # fmt: skip
    r = client.get("/players/search", params={"q": "echo williams"}).json()
    assert r[0]["key"] == BYE[0] and r[0]["projected"] is False


def test_start_sit(client):
    r = client.post("/start-sit", json={"players": ["00-A", "00-B"]})
    assert r.status_code == 200
    d = r.json()
    assert d["scoring"] == "default half-PPR"
    assert d["recommendation"] == "00-A"
    assert {p["key"] for p in d["players"]} == {"00-A", "00-B"}
    assert all(p["drivers"] for p in d["players"])


def test_start_sit_uses_league_scoring(client):
    half = client.post("/start-sit", json={"players": ["00-C", "00-D"]}).json()
    full = client.post("/start-sit", json={"players": ["00-C", "00-D"], "league": "sleeper:1"})
    full = full.json()
    assert full["scoring"] == "Test League (sleeper)"
    by = {p["key"]: p["mean"] for p in full["players"]}
    assert by["00-C"] > {p["key"]: p["mean"] for p in half["players"]}["00-C"]  # PPR > half


def test_start_sit_rejects_unprojected_and_bad_input(client):
    r = client.post("/start-sit", json={"players": ["00-A", BYE[0]]})
    assert r.status_code == 422 and BYE[0] in r.json()["detail"]
    assert client.post("/start-sit", json={"players": ["00-A", "00-A"]}).status_code == 422
    assert client.post("/start-sit", json={"players": ["00-A"]}).status_code == 422
    r = client.post("/start-sit", json={"players": ["00-A", "00-B"], "league": "sleeper:9"})
    assert r.status_code == 404


def test_projections(client):
    r = client.get("/projections", params={"keys": "00-C,00-A"}).json()
    assert [p["key"] for p in r] == ["00-C", "00-A"]
    assert r[0]["p10"] <= r[0]["median"] <= r[0]["p90"]
    assert "receptions" in r[0]["expected_stats"]


def test_my_team(client):
    d = client.get("/leagues/sleeper:1/my-team").json()
    assert d["team"] == "Mine" and d["week"] == WEEK
    assert [s["slot"] for s in d["lineup"]] == ["QB", "RB", "WR", "FLEX"]
    assert d["lineup"][0]["key"] is None  # QB not rostered
    slots = {s["slot"]: s for s in d["lineup"]}
    assert slots["RB"]["key"] == "00-A" and slots["WR"]["key"] == "00-C"
    (benched,) = d["bench"]  # the lower of the two FLEX candidates sits
    assert benched["mean"] <= slots["FLEX"]["mean"]
    assert d["not_playing"] == [BYE[1]]
    assert client.get("/leagues").json()[0]["league"] == "sleeper:1"
