"""Chance of winning the week's fantasy matchup, and the close-call tiebreaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ffa.db.models import RosterEntryRow, TeamRow
from ffa.scoring.rules import from_sleeper
from ffa.service import fantasy_matchup as fm
from ffa.service import gamestate, startsit
from ffa.warehouse import duck
from tests.test_api import SEASON, WEEK, client, db, warehouse  # noqa: F401  (fixtures)

MATCHUPS = [
    {"roster_id": 8, "matchup_id": 3, "starters": ["00-A", "00-C", "00-B", "0"],
     "players_points": {"00-C": 30.0}},
    {"roster_id": 9, "matchup_id": 3, "starters": ["00-Q"], "players_points": {"00-Q": 12.0}},
    {"roster_id": 10, "matchup_id": 4, "starters": [], "players_points": {}},
]  # fmt: skip


@pytest.fixture
def league(db):  # noqa: F811
    db.add(TeamRow(id="sleeper:1:9", league_id="sleeper:1", platform_team_id="9",
                   name="Theirs"))  # fmt: skip
    db.add(RosterEntryRow(team_id="sleeper:1:9", platform_player_id="00-Q", player_key="00-Q",
                          name="Quinn Passer", slot_type="starter"))  # fmt: skip
    db.commit()
    from ffa.db.models import LeagueRow

    row = db.get(LeagueRow, "sleeper:1")
    return db, row, from_sleeper(row.scoring)


def test_win_odds_per_candidate_against_the_set_lineup(warehouse, league):  # noqa: F811
    db_, row, rules = league
    con = duck.connect(warehouse)
    o = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules, matchups=MATCHUPS)
    assert o["available"] and o["opponent"] == "Theirs"
    assert o["opponent_lineup"] == "his lineup as set in Sleeper"
    by = {c["key"]: c for c in o["candidates"]}
    assert 0 < by["00-C"]["p_win"] < 1 and 0 < by["00-D"]["p_win"] < 1
    assert by["00-C"]["my_expected"] > by["00-D"]["my_expected"]  # 6 vs 4 catches a game
    assert by["00-C"]["p_win"] > by["00-D"]["p_win"]


def test_best_lineup_when_none_set_and_no_matchup(warehouse, league):  # noqa: F811
    db_, row, rules = league
    con = duck.connect(warehouse)
    unset = [MATCHUPS[0], MATCHUPS[1] | {"starters": []}]
    o = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules, matchups=unset)
    assert o["opponent_lineup"] == "his best lineup (none set)"
    o = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules,
                    matchups=[MATCHUPS[2]])  # fmt: skip
    assert not o["available"] and "No fantasy matchup" in o["reason"]


def test_started_starters_are_locked_and_final_ones_count_awarded_points(
    warehouse,  # noqa: F811
    league,
    monkeypatch,
):
    """DAL (Alpha, Charlie, Quinn) kicked off; JAX (Bravo, Delta) hasn't. Charlie is a set
    starter: once his game starts his slot is locked, and once it's final his 30 awarded
    points replace his projection, whoever else starts."""
    db_, row, rules = league
    real = startsit.load_week
    ko = datetime(2026, 10, 15, 0, 15, tzinfo=UTC)

    def with_kickoffs(*a, **k):
        w = real(*a, **k)
        w.kickoffs = {"DAL": ko, "JAX": ko + timedelta(days=3)}
        return w

    monkeypatch.setattr(startsit, "load_week", with_kickoffs)
    con = duck.connect(warehouse)
    live = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules,
                       matchups=MATCHUPS, now=ko + timedelta(hours=1))  # fmt: skip
    final = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules,
                        matchups=MATCHUPS, now=ko + timedelta(hours=5))  # fmt: skip
    assert live["final_games_counted"] == [] and final["final_games_counted"] == ["DAL"]
    c = {x["key"]: x for x in final["candidates"]}
    assert c["00-C"]["p_win"] is None and "locked" in c["00-C"]["note"]
    d_live = next(x for x in live["candidates"] if x["key"] == "00-D")
    d_final = c["00-D"]
    assert d_final["my_expected"] > d_live["my_expected"]  # Charlie's 30 now counts


def test_my_team_total_matches_the_matchup(client, league, monkeypatch):  # noqa: F811
    """The lineup view and the win odds use the same rule: locked slots as set, final
    games at their awarded points."""
    real = startsit.load_week
    ko = datetime(2026, 10, 15, 0, 15, tzinfo=UTC)

    def with_kickoffs(*a, **k):
        w = real(*a, **k)
        w.kickoffs = {"DAL": ko, "JAX": ko + timedelta(days=3)}
        return w

    monkeypatch.setattr(startsit, "load_week", with_kickoffs)
    monkeypatch.setattr(gamestate, "fetch_matchups", lambda *_a, **_k: MATCHUPS)
    monkeypatch.setattr(
        gamestate,
        "status",
        lambda kickoff, now=None: (
            None if kickoff is None else ("final" if kickoff == ko else "upcoming")
        ),
    )
    t = client.get("/leagues/sleeper:1/my-team").json()
    slots = {s["slot"]: s for s in t["lineup"]}
    charlie = next(s for s in t["lineup"] if s["key"] == "00-C")
    assert charlie["locked"] and charlie["game_status"] == "final"
    assert charlie["expected_points"] == 30.0 and charlie["actual_points"] == 30.0
    assert t["points_so_far"] == 30.0
    assert t["lineup_total"] == pytest.approx(sum(s["expected_points"] for s in t["lineup"]))
    assert "QB" in slots


def _close_call(warehouse):  # noqa: F811
    con = duck.connect(warehouse)
    c = startsit.compare(startsit.load_week(con, SEASON, WEEK, ["00-C", "00-D"]),
                         from_sleeper({"rec": 1.0, "rec_yd": 0.1, "rec_td": 6}))  # fmt: skip
    c.close_call = True
    return c


def test_tiebreaker_flips_only_a_close_call(warehouse):  # noqa: F811
    c = _close_call(warehouse)
    rec, other = c.recommendation, next(v.key for v in c.players if v.key != c.recommendation)
    odds = {"available": True, "opponent": "Theirs", "candidates": [
        {"key": rec, "p_win": 0.40}, {"key": other, "p_win": 0.46}]}  # fmt: skip
    out = fm.apply_tiebreak(c, odds)
    assert out.recommendation == other and out.players[0].key == other
    assert any(n.startswith("Tiebreaker:") and "46%" in n for n in out.notes)
    c2 = _close_call(warehouse)
    c2.close_call = False
    out2 = fm.apply_tiebreak(c2, odds)
    assert out2.recommendation == rec  # not a coin flip: expected points decide
    assert any(n.startswith("Chance you beat Theirs") for n in out2.notes)


def test_tiebreaker_ignores_noise_and_unavailable_odds(warehouse):  # noqa: F811
    c = _close_call(warehouse)
    rec, other = c.recommendation, next(v.key for v in c.players if v.key != c.recommendation)
    odds = {"available": True, "opponent": "Theirs", "candidates": [
        {"key": rec, "p_win": 0.500}, {"key": other, "p_win": 0.505}]}  # fmt: skip
    assert fm.apply_tiebreak(c, odds).recommendation == rec
    c = _close_call(warehouse)
    out = fm.apply_tiebreak(c, {"available": False, "reason": "No fantasy matchup this week"})
    assert out.recommendation == rec and "Matchup odds unavailable" in out.notes[-1]


def test_api_start_sit_reports_gap_drivers_and_matchup(client, league, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gamestate, "fetch_matchups", lambda *_a, **_k: MATCHUPS)
    r = client.post("/start-sit", json={"players": ["00-C", "00-D"], "league": "sleeper:1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["pairwise"][0]["median_gap"] is not None
    assert d["decision_drivers"] and {"factor", "points", "favors"} <= set(d["decision_drivers"][0])
    assert d["matchup"]["available"] and d["matchup"]["opponent"] == "Theirs"


def test_start_sit_never_picks_a_player_whose_game_started(warehouse, monkeypatch):  # noqa: F811
    con = duck.connect(warehouse)
    w = startsit.load_week(con, SEASON, WEEK, ["00-C", "00-D"])
    rules = from_sleeper({"rec": 1.0, "rec_yd": 0.1, "rec_td": 6})
    best = startsit.compare(w, rules).recommendation
    other = "00-D" if best == "00-C" else "00-C"
    team = w.inputs.filter(pl.col("key") == best)["team"][0]
    monkeypatch.setattr(gamestate, "status", lambda kickoff, now=None:
                        "final" if kickoff == "KO" else "upcoming")  # fmt: skip
    w.kickoffs = {team: "KO"}
    c = startsit.compare(w, rules, actual={best: 21.5})
    assert c.recommendation == other and not c.close_call
    assert any("game is over (21.5 points)" in n and "locked" in n for n in c.notes)
