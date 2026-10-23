"""Chance of winning the week's fantasy matchup, and the close-call tiebreaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ffa.db.models import RosterEntryRow, TeamRow
from ffa.scoring.rules import from_sleeper
from ffa.service import fantasy_matchup as fm
from ffa.service import startsit
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


def test_finished_games_count_with_awarded_points(warehouse, league, monkeypatch):  # noqa: F811
    db_, row, rules = league
    real = startsit.load_week
    ko = datetime(2026, 10, 15, 0, 15, tzinfo=UTC)

    def with_kickoffs(*a, **k):
        w = real(*a, **k)
        w.kickoffs = {"DAL": ko, "JAX": ko + timedelta(days=3)}
        return w

    monkeypatch.setattr(startsit, "load_week", with_kickoffs)
    con = duck.connect(warehouse)
    before = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules,
                         matchups=MATCHUPS, now=ko + timedelta(hours=1))  # fmt: skip
    after = fm.win_odds(con, db_, row, SEASON, WEEK, ["00-C", "00-D"], rules,
                        matchups=MATCHUPS, now=ko + timedelta(hours=5))  # fmt: skip
    assert before["final_games_counted"] == [] and after["final_games_counted"] == ["DAL"]
    c_before = next(c for c in before["candidates"] if c["key"] == "00-C")
    c_after = next(c for c in after["candidates"] if c["key"] == "00-C")
    assert c_after["my_expected"] > c_before["my_expected"]  # his 30 awarded points count


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
    monkeypatch.setattr(fm, "fetch_matchups", lambda *_a, **_k: MATCHUPS)
    r = client.post("/start-sit", json={"players": ["00-C", "00-D"], "league": "sleeper:1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["pairwise"][0]["median_gap"] is not None
    assert d["decision_drivers"] and {"factor", "points", "favors"} <= set(d["decision_drivers"][0])
    assert d["matchup"]["available"] and d["matchup"]["opponent"] == "Theirs"
