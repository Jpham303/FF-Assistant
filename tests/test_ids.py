import polars as pl
import pytest

from ffa.ingest import ids


@pytest.fixture
def crosswalk() -> pl.DataFrame:
    players = pl.DataFrame(
        {
            "gsis_id": ["G-ROOK", "G-VET", "G-MHJ", "G-MHS", "G-KW3", "G-MW-OLD", "G-MW-NEW"],
            "name": [
                "Tetairoa McMillan", "Davante Adams", "Marvin Harrison", "Marvin Harrison",
                "Kenneth Walker III", "Mike Williams", "Mike Williams",
            ],
            "position": ["WR", "WR", "WR", "WR", "RB", "WR", "WR"],
            "team": ["CAR", "LAR", "ARI", "IND", "SEA", "TB", "NYJ"],
            "last_season": [2026, 2026, 2026, 2008, 2026, 2013, 2025],
        }
    )  # fmt: skip
    rosters = pl.DataFrame(
        {
            "gsis_id": ["G-ROOK", "G-VET", "G-VET"],
            "sleeper_id": ["12526", "2133", "2133"],
            "yahoo_id": [None, "26650", "26650"],
            "season": [2026, 2025, 2026],
            "week": [4, 18, 4],
        }
    )
    dp = pl.DataFrame(
        {
            "gsis_id": ["G-VET", "G-KW3"],
            "sleeper_id": ["2133", "8151"],
            "yahoo_id": ["99999", "33100"],  # G-VET disagrees with rosters on purpose
            "name": ["Davante Adams", "Kenneth Walker"],
            "position": ["WR", "RB"],
            "team": ["LAR", "SEA"],
        }
    )
    return ids.build_crosswalk(players, rosters, dp)


def test_priority_and_conflicts(crosswalk):
    vet = crosswalk.filter(pl.col("gsis_id") == "G-VET").row(0, named=True)
    assert vet["yahoo_id"] == "26650"  # nflverse rosters beat DynastyProcess
    assert vet["yahoo_id_source"] == "rosters"
    assert vet["yahoo_id_conflict"] is True
    kw3 = crosswalk.filter(pl.col("gsis_id") == "G-KW3").row(0, named=True)
    assert kw3["sleeper_id"] == "8151" and kw3["sleeper_id_source"] == "dp"


def test_team_aliases_normalized(crosswalk):
    assert crosswalk.filter(pl.col("gsis_id") == "G-VET")["team"][0] == "LA"


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("D.K. Metcalf", "dkmetcalf"),
        ("Kenneth Walker III", "kennethwalker"),
        ("Amon-Ra St. Brown", "amonrastbrown"),
        ("Odell Beckham Jr.", "odellbeckham"),
        ("Gardner Minshew II", "gardnerminshew"),
        ("José Ramírez", "joseramirez"),
    ],
)
def test_normalize_name(raw, key):
    assert ids.normalize_name(raw) == key


def test_resolve_by_platform_id(crosswalk):
    r = ids.Resolver(crosswalk)
    assert r.resolve("sleeper", "12526") == ids.Match("G-ROOK", "id")
    assert r.resolve("yahoo", "26650") == ids.Match("G-VET", "id")


def test_rookie_without_yahoo_id_resolves_by_name(crosswalk):
    m = ids.Resolver(crosswalk).resolve(
        "yahoo", "41234", name="Tetairoa McMillan", position="WR", team="Car"
    )
    assert m == ids.Match("G-ROOK", "name")


def test_same_name_resolved_by_team(crosswalk):
    r = ids.Resolver(crosswalk)
    assert (
        r.resolve("yahoo", None, name="Marvin Harrison", position="WR", team="ARI").key == "G-MHJ"
    )
    no_team = r.resolve("yahoo", None, name="Marvin Harrison Jr.", position="WR")
    # Sr. retired long ago, so the active player is the only plausible match.
    assert no_team.key == "G-MHJ"


def test_ambiguous_name_is_not_guessed():
    players = pl.DataFrame(
        {
            "gsis_id": ["A", "B"],
            "name": ["Josh Johnson", "Josh Johnson"],
            "position": ["QB", "QB"],
            "team": ["SF", "BAL"],
            "last_season": [2025, 2026],
        }
    )
    empty = pl.DataFrame(
        schema={"gsis_id": pl.Utf8, "sleeper_id": pl.Utf8, "yahoo_id": pl.Utf8,
                "season": pl.Int32, "week": pl.Int32}
    )  # fmt: skip
    dp = empty.drop("season", "week").with_columns(
        name=pl.lit(None, pl.Utf8), position=pl.lit(None, pl.Utf8), team=pl.lit(None, pl.Utf8)
    )
    r = ids.Resolver(ids.build_crosswalk(players, empty, dp))
    assert r.resolve("yahoo", None, name="Josh Johnson", position="QB") == ids.Match(
        None, "ambiguous"
    )


def test_defense_keys():
    r = ids.Resolver(
        pl.DataFrame(
            schema={c: pl.Utf8 for c in
                    ["gsis_id", "sleeper_id", "yahoo_id", "name_key", "position", "team"]}
        ).with_columns(last_season=pl.lit(None, pl.Int32))
    )  # fmt: skip
    assert r.resolve("sleeper", "LAR", position="DEF", team="LAR") == ids.Match("DST_LA", "dst")
    assert r.resolve("yahoo", "100003", position="DEF", team="Jac").key == "DST_JAX"
