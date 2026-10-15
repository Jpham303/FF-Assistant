import polars as pl
import pytest

from ffa.model import player
from ffa.model.asof import Decay

ZERO = {n: 0.0 for pair in player.RATIOS.values() for n in pair if not n.startswith("team_")}
ZERO |= {"team_targets": 0.0, "team_designed_runs": 0.0, "team_dropbacks": 0.0,
         "team_pass_td": 0.0, "team_rush_td": 0.0, "rush_att": 0.0, "rec": 0.0}  # fmt: skip


def games(key: str, n: int, **per_game) -> list[dict]:
    rows = []
    for w in range(1, n + 1):
        row = ZERO | {"key": key, "season": 2024, "week": w, "game_id": f"g{w}",
                      "team": "CIN", "position": "WR", "team_targets": 35.0,
                      "team_pass_td": 2.0, "team_designed_runs": 25.0,
                      "team_dropbacks": 38.0}  # fmt: skip
        rows.append(row | per_game)
    return rows


def model() -> player.PlayerModel:
    m = player.PlayerModel(
        decay={n: Decay(8, 4) for n in player.RATIOS},
        k={n: 1.0 for n in player.RATIOS} | {"tgt_share": 35.0, "rec_td_share": 4.0},
        td_rz_weight={"rec_td_share": 0.0, "rush_td_share": 0.0},
        position_prior={n: {"WR": 0.1, "RB": 0.1, "TE": 0.1, "QB": 0.1}
                        for n in [*player.SHARES, *player.POSITION_PRIORS]},
        depth_prior={"catch_rate": {"WR": [0.8, -0.02]}, "ypr": {"WR": [6.0, 0.6]}},
    )  # fmt: skip
    m.position_prior["tgt_share"]["WR"] = 0.15
    return m


def frame() -> pl.DataFrame:
    # A rookie with 1 game at a 40% share; a veteran with 10 games at 25%; both score
    # 2 of their team's TDs (100% TD share) in their last game only.
    rookie = games("rookie", 2, targets=14.0, rec=10.0, rec_yd=120.0, air_yards=140.0)
    vet = games("vet", 11, targets=8.75, rec=6.0, rec_yd=70.0, air_yards=90.0)
    return pl.DataFrame(rookie + vet).with_columns(rz_targets=0.0, rz_carries=0.0)


def test_little_evidence_shrinks_hard_lots_shrinks_little():
    m = model()
    est = m.estimates(m.history(frame()))
    last = est.group_by("key").agg(pl.all().sort_by("week").last())
    r = last.filter(pl.col("key") == "rookie").row(0, named=True)
    v = last.filter(pl.col("key") == "vet").row(0, named=True)
    # One game of evidence (35 targets, decayed one game at an 8-game half-life) against
    # a prior worth 35 targets: just under halfway from the prior to the observed 40%.
    w = 0.5 ** (1 / 8) * 35
    assert r["tgt_share_s"] == pytest.approx((0.4 * w + 0.15 * 35) / (w + 35))
    assert abs(v["tgt_share_s"] - 0.25) < abs(r["tgt_share_s"] - 0.4)


def test_td_share_prior_is_target_share():
    m = model()
    est = m.estimates(m.history(frame()))
    v = est.filter((pl.col("key") == "vet") & (pl.col("week") == 11)).row(0, named=True)
    # No TDs scored before week 11, so the TD-share estimate sits below the target share
    # it shrinks toward, pulled down by the zero-TD history.
    assert 0 < v["rec_td_share_s"] < v["tgt_share_s"]


def test_depth_prior_for_efficiency():
    m = model()
    est = m.estimates(m.history(frame()))
    r = est.filter((pl.col("key") == "rookie") & (pl.col("week") == 1)).row(0, named=True)
    # No history at all: adot falls back to the position prior, catch rate and yards
    # per catch to the depth-of-target line evaluated there.
    assert r["adot_s"] == pytest.approx(0.1)
    assert r["catch_rate_s"] == pytest.approx(0.8 - 0.02 * 0.1)
    assert r["ypr_s"] == pytest.approx(6.0 + 0.6 * 0.1)


def test_save_load_roundtrip(tmp_path):
    m = model()
    m.save(tmp_path / "p.json")
    m2 = player.PlayerModel.load(tmp_path / "p.json")
    assert m2.decay == m.decay and m2.k == m.k and m2.depth_prior == m.depth_prior
