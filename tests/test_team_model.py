import numpy as np
import polars as pl
import pytest

from ffa.model import team
from ffa.model.asof import Decay


def synthetic(n_weeks: int = 17, seasons=(2016, 2017, 2018), seed: int = 0) -> pl.DataFrame:
    """Two-team league where plays and pass rate follow known rules."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, n_weeks + 1):
            spread = rng.normal(0, 6)
            total = rng.normal(45, 4)
            for team_code, opp, sign, pace in (("FAST", "SLOW", 1, 66), ("SLOW", "FAST", -1, 58)):
                sf = sign * spread
                plays = pace + 0.1 * sf + rng.normal(0, 2)
                p = 1 / (1 + np.exp(-(0.4 - 0.03 * sf)))
                drop = rng.binomial(int(plays), p)
                implied = (total + sf) / 2
                td = rng.poisson(np.exp(-3 + 1.2 * np.log(max(implied, 3))))
                ptd = rng.binomial(td, 0.6)
                rows.append(dict(
                    game_id=f"{season}_{week}", season=season, week=week, game_type="REG",
                    team=team_code, opponent=opp, spread_for=sf, total_line=total,
                    implied=implied, plays=float(int(plays)), dropbacks=float(drop),
                    pass_rate_oe=drop / int(plays) - 0.6, pass_td=float(ptd),
                    rush_td=float(td - ptd), fg_att=float(rng.poisson(1.8)),
                ))  # fmt: skip
    tg = pl.DataFrame(rows)
    opp = tg.select(
        "game_id", pl.col("team").alias("opponent"),
        pl.col("plays").alias("opp_plays"), pl.col("pass_rate_oe").alias("opp_pass_rate_oe"),
    )  # fmt: skip
    return tg.join(opp, on=["game_id", "opponent"])


def test_fit_recovers_known_structure(tmp_path):
    k = {n: 4.0 for n in team.TENDENCIES}
    m = team.fit(synthetic(), [2016, 2017], [2018], Decay(8, 4), k)
    # Pace is a team trait here; the model must learn to use it.
    assert m.coef["plays"]["pace_s"] > 0.5
    assert m.metrics["plays"]["valid_loss"] < m.metrics["plays"]["lines_only_loss"]
    assert m.coef["pass_rate"]["spread_for"] == pytest.approx(-0.03, abs=0.015)
    # 68 training rows cannot pin the TD slope precisely (SE ~0.5); check direction only.
    assert m.coef["off_td"]["log_implied"] > 0.3

    m.save(tmp_path / "team.json")
    m2 = team.TeamModel.load(tmp_path / "team.json")
    assert m2.coef == m.coef and m2.decay == m.decay
    pred = m2.predict(m2.features(synthetic(seasons=(2019,))))
    fast = pred.filter((pl.col("team") == "FAST") & (pl.col("week") > 8))
    slow = pred.filter((pl.col("team") == "SLOW") & (pl.col("week") > 8))
    assert fast["mu_plays"].mean() > slow["mu_plays"].mean() + 4
    assert pred["p_pass"].is_between(0, 1).all() and (pred["mu_off_td"] > 0).all()
