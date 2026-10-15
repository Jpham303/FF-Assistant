"""Monte Carlo simulation of a week: team volume -> player stat lines -> fantasy points.

Each of `n` simulations plays every game once. Within a game everything is consistent:
targets and carries are split from the team's simulated volume, the quarterback's passing
is the sum of his receivers' production, extra points follow the team's touchdowns, and a
defense's sacks, takeaways and points allowed come from the opponent's simulated game.
Players inactive in a simulation (injury) have their share redistributed to active
teammates in proportion to the teammates' own shares.

Output: canonical stat lines per (player or defense, simulation), scored by the same
validated engine as real games, so any league's rules apply.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffa.model.inputs import WeekInputs
from ffa.model.sim_params import SimParams
from ffa.scoring import score
from ffa.scoring.rules import ScoringRules

OTHER_YPR = 9.5  # yards per catch for unmodeled receivers (league typical)
OTHER_CATCH = 0.65
SACK_YARDS = 7.0
MAX_FGA = 8
OTHER_SHARE = 0.02


def _underdispersed(rng, mu: np.ndarray, phi: float, size: int) -> np.ndarray:
    """Counts with mean mu and variance phi*mu (phi < 1): Binomial(m, mu/m), m = mu/(1-phi)."""
    m = np.maximum(np.ceil(mu / max(1 - phi, 1e-3)), 1).astype(int)
    p = np.clip(mu / m, 0, 1)
    return rng.binomial(m[:, None], p[:, None], size=(len(mu), size))


def _allocate(rng, total: np.ndarray, shares: np.ndarray, active: np.ndarray, conc: float):
    """Split `total[s]` across players (+ an 'other' pool) for each simulation s.

    shares: (P,) expected shares; active: (P, S) bool. The other pool keeps what the
    roster as a whole doesn't account for; an inactive player's share goes to active
    teammates in proportion to theirs. Dirichlet-multinomial with concentration `conc`.
    Returns (P + 1, S) counts, the last row being the other pool.
    """
    shares = np.clip(np.nan_to_num(shares), 0, None)
    # Shares are estimated player by player, so a team's sum drifts from 1 (0.65-1.35 in
    # practice). Fix the pool for players outside the active roster at OTHER_SHARE and
    # scale active players to fill the rest.
    other = OTHER_SHARE
    w = shares[:, None] * active  # (P, S)
    tot = w.sum(axis=0)
    scale = np.where(tot > 0, (1 - other) / np.maximum(tot, 1e-12), 0.0)
    w = w * scale
    w = np.vstack([w, np.full((1, w.shape[1]), other)])  # (P+1, S)
    w = w / w.sum(axis=0, keepdims=True)
    g = rng.gamma(np.maximum(conc * w, 1e-6))
    p = (g / g.sum(axis=0, keepdims=True)).T  # (S, P+1)
    p[:, :] = np.where(p.sum(axis=1, keepdims=True) > 0, p, 1.0 / p.shape[1])
    return rng.multinomial(total.astype(np.int64), p).T


def _choose(rng, weights: np.ndarray, active: np.ndarray) -> np.ndarray:
    """For each simulation, pick one active player with probability ~ weights. (P, S) mask."""
    w = np.clip(np.nan_to_num(weights), 1e-9, None)[:, None] * active
    tot = w.sum(axis=0)
    u = rng.random(w.shape[1]) * tot
    idx = (np.cumsum(w, axis=0) < u).sum(axis=0)
    out = np.zeros_like(active, dtype=bool)
    ok = tot > 0
    out[np.minimum(idx[ok], w.shape[0] - 1), np.nonzero(ok)[0]] = True
    return out


def simulate(inp: WeekInputs, sp: SimParams, n: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Simulated canonical stat lines: one row per (key, sim)."""
    rng = np.random.default_rng(seed)
    teams = inp.teams
    T = teams.height
    t_idx = {t: i for i, t in enumerate(teams["team"].to_list())}

    mu_plays = teams["mu_plays"].to_numpy()[:, None]
    sd_plays = teams["sd_plays"].to_numpy()[:, None]
    plays = np.clip(np.rint(rng.normal(mu_plays, sd_plays, (T, n))), 30, 100)
    dropbacks = rng.binomial(plays.astype(int), teams["p_pass"].to_numpy()[:, None])
    designed = plays - dropbacks
    sack_p = np.clip(sp.sack_rate * teams["off_sack_mult"].fill_null(1).to_numpy()
                     * teams["opp_def_sack_mult"].fill_null(1).to_numpy(), 0.01, 0.2)  # fmt: skip
    sacks = rng.binomial(dropbacks, sack_p[:, None])
    team_targets = rng.binomial(dropbacks, sp.targets_per_dropback)
    tds = _underdispersed(rng, teams["mu_off_td"].to_numpy(), sp.td_dispersion, n)
    pass_tds = rng.binomial(tds, teams["p_pass_td"].to_numpy()[:, None])
    rush_tds = tds - pass_tds
    fga = np.minimum(
        _underdispersed(rng, teams["mu_fg_att"].to_numpy(), sp.fg_dispersion, n), MAX_FGA
    )
    xp_att = rng.binomial(tds, 1 - sp.two_point_rate)
    xp_made = rng.binomial(xp_att, sp.xp_rate)
    dist_pmf, (a, b) = sp.kicking(inp.season)
    dists = np.array([int(d) for d in dist_pmf], dtype=float)
    probs = np.array(list(dist_pmf.values()))
    fg_dist = rng.choice(dists, size=(T, n, MAX_FGA), p=probs / probs.sum())
    attempted = np.arange(MAX_FGA)[None, None, :] < fga[:, :, None]
    p_make = 1 / (1 + np.exp(-(a + b * fg_dist)))
    made = attempted & (rng.random((T, n, MAX_FGA)) < p_make)

    lines: list[pl.DataFrame] = []
    team_rec_yd = np.zeros((T, n))
    team_rush_yd = np.zeros((T, n))
    team_ints = np.zeros((T, n))
    team_fum = np.zeros((T, n))

    players = inp.players
    for team, ti in t_idx.items():
        roster = players.filter(pl.col("team") == team)
        sk = roster.filter(pl.col("position") != "K")
        P = sk.height
        if P == 0:
            continue

        def col(c: str, frame: pl.DataFrame = sk) -> np.ndarray:
            return frame[c].fill_null(0).to_numpy().astype(float)

        active = rng.random((P, n)) < col("p_play")[:, None]
        pos = sk["position"].to_numpy()
        team_m = teams.row(ti, named=True)

        tgt = _allocate(rng, team_targets[ti], col("tgt_share_s") * col("m_tgt"), active,
                        sp.target_concentration)  # fmt: skip
        car = _allocate(rng, designed[ti], col("car_share_s"), active, sp.carry_concentration)
        p_tgt, o_tgt = tgt[:P], tgt[P]
        p_car = car[:P]

        is_qb = pos == "QB"
        qb_pick = np.zeros((P, n), dtype=bool)
        if is_qb.any():
            qb_pick[is_qb] = _choose(rng, col("db_share_s")[is_qb], active[is_qb])
        qb_db = qb_pick * dropbacks[ti]
        scrambles = rng.binomial(
            qb_db.astype(int), np.clip(col("scramble_rate_s"), 0, 0.5)[:, None]
        )
        rush_att = p_car + scrambles

        rec = rng.binomial(
            p_tgt, np.clip(col("catch_rate_s") * col("m_catch"), 0.05, 0.98)[:, None]
        )
        ypr = np.clip(col("ypr_s") * col("m_ypr"), 1.0, None)[:, None]
        rec_yd = np.where(rec > 0, rng.gamma(np.maximum(sp.rec_yd_shape * rec, 1e-9),
                                             ypr / sp.rec_yd_shape), 0.0)  # fmt: skip
        ypc = (col("ypc_s") * col("m_ypc"))[:, None]
        rush_yd = np.where(
            rush_att > 0, rng.normal(rush_att * ypc, sp.rush_yd_sd * np.sqrt(rush_att)), 0.0
        )

        o_rec = rng.binomial(o_tgt, OTHER_CATCH)
        o_yd = np.where(o_rec > 0, rng.gamma(np.maximum(sp.rec_yd_shape * o_rec, 1e-9),
                                             OTHER_YPR / sp.rec_yd_shape), 0.0)  # fmt: skip

        # Touchdowns go to players in proportion to realized opportunity times their
        # TD-per-opportunity propensity, so a player without a touch can't score.
        rec_prop = np.clip(
            np.nan_to_num(col("rec_td_share_s") / np.maximum(col("tgt_share_s"), 1e-3)), 0, 5
        )
        w_rec = np.vstack([p_tgt * rec_prop[:, None], o_tgt[None, :] * 1.0]).astype(float)
        rush_prop = np.clip(
            np.nan_to_num(col("rush_td_share_s") / np.maximum(col("car_share_s"), 1e-3)), 0, 5
        )
        w_rush = (rush_att * rush_prop[:, None]).astype(float)
        rec_td = _split(rng, pass_tds[ti], w_rec)[:P]
        rush_td = _split(rng, rush_tds[ti], w_rush)

        pass_att = np.rint(qb_pick * team_targets[ti] * sp.pass_att_per_target)
        ints = rng.binomial(pass_att.astype(int), np.clip(
            col("int_rate_s") * (team_m["opp_def_int_mult"] or 1.0), 0, 0.2)[:, None])  # fmt: skip
        pass_yd = qb_pick * (rec_yd.sum(axis=0) + o_yd)
        pass_td = qb_pick * pass_tds[ti]
        pass_cmp = qb_pick * (rec.sum(axis=0) + o_rec)
        touches = (rec + rush_att).astype(int)
        fum = rng.binomial(
            touches,
            np.clip(col("fum_rate_s") * (team_m["opp_def_fum_mult"] or 1.0), 0, 0.2)[:, None],
        )

        team_rec_yd[ti] = rec_yd.sum(axis=0) + o_yd
        team_rush_yd[ti] = rush_yd.sum(axis=0)
        team_ints[ti] = ints.sum(axis=0)
        team_fum[ti] = fum.sum(axis=0)

        stats = {
            "pass_yd": pass_yd, "pass_td": pass_td, "pass_int": ints, "pass_cmp": pass_cmp,
            "pass_att": pass_att, "pass_inc": pass_att - pass_cmp, "pass_sack": qb_pick * sacks[ti],
            "rush_att": rush_att, "rush_yd": rush_yd, "rush_td": rush_td, "rec": rec,
            "rec_tgt": p_tgt, "rec_yd": rec_yd, "rec_td": rec_td, "fum_lost": fum, "fum": fum,
        }  # fmt: skip
        lines.append(_frame(sk, stats, active, n))

        kick = roster.filter(pl.col("position") == "K")
        if kick.height:
            k = kick.sort("p_play", descending=True).head(1)
            m, a = made[ti], attempted[ti]
            d = fg_dist[ti]

            def band(lo: int, hi: int, d: np.ndarray = d) -> np.ndarray:
                return (d >= lo) & (d < hi)

            kstats = {
                "fgm": m.sum(-1), "fgmiss": (a & ~m).sum(-1),
                "fgm_yds": (d * m).sum(-1),
                "fgm_yds_over_30": (np.clip(d - 30, 0, None) * m).sum(-1),
                "xpm": xp_made[ti], "xpmiss": xp_att[ti] - xp_made[ti],
                **{f"fgm_{lo}_{hi - 1}": (m & band(lo, hi)).sum(-1)
                   for lo, hi in ((0, 20), (20, 30), (30, 40), (40, 50), (50, 60))},
                "fgm_60p": (m & (d >= 60)).sum(-1),
                **{f"fgmiss_{lo}_{hi - 1}": (a & ~m & band(lo, hi)).sum(-1)
                   for lo, hi in ((0, 20), (20, 30), (30, 40), (40, 50), (50, 60))},
                "fgmiss_60p": (a & ~m & (d >= 60)).sum(-1),
            }  # fmt: skip
            lines.append(_frame(k, {s: v[None, :] for s, v in kstats.items()}, None, n))

    # Defenses: built from the opponent's simulated game.
    dst = sp.dst
    for team, ti in t_idx.items():
        oi = t_idx.get(teams["opponent"][ti])
        if oi is None:
            continue
        sacks_d = sacks[oi]
        int_d = team_ints[oi]
        fum_rec = team_fum[oi]
        takeaways = (int_d + fum_rec).astype(int)
        opp_fg_made = made[oi].sum(-1)
        s = {
            "sack": sacks_d, "sack_yd": sacks_d * SACK_YARDS, "int": int_d, "fum_rec": fum_rec,
            "ff": rng.poisson(dst["ff_per_fum_rec"] * np.maximum(fum_rec, 0.3)),
            "def_td": rng.binomial(takeaways, dst["def_td_per_takeaway"]),
            "def_st_td": rng.poisson(dst["def_st_td_per_game"], n),
            "blk_kick": rng.poisson(dst["blk_kick_per_game"], n),
            "safe": rng.poisson(dst["safe_per_game"], n),
            "def_st_fum_rec": rng.poisson(dst["def_st_fum_rec_per_game"], n),
            "def_st_ff": rng.poisson(dst["def_st_ff_per_game"], n),
            "pts_allow": 6 * tds[oi] + xp_made[oi] + 3 * opp_fg_made,
            "yds_allow": team_rec_yd[oi] + team_rush_yd[oi] - sacks_d * SACK_YARDS,
        }  # fmt: skip
        key = pl.DataFrame({"key": [f"DST_{team}"], "name": [f"{team} D/ST"], "team": [team],
                            "position": ["DEF"]})  # fmt: skip
        lines.append(_frame(key, {k: np.asarray(v)[None, :] for k, v in s.items()}, None, n))

    return pl.concat(lines, how="diagonal_relaxed").fill_null(0.0)


def _split(rng, total: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multinomial split of `total[s]` with per-simulation weights (R, S) -> (R, S)."""
    w = np.clip(weights, 0, None)
    tot = w.sum(axis=0)
    p = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 0).T
    p[tot == 0, -1] = 1.0  # nobody touched it: give it to the last row (other / last back)
    return rng.multinomial(total.astype(np.int64), p).T


def _frame(meta: pl.DataFrame, stats: dict[str, np.ndarray], active, n: int) -> pl.DataFrame:
    """(rows, n) stat arrays -> long frame of (key, sim, stats). Inactive rows score 0."""
    R = meta.height
    data = {"sim": np.tile(np.arange(n, dtype=np.int32), R)}
    for name, arr in stats.items():
        a = np.broadcast_to(np.asarray(arr, dtype=np.float32), (R, n))
        data[name] = (a * active if active is not None else a).reshape(-1)
    if active is not None:
        data["active"] = active.reshape(-1)
    base = meta.select("key", "name", "team", "position")
    return (
        pl.DataFrame(data)
        .with_columns(
            *[pl.lit(None, pl.Utf8).alias(c) for c in ("key", "name", "team", "position")]
        )
        .with_columns(
            pl.Series("key", np.repeat(base["key"].to_numpy(), n)),
            pl.Series("name", np.repeat(base["name"].to_numpy(), n)),
            pl.Series("team", np.repeat(base["team"].to_numpy(), n)),
            pl.Series("position", np.repeat(base["position"].to_numpy(), n)),
        )
    )


def points(sim_lines: pl.DataFrame, rules: ScoringRules) -> pl.DataFrame:
    """Fantasy points per (key, sim) under a league's rules."""
    return score(sim_lines, rules).select("key", "name", "team", "position", "sim", "fpts")


def summarize(pts: pl.DataFrame) -> pl.DataFrame:
    return (
        pts.group_by("key", "name", "team", "position")
        .agg(
            pl.col("fpts").mean().alias("mean"),
            pl.col("fpts").median().alias("median"),
            pl.col("fpts").quantile(0.1).alias("p10"),
            pl.col("fpts").quantile(0.9).alias("p90"),
        )
        .sort("mean", descending=True)
    )
