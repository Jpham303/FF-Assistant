"""Scoring rules in canonical form, and the translation from each platform's settings.

A league's rules are three kinds of term:
    linear     points per unit of a stat          rec: 0.5
    band       points when a stat falls in [lo,hi) pts_allow 7-13: +4
    position   points per unit, one position only bonus_rec_te: +0.5 per TE catch

Platform settings we cannot compute are never silently dropped: nonzero ones land in
`unsupported` (missing from our totals) or `approximate` (computed, with a known caveat).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ffa.scoring.stats import DST_STAT_NAMES, PLAYER_STAT_NAMES

INF = float("inf")


class Band(BaseModel):
    stat: str
    lo: float
    hi: float = INF  # exclusive
    points: float
    positions: frozenset[str] | None = None  # None = every row


class PositionTerm(BaseModel):
    stat: str
    position: str
    points: float


class ScoringRules(BaseModel):
    source: str
    linear: dict[str, float] = Field(default_factory=dict)
    bands: list[Band] = Field(default_factory=list)
    position_terms: list[PositionTerm] = Field(default_factory=list)
    unsupported: dict[str, str] = Field(default_factory=dict)  # platform key -> why
    approximate: dict[str, str] = Field(default_factory=dict)  # platform key -> caveat

    def stats_used(self) -> set[str]:
        return (
            set(self.linear) | {b.stat for b in self.bands} | {t.stat for t in self.position_terms}
        )


DEF_ONLY = frozenset({"DEF"})
OFFENSE = frozenset({"QB", "RB", "WR", "TE", "K", "FB"})

# ---------------------------------------------------------------- Sleeper

# Sleeper key -> canonical stat, for keys that are simple points-per-unit.
_SLEEPER_LINEAR = {
    k: k for k in [*PLAYER_STAT_NAMES, *DST_STAT_NAMES] if k not in {"rec_tgt", "rush_rec_yd"}
}
# Keys that stand for the sum of several canonical stats.
_SLEEPER_COMPOSITE = {
    "fgm_50p": ["fgm_50_59", "fgm_60p"],
    "fgmiss_50p": ["fgmiss_50_59", "fgmiss_60p"],
}


def _r(lo: float, hi: float = INF) -> tuple[float, float]:
    return lo, hi


# Threshold bonuses. Sleeper's 100- and 200-yard game bonuses are separate brackets.
_SLEEPER_BANDS: dict[str, tuple[str, tuple[float, float], frozenset[str] | None]] = {
    "pts_allow_0": ("pts_allow", _r(0, 1), DEF_ONLY),
    "pts_allow_1_6": ("pts_allow", _r(1, 7), DEF_ONLY),
    "pts_allow_7_13": ("pts_allow", _r(7, 14), DEF_ONLY),
    "pts_allow_14_20": ("pts_allow", _r(14, 21), DEF_ONLY),
    "pts_allow_21_27": ("pts_allow", _r(21, 28), DEF_ONLY),
    "pts_allow_28_34": ("pts_allow", _r(28, 35), DEF_ONLY),
    "pts_allow_35p": ("pts_allow", _r(35), DEF_ONLY),
    "yds_allow_0_100": ("yds_allow", _r(-INF, 100), DEF_ONLY),
    "yds_allow_100_199": ("yds_allow", _r(100, 200), DEF_ONLY),
    "yds_allow_200_299": ("yds_allow", _r(200, 300), DEF_ONLY),
    "yds_allow_300_349": ("yds_allow", _r(300, 350), DEF_ONLY),
    "yds_allow_350_399": ("yds_allow", _r(350, 400), DEF_ONLY),
    "yds_allow_400_449": ("yds_allow", _r(400, 450), DEF_ONLY),
    "yds_allow_450_499": ("yds_allow", _r(450, 500), DEF_ONLY),
    "yds_allow_500_549": ("yds_allow", _r(500, 550), DEF_ONLY),
    "yds_allow_550p": ("yds_allow", _r(550), DEF_ONLY),
    "bonus_pass_yd_300": ("pass_yd", _r(300, 400), OFFENSE),
    "bonus_pass_yd_400": ("pass_yd", _r(400), OFFENSE),
    "bonus_rush_yd_100": ("rush_yd", _r(100, 200), OFFENSE),
    "bonus_rush_yd_200": ("rush_yd", _r(200), OFFENSE),
    "bonus_rec_yd_100": ("rec_yd", _r(100, 200), OFFENSE),
    "bonus_rec_yd_200": ("rec_yd", _r(200), OFFENSE),
    "bonus_rush_rec_yd_100": ("rush_rec_yd", _r(100, 200), OFFENSE),
    "bonus_rush_rec_yd_200": ("rush_rec_yd", _r(200), OFFENSE),
    "bonus_rush_att_20": ("rush_att", _r(20), OFFENSE),
    "bonus_pass_cmp_25": ("pass_cmp", _r(25), OFFENSE),
}
_SLEEPER_POSITION = {
    "bonus_rec_rb": ("rec", "RB"),
    "bonus_rec_wr": ("rec", "WR"),
    "bonus_rec_te": ("rec", "TE"),
}
# Nonzero values for these change totals but cannot be computed from weekly stats.
_SLEEPER_NEEDS_PBP = "needs play-by-play data (not yet supported)"
_SLEEPER_UNSUPPORTED = {
    **{
        k: _SLEEPER_NEEDS_PBP
        for k in [
            "pass_td_40p", "pass_td_50p", "rush_td_40p", "rush_td_50p", "rec_td_40p",
            "rec_td_50p", "rec_0_4", "rec_5_9", "rec_10_19", "rec_20_29", "rec_30_39",
            "rec_40p", "rush_40p", "pass_cmp_40p", "pass_int_td", "def_3_and_out",
            "def_4_and_stop", "def_forced_punts", "bonus_def_int_td_50p",
            "bonus_def_fum_td_50p", "int_ret_yd", "fum_ret_yd", "blk_kick_ret_yd",
            "fg_ret_yd", "def_2pt",
        ]
    },
    **{
        k: "individual defensive players (IDP) are not supported"
        for k in [
            "tkl", "tkl_solo", "tkl_ast", "bonus_tkl_10p", "bonus_sack_2p", "idp_sack_yd",
            "idp_pass_def_3p", "def_st_tkl_solo", "st_tkl_solo",
        ]
    },
    **{
        k: "first-down bonus by position is not supported yet"
        for k in ["bonus_fd_qb", "bonus_fd_rb", "bonus_fd_wr", "bonus_fd_te"]
    },
    "st_ff": "forced fumbles by offensive players on special teams are not tracked (rare)",
    "st_fum_rec": "fumble recoveries by offensive players on special teams are not tracked (rare)",
}  # fmt: skip
_SLEEPER_APPROXIMATE = {
    "def_st_fum_rec": "special-teams recoveries are scored at the fum_rec value",
    "def_st_ff": "special-teams forced fumbles are scored at the ff value",
}


def from_sleeper(settings: dict[str, float]) -> ScoringRules:
    rules = ScoringRules(source="sleeper")
    for key, raw in settings.items():
        pts = float(raw)
        if pts == 0:
            continue
        if key in _SLEEPER_BANDS:
            stat, (lo, hi), positions = _SLEEPER_BANDS[key]
            rules.bands.append(Band(stat=stat, lo=lo, hi=hi, points=pts, positions=positions))
        elif key in _SLEEPER_POSITION:
            stat, pos = _SLEEPER_POSITION[key]
            rules.position_terms.append(PositionTerm(stat=stat, position=pos, points=pts))
        elif key in _SLEEPER_COMPOSITE:
            for stat in _SLEEPER_COMPOSITE[key]:
                rules.linear[stat] = rules.linear.get(stat, 0.0) + pts
        elif key in _SLEEPER_LINEAR:
            rules.linear[key] = rules.linear.get(key, 0.0) + pts
        elif key in _SLEEPER_APPROXIMATE:
            rules.approximate[key] = _SLEEPER_APPROXIMATE[key]
        else:
            rules.unsupported[key] = _SLEEPER_UNSUPPORTED.get(key, "unknown Sleeper setting")
    return rules


# ---------------------------------------------------------------- reference rules


def nflverse_standard(ppr: float = 1.0) -> ScoringRules:
    """The rules behind nflverse's own fantasy_points / fantasy_points_ppr columns.
    Used to verify the engine against an independent calculation."""
    return ScoringRules(
        source=f"nflverse standard (ppr={ppr})",
        linear={
            "pass_yd": 0.04, "pass_td": 4, "pass_int": -2, "pass_2pt": 2,
            "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
            "rec": ppr, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
            "st_td": 6, "fum_lost": -2,
        },
    )  # fmt: skip
