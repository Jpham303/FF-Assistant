"""Time a short rest-of-season run, to diagnose slow ros_projections runs.

Projects the next `--weeks` weeks (default 2) exactly as the ros_projections asset does,
prints one line per week, and writes nothing. Run inside the daemon container:

    docker compose exec dagster-daemon python -m ffa.ros_timing

and again with thread caps to compare:

    docker compose exec -e POLARS_MAX_THREADS=4 -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 dagster-daemon python -m ffa.ros_timing
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=2)
    args = ap.parse_args()

    import polars as pl

    from ffa.config import settings
    from ffa.service import projections, ros
    from ffa.tracking import snapshots
    from ffa.warehouse import duck

    names = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    caps = {k: os.environ.get(k, "-") for k in names}
    print(f"CPUs {os.cpu_count()}, Polars threads {pl.thread_pool_size()}, caps {caps}")
    t0 = time.monotonic()
    con = duck.connect(settings.warehouse_dir)
    season = settings.current_season
    week = projections.upcoming_week(con, season)
    if week is None:
        raise SystemExit("No upcoming week")
    table = json.loads((Path(settings.models_dir) / "horizon.json").read_text())
    try:
        from ffa.db.session import session as db_session

        db = db_session()
        scoring = snapshots.leagues_scoring(db)
        db.close()
    except Exception:  # no app database: standard scoring only
        scoring = snapshots.leagues_scoring(None)
    q = lambda t: con.sql(f"select * from {t} where season <= {season}").pl()  # noqa: E731
    frames = ros.Frames(
        q("team_games"), q("player_games"),
        con.sql(f"select * from rosters where season >= {season - 1}").pl(),
        con.sql(f"select * from injuries where season >= {season - 1}").pl(),
        projections.latest_sleeper_status(con, season, week)[0],
    )  # fmt: skip
    models = projections.load_models(settings.models_dir)
    print(f"Loading data: {time.monotonic() - t0:.0f} s ({len(scoring)} scoring setups)")
    ros.project(frames, models, table, season, week, week + args.weeks - 1, scoring,
                progress=print)  # fmt: skip
    print(f"TOTAL {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
