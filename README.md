# FF Assistant

A fantasy football assistant built on a backtested statistical engine, not just a chatbot.

- **Projections:** probabilistic weekly projections of raw stats (targets, carries, yards, TDs), converted to points under any league's scoring rules
- **Matchups:** opponent-adjusted run and pass defense, game script from Vegas lines, pace
- **Start/sit:** P(player A outscores player B) with the drivers behind it
- **Trades:** roster-aware rest-of-season value and trade target search
- **Interfaces:** FastAPI backend, Streamlit app, and an MCP server so AI agents can query the same API

Leagues: Sleeper and Yahoo, with custom scoring read from each platform.

## Architecture

| Layer | Tech |
| --- | --- |
| Pipeline | Dagster |
| Analytics warehouse | Parquet + DuckDB |
| App database | Postgres |
| API | FastAPI |
| Frontend | Streamlit |
| AI interface | MCP server over the API |

Data comes from [nflverse](https://github.com/nflverse) and the official Sleeper and Yahoo APIs. No scraping.

## Run locally

```bash
cp .env.example .env        # fill in values
docker compose up --build
```

- API: http://localhost:8000/health
- Dagster: http://localhost:3000

## Load data

The first time, backfill every season (about 30 seconds, ~100 MB):

- Dagster UI → **Assets** → group `nflverse` → **Materialize all** → select all partitions

After that, two schedules keep the current season fresh (turn them on in the UI):

| Schedule | When | What |
| --- | --- | --- |
| `nightly_current_season` | 2:15am Central | all datasets for the current season |
| `hourly_injuries` | :05 every hour | injury reports and rosters, kept as timestamped snapshots |
| `nightly_dims` | 2:45am Central | player ID crosswalk (nflverse ↔ Sleeper ↔ Yahoo) |

### Player IDs

Every player is keyed by nflverse's `gsis_id`; team defenses use `DST_<TEAM>`. The
`player_ids` table maps Sleeper and Yahoo IDs onto it from three sources (Sleeper's player
DB, nflverse rosters, DynastyProcess), recording which source supplied each ID and flagging
disagreements. Players with no platform ID — usually recent rookies on Yahoo — are matched
at import by normalized name and position, then team. Ambiguous names are never guessed.

Data lands in `data/warehouse/{dataset}/season=YYYY/week=WW/`. Query it with
`ffa.warehouse.duck.connect()`; use `duck.as_of()` to see data as it stood at a past moment.

## Connect your league

```bash
uv run ffa leagues sleeper <your_username>          # find your league ID
uv run ffa sync sleeper <league_id> --dry-run       # preview the import and match report
```

Then set `SLEEPER_USERNAME` and `SLEEPER_LEAGUE_ID` in `.env`. The `sleeper_league` asset
syncs settings, teams and rosters into Postgres hourly (`hourly_leagues` schedule), and
logs any player it could not match.

| Table | Contents |
| --- | --- |
| `leagues` | lineup slots, bench/IR/taxi sizes, raw scoring settings |
| `teams` | owner, record, points for, which team is yours |
| `roster_entries` | every rostered player: canonical key, how it was matched, starter/bench/IR slot |

Database schema changes go through Alembic (`migrations/`); the `migrate` service applies
them on `docker compose up`.

## Develop

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest
```

## Status

Phase 0 (foundation) in progress.
