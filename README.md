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

## Scoring

The engine projects and scores **raw stats**, then applies each league's rules, so any
scoring system works without retraining anything.

```python
from ffa.scoring import stat_lines, score, breakdown, from_sleeper

rules = from_sleeper(league.scoring)  # platform settings -> canonical rules
pts = score(stat_lines(player_stats, pbp, schedules), rules)
why = breakdown(...)  # per-term points, for explanations
```

- **Canonical stats** follow Sleeper's vocabulary (`rec`, `pass_yd`, `fgm_yds`, `pts_allow`…).
  Players come from nflverse weekly stats; team defenses are built play by play, because
  the platforms score things team totals blur together: special-teams vs defensive
  fumbles, uncredited strip-sacks, blocked extra points, and points allowed that excludes
  return touchdowns and safeties the defense was not on the field for.
- **Rules** are linear terms, bands (points-allowed tiers, yardage bonuses) and
  position-specific terms (TE premium). Platform settings we can't compute are listed in
  `rules.unsupported` or `rules.approximate`, never silently dropped.

### Verification

| Check | Result |
| --- | --- |
| Our points vs **Sleeper's awarded points**, a real 12-team league, weeks 1–4 of 2026 | **683 / 683 player-weeks exact** |
| Offense vs nflverse's own fantasy points, every player-week since 2015 | 204,077 / 204,077 (return fumbles aside, which nflverse omits) |
| Play-by-play scoring events vs final scores, every game since 2015 | 6,184 / 6,184 team-games exact |

Re-run the Sleeper comparison any week; it exits non-zero on any mismatch and prints our
per-term breakdown for it:

```bash
uv run ffa check-scoring sleeper <league_id>
```

## Projections

A simulation model, not a single-number guess. Each week, every game is simulated 2,000
times from data available before kickoff:

1. **Team volume** from the betting lines plus team tendencies: plays, pass rate, TDs, FGs
2. **Player shares** (targets, carries, dropbacks, red-zone looks) and **efficiency**,
   recency-weighted and shrunk toward priors fitted from data
3. **Matchups**: where each defense gives up production relative to expectation
4. **Availability** from injury status, final practice and role
5. Players' stat lines from their team's simulated game (the QB's passing is the sum of
   his receivers'; a defense's points allowed is the opponent's simulated score), scored
   with the league's rules

**2025 holdout** (model frozen before it was run; details in
[docs/evaluation_2025.md](docs/evaluation_2025.md)):

| | Model | Season average | Opportunity average |
| --- | --- | --- | --- |
| Start/sit, all startable pairs | **62.3%** | 58.9% | 61.1% |
| Start/sit, close calls | **58.9%** | 53.6% | 56.0% |
| Projection error | lowest at every position | | |

## API

The weekly projections job (`weekly_projections`, hourly at :25 after the injury snapshot)
simulates the upcoming week and stores league-neutral stat lines, so any league's scoring
is applied at request time. Interactive docs at http://localhost:8000/docs.

| Endpoint | What it returns |
| --- | --- |
| `GET /week` | current season, upcoming week, whether projections exist |
| `GET /players/search?q=` | player keys by name, typos allowed; `match` (1.0 = exact) and `projected` (false on a bye). A bye-week player is reported as not playing, never swapped for a similar name |
| `GET /projections?keys=` | mean, median, 10th–90th percentile, play probability, expected stats, points per scoring rule, and plain-language drivers each carrying its number |
| `POST /start-sit` | 2–6 players: recommendation, P(each outscores each other) from simulations that share each game, close-call and injury notes |
| `GET /leagues/{id}/my-team` | best lineup for your roster under the league's slots and scoring; bench; who isn't playing |

Pass `league` (e.g. `sleeper:<league_id>`) to score with that league's settings; without
it, `SLEEPER_LEAGUE_ID` or standard half-PPR is used.

## Develop

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest
```

## Status

Phase 0 (data, leagues, scoring) done. Phase 1 (projections, start/sit API, MCP server) in progress.
