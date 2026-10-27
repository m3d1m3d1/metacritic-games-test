# Metacritic games catalog service

An async Python service that collects new games from Metacritic, stores platform-level
ratings, and presents a searchable catalog with separate critic and user review summaries.
It also includes background retries, similar-game recommendations, optional YouTube
enrichment, and live worker monitoring.

```mermaid
flowchart LR
    MC[Metacritic] --> W[Scheduled worker]
    W --> DB[(PostgreSQL + pgvector)]
    W --> LLM[Groq]
    W -. optional .-> YT[YouTube captions]
    B[Browser] <--> API[FastAPI catalog and monitor]
    API <--> DB
    W --> DB
```

## Local setup
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up --build -d                         # database, migrations, web, worker

python -m scripts.probe                              # check Metacritic access + parsers on live pages
python -m pytest                                     # parser tests (no DB needed)
```

The web UI is available at `http://localhost:8000/`; Docker Compose applies Alembic migrations
before starting the web and worker containers. The worker's FastEmbed cache is stored in a
named volume so the embedding model is not downloaded again after container recreation.
The web port and PostgreSQL port bind to localhost by default. A reverse proxy can route to
the `web:8000` service on the Compose network.

For a server deployment, set a unique `POSTGRES_PASSWORD`, `POSTGRES_USER`, and `POSTGRES_DB`
in `.env` before starting Compose. Set `GROQ_API_KEY` there to enable generated review
summaries and letsplay conclusions; configure `METACRITIC_PROXY` if the host requires a proxy.
Do not commit `.env`; `.gitignore` and `.dockerignore` exclude it.
For a shared demo, set both `DEMO_AUTH_USERNAME` and `DEMO_AUTH_PASSWORD` in `.env`.
The application then requires HTTP Basic Auth on the catalog, monitoring pages, API, and live
event stream. Use a password with at least 16 characters and share it with reviewers separately
from the public URL. `/health` is intentionally unauthenticated for the container healthcheck.
Leaving both settings empty keeps the local development server open.
For a faster catalog-focused demo, set `YOUTUBE_ENRICHMENT_ENABLED=false`; the default is
`true` so normal crawls still include the optional YouTube feature.

GitHub Actions runs the Python tests against PostgreSQL 16 with pgvector, validates the
Compose configuration, and builds the application image on pushes and pull requests.
To inspect or stop the local stack, use `docker compose logs -f web worker` and
`docker compose down`; avoid `docker compose down -v` unless the database data should be
deleted.

DB tests need a throw-away database with the pgvector extension available:
```bash
docker compose exec db createdb -U games games_test
TEST_DATABASE_URL=postgresql+asyncpg://games:games@localhost:5432/games_test python -m pytest
```
(PowerShell: `$env:TEST_DATABASE_URL="postgresql+asyncpg://games:games@localhost:5432/games_test"`)

For PowerShell, use these commands to run the PostgreSQL tests:
```powershell
docker compose up -d db
docker compose exec db createdb -U games games_test
$env:TEST_DATABASE_URL = "postgresql+asyncpg://games:games@localhost:5432/games_test"
python -m pytest -q
```

Parser tests use small synthetic HTML examples and do not require saved Metacritic pages.

`games_test` must be a disposable database: the integration-test fixture drops and recreates
its schema. Run `createdb` once; if it reports that the database already exists, continue
with the remaining commands. The environment variable lasts only for the current PowerShell
window, so set it again in a new terminal.

## Layout
- `app/scraper/` - HTTP client and parsers (Nuxt payload based)
- `app/db/` - models, upsert/crawl repository
- `app/worker.py` - hourly scheduled and manual crawl worker
- `app/main.py` - FastAPI monitoring page and crawl-run API
- `alembic/` - migrations

## API and monitoring
Run the web application and crawler as separate processes so that HTTP requests remain
responsive while scraping:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m app.worker
```

Open `http://localhost:8000/` for the game catalog. Search by title, filter by platform,
sort by Metascore/Userscore/title/release date, and open a game's detail page. Similar games
are ranked by embedding distance when vectors exist; until then, matching genres are used.
Game cards keep the active catalog filters and sorting in their links. Detail pages have
fixed previous/next arrows that follow that same filtered and sorted result set, including
across catalog pages; the catalog return link preserves the current query.
Cover images are fetched server-side from Metacritic through `METACRITIC_PROXY` when configured,
and served from the same origin as the app. A bounded in-memory cache and browser ETags avoid
re-downloading unchanged covers; browsers do not need direct access to Metacritic's image CDN.
Only stored HTTPS image URLs on Metacritic hosts are accepted by the image endpoint.
The detail page includes platform scores, description, video link, and available critic/user
review summaries. The crawler also searches the first 10 YouTube gameplay/letsplay results,
selects the result with the highest reported view count, and uses available English or Russian
captions to generate a short conclusion. If search, captions, or the LLM are unavailable, the
game crawl still succeeds and the letsplay section is omitted. No audio is downloaded or
transcribed locally.
The detail page's language switch keeps the Metacritic/source text intact and can translate
the description, critic/user review summaries, and YouTube conclusion into Russian on demand.
Translations are cached in Postgres and regenerated when their source text changes. A provider
failure leaves the original visible and is reported beside the language controls.

Game embeddings are generated locally with `BAAI/bge-small-en-v1.5` (384 dimensions) from the
title, genres, and description. The model is downloaded on first use and cached by FastEmbed.
The embedding is recomputed only when that source text changes; similar-game results use
pgvector cosine distance when both games have vectors, with genre overlap as a fallback.

Open `http://localhost:8000/monitor` for the monitoring page. It shows recent crawl runs and
their per-game statuses, live worker heartbeats, current worker state, and processed-game
counters. The expandable live-log panel shows crawl and summary-queue events, and the summary
queue counters distinguish pending, due, and retrying summaries. Server-Sent Events refresh
the status and logs every two seconds without reloading the page. Worker events are retained
in Postgres for 30 days. A prominent catalog button returns to the game list.
Workers are considered offline if their heartbeat is older than 20 seconds. The button
requests a manual run. The JSON endpoints are:
- `GET /api/crawl-runs?limit=20` - recent runs and their items
- `POST /api/crawl-runs` - request a manual run (returns HTTP 202)
- `GET /api/monitor/status` - current worker and crawl status
- `GET /api/monitor/events` - live worker, crawl, summary-queue, and log stream (`text/event-stream`)
- `POST /api/games/{slug}/translation` - generate/cache Russian text on demand; returns
  HTTP 503 with `Retry-After` when the translation provider is rate-limited

The web process and scheduler worker must use the same `DATABASE_URL`. Manual requests are
stored as `queued` runs and picked up by the worker (default poll interval: 5 seconds), so
they survive a web-process restart. A Postgres advisory lock prevents scheduled and manual
crawls from running concurrently, including across separate processes.
Apply the latest Alembic migration (`alembic upgrade head`) to create worker monitoring,
letsplay, and cached translation columns before starting web and worker processes. Docker
Compose runs this migration once via the `migrate` service before either application service
starts.

Review summaries use an OpenAI-compatible Chat Completions API. The default provider is Groq;
set `GROQ_API_KEY` in `.env` to enable generated review summaries and letsplay conclusions.
The default endpoint is `https://api.groq.com/openai/v1` and the default model is
`openai/gpt-oss-120b`; configure `LLM_API_BASE_URL` and `SUMMARY_MODEL` to use another
OpenAI-compatible provider. Groq requests use JSON-object mode for review summaries and do not
send OpenRouter-specific fallback or reasoning parameters. `LLM_API_KEY`, `OPENROUTER_API_KEY`,
and the legacy `SILICONFLOW_API_KEY` remain accepted when configuring the application directly.
Summaries are generated separately for
critic and user reviews on the lead platform, using the review page's first page (up to 10
critic reviews and 50 user reviews). The source hash prevents repeat API calls when those
reviews have not changed. Review text is queued for background summarization so catalog
collection does not wait for Groq. Transient API errors and invalid or empty model
responses are retried up to three times with backoff. HTTP 429 responses are not retried
immediately: the provider's `Retry-After` value and an exponential cooldown are applied to
the pending queue to avoid repeatedly hitting a rate limit. The worker persists unfinished
summaries and retries them without repeating the Metacritic crawl. A provider quota or rate limit still needs to be resolved at the configured LLM provider for
summaries to complete.

## AI summary evaluation
The evaluation dataset in `evals/review_summary_cases.json` uses invented review text and
covers balanced feedback, positive-only feedback, and an instruction-injection attempt embedded
in a review. Validate the dataset without contacting an LLM provider:
```bash
python -m scripts.evaluate_summaries --dry-run
```

Run the cases against the configured model and save a JSON report:
```bash
python -m scripts.evaluate_summaries --output evaluation-report.json
```

The report measures coverage of predefined phrase alternatives, checks for selected unsupported
claims, and enforces expected empty feedback lists. These are heuristic regression signals, not
a proof of factuality or a replacement for human review. Live evaluation makes one provider
request per case and may be delayed by provider rate limits.

## Hourly crawler
```bash
python -m app.worker --once --batch-size 20
python -m app.worker
```

`app.worker` runs a cadence from `CRAWL_INTERVAL_MINUTES` (default 60). The worker:
- pulls the 'New Releases' list and the next browse page from the DB cursor;
- uses `TIMEZONE` (default `UTC`) to determine the crawl day and skips games already
  processed successfully or failed `CRAWL_MAX_FAILURES_PER_DAY` times (default 2) that day;
- fetches the game page plus per-platform user-score pages, with platform requests running
  concurrently up to `METACRITIC_MAX_CONCURRENCY` (default 2, maximum 4);
- upserts the game and platform rows into Postgres;
- writes `crawl_runs` / `crawl_items` rows for monitoring.

If Metacritic returns a block/challenge page, the current game is recorded as failed and the
run stops rather than continuing to send requests. At the next run, it can be retried until
the daily failure limit is reached. On acquiring the crawl lock, the worker also marks any
unfinished run from a previous crashed process as failed.
