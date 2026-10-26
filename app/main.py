from __future__ import annotations

import asyncio
import html
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import current_day, settings
from app.db import repo
from app.db.models import CrawlRun, Game, WorkerLog, WorkerStatus
from app.db.session import SessionLocal, get_session
from app.llm import ReviewSummaryRateLimited, translate_game_content

app = FastAPI(title="Metacritic Games", version="0.1.0")
logger = logging.getLogger(__name__)


def _format_datetime(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if value else "—"


def _render_run(run: CrawlRun) -> str:
    error = f"<p class='error'>{html.escape(run.error)}</p>" if run.error else ""
    items = "".join(
        "<tr>"
        f"<td>{html.escape(item.slug)}</td>"
        f"<td>{html.escape(item.source)}</td>"
        f"<td><span class='status {html.escape(item.status)}'>{html.escape(item.status)}</span></td>"
        f"<td>{html.escape(item.error or '')}</td>"
        "</tr>"
        for item in sorted(run.items, key=lambda item: item.id)
    )
    items_table = (
        "<details><summary>Игры в запуске</summary>"
        "<table><thead><tr><th>Slug</th><th>Источник</th><th>Статус</th><th>Ошибка</th></tr></thead>"
        f"<tbody>{items}</tbody></table></details>"
        if items
        else ""
    )
    return (
        "<tr>"
        f"<td>#{run.id}</td>"
        f"<td>{html.escape(run.trigger)}</td>"
        f"<td><span class='status {html.escape(run.status)}'>{html.escape(run.status)}</span></td>"
        f"<td>{html.escape(run.day.isoformat())}</td>"
        f"<td>{html.escape(_format_datetime(run.started_at))}</td>"
        f"<td>{html.escape(_format_datetime(run.finished_at))}</td>"
        f"<td>{run.ok_count}/{run.planned} ({run.failed_count} ошибок)</td>"
        f"<td>{error}{items_table}</td>"
        "</tr>"
    )


def _worker_payload(workers: list[WorkerStatus]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    result = []
    for worker in workers:
        online = now - worker.last_seen_at <= timedelta(seconds=20)
        result.append(
            {
                "worker_id": worker.worker_id,
                "state": worker.state if online else "offline",
                "last_seen_at": worker.last_seen_at.isoformat(),
                "current_run_id": worker.current_run_id,
                "error": worker.error,
                "online": online,
            }
        )
    return result


def _log_payload(logs: list[WorkerLog]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "worker_id": item.worker_id,
            "level": item.level,
            "message": item.message,
            "run_id": item.run_id,
            "created_at": item.created_at.isoformat(),
        }
        for item in logs
    ]


def _monitor_payload(
    runs: list[CrawlRun],
    workers: list[WorkerStatus],
    logs: list[WorkerLog] | None = None,
    summary_stats: dict[str, int | str | None] | None = None,
) -> dict[str, Any]:
    return {
        "workers": _worker_payload(workers),
        "runs_html": "".join(_render_run(run) for run in runs)
        or "<tr><td colspan='8'>Запусков пока нет.</td></tr>",
        "total_games": sum(run.ok_count for run in runs),
        "active_runs": sum(run.status in ("queued", "running") for run in runs),
        "pending_summaries": summary_stats
        or {"pending": 0, "due": 0, "retrying": 0, "last_error": None},
        "logs": _log_payload(logs or []),
    }


def render_dashboard(
    runs: list[CrawlRun],
    workers: list[WorkerStatus] | None = None,
    logs: list[WorkerLog] | None = None,
    summary_stats: dict[str, int | str | None] | None = None,
) -> str:
    rows = "".join(_render_run(run) for run in runs)
    if not rows:
        rows = "<tr><td colspan='8'>Запусков пока нет.</td></tr>"
    worker_rows = _worker_payload(workers or [])
    summary_counts = summary_stats or {
        "pending": 0,
        "due": 0,
        "retrying": 0,
        "last_error": None,
    }
    summary_error = (
        f"<p class='error' id='summary-error'>Последняя ошибка: "
        f"{html.escape(str(summary_counts['last_error'])[:300])}</p>"
        if summary_counts["last_error"]
        else "<p class='error' id='summary-error' hidden></p>"
    )
    log_items = "".join(
        "<li class='log-entry'>"
        f"<time>{html.escape(item.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S'))}</time> "
        f"<span class='log-level {html.escape(item.level.lower())}'>{html.escape(item.level)}</span> "
        f"<span>{html.escape(item.message)}</span></li>"
        for item in logs or []
    ) or "<li class='muted'>Событий пока нет.</li>"
    worker_html = "".join(
        "<article class='worker'>"
        f"<strong>{html.escape(item['worker_id'])}</strong> · "
        f"<span class='status {html.escape(item['state'])}'>{html.escape(item['state'])}</span>"
        f"<div>Последний сигнал: <time>{html.escape(item['last_seen_at'])}</time></div>"
        f"<div>Текущий запуск: {html.escape(str(item['current_run_id'] or '—'))}</div>"
        f"<div class='error'>{html.escape(item['error'] or '')}</div></article>"
        for item in worker_rows
    ) or "<p id='no-workers'>Воркер пока не сообщил о себе.</p>"
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Metacritic Games — мониторинг</title>
  <style>
    body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #20242a; }}
    h1 {{ margin-bottom: .25rem; }}
    .muted {{ color: #626b76; }}
    form {{ margin: 1.5rem 0; }}
    button {{ background: #2459d3; border: 0; border-radius: 5px; color: white; cursor: pointer; font: inherit; padding: .65rem 1rem; }}
    button:hover {{ background: #1746b0; }}
    table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; }}
    th, td {{ border-bottom: 1px solid #dce0e5; padding: .55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    .status {{ border-radius: 4px; display: inline-block; padding: .1rem .4rem; background: #edf0f3; }}
    .status.ok {{ background: #e2f4e8; }}
    .status.failed, .status.partial {{ background: #ffe8e5; }}
    .status.queued, .status.running, .status.pending {{ background: #fff3cd; }}
    .error {{ color: #a32b20; white-space: pre-wrap; }}
    details {{ min-width: 260px; }}
    details table {{ font-size: .9rem; }}
    .workers {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .worker {{ border: 1px solid #dce0e5; border-radius: 6px; padding: .8rem; min-width: 260px; }}
    .status.idle {{ background: #e2f4e8; }}
    .status.offline, .status.stopped {{ background: #ffe8e5; }}
    .logs-panel {{ margin: 1rem 0; border: 1px solid #dce0e5; border-radius: 6px; padding: .7rem; }}
    .logs-panel summary {{ cursor: pointer; font-weight: 600; }}
    .log-list {{ background: #171b22; color: #e3e8ef; font: 13px/1.5 ui-monospace, monospace; height: 280px; overflow: auto; padding: .7rem 2.5rem; }}
    .log-list li {{ padding: .12rem 0; overflow-wrap: anywhere; }}
    .log-list time {{ color: #9ba8b7; }}
    .log-level {{ display: inline-block; min-width: 4.5em; }}
    .log-level.error {{ color: #ff8c83; }}
    .log-level.warning {{ color: #ffd166; }}
    .log-level.info {{ color: #80c7ff; }}
    .catalog-link {{ display: inline-flex; align-items: center; gap: .45rem; margin: .5rem 0 1rem; padding: .7rem 1rem; border-radius: 7px; background: #2459d3; color: #fff; font-weight: 650; text-decoration: none; box-shadow: 0 2px 5px #152c5526; transition: background .15s, transform .15s; }}
    .catalog-link:hover {{ background: #1746b0; color: #fff; text-decoration: none; transform: translateY(-1px); }}
  </style>
</head>
<body>
  <h1>Metacritic Games</h1>
  <a class="catalog-link" href="/">← Вернуться в каталог игр</a>
  <p class="muted">Мониторинг обходов. Интервал планировщика: {settings.crawl_interval_minutes} мин.</p>
  <form method="post" action="/crawl-runs">
    <button type="submit">Запустить сбор сейчас</button>
  </form>
  <h2>Состояние воркеров</h2>
  <div id="workers" class="workers">{worker_html}</div>
  <p id="summary-queue">Очередь резюме: <strong id="summary-pending">{summary_counts["pending"]}</strong> ожидают, <strong id="summary-due">{summary_counts["due"]}</strong> готовы к обработке, <strong id="summary-retrying">{summary_counts["retrying"]}</strong> с ошибками.</p>
  {summary_error}
  <details class="logs-panel">
    <summary>Живые логи воркера</summary>
    <ol class="log-list" id="worker-logs">{log_items}</ol>
  </details>
  <p>Успешно обработано в показанных запусках: <strong id="total-games">{sum(run.ok_count for run in runs)}</strong></p>
  <h2>Последние запуски</h2>
  <table>
    <thead><tr><th>ID</th><th>Триггер</th><th>Статус</th><th>Дата</th><th>Запрошен / старт</th><th>Завершён</th><th>Обработано</th><th>Подробности</th></tr></thead>
    <tbody id="runs">{rows}</tbody>
  </table>
  <script>
    const workers = document.getElementById("workers");
    const runs = document.getElementById("runs");
    const logList = document.getElementById("worker-logs");
    const source = new EventSource("/api/monitor/events");
    source.onmessage = (event) => {{
      const data = JSON.parse(event.data);
      document.getElementById("total-games").textContent = data.total_games;
      document.getElementById("summary-pending").textContent = data.pending_summaries.pending;
      document.getElementById("summary-due").textContent = data.pending_summaries.due;
      document.getElementById("summary-retrying").textContent = data.pending_summaries.retrying;
      const summaryError = document.getElementById("summary-error");
      summaryError.hidden = !data.pending_summaries.last_error;
      summaryError.textContent = data.pending_summaries.last_error
        ? `Последняя ошибка резюме: ${{data.pending_summaries.last_error}}`
        : "";
      runs.innerHTML = data.runs_html;
      const wasAtBottom = logList.scrollHeight - logList.scrollTop - logList.clientHeight < 24;
      logList.replaceChildren();
      if (!data.logs.length) {{
        const empty = document.createElement("li");
        empty.className = "muted";
        empty.textContent = "Событий пока нет.";
        logList.append(empty);
      }}
      for (const entry of data.logs) {{
        const item = document.createElement("li");
        item.className = "log-entry";
        const time = document.createElement("time");
        time.textContent = new Date(entry.created_at).toLocaleString();
        const level = document.createElement("span");
        level.className = `log-level ${{entry.level.toLowerCase()}}`;
        level.textContent = entry.level;
        const message = document.createElement("span");
        message.textContent = entry.message;
        item.append(time, document.createTextNode(" "), level, document.createTextNode(" "), message);
        logList.append(item);
      }}
      if (wasAtBottom) logList.scrollTop = logList.scrollHeight;
      workers.replaceChildren();
      if (!data.workers.length) {{
        const empty = document.createElement("p");
        empty.textContent = "Воркер пока не сообщил о себе.";
        workers.append(empty);
        return;
      }}
      for (const worker of data.workers) {{
        const card = document.createElement("article");
        card.className = "worker";
        const title = document.createElement("strong");
        title.textContent = worker.worker_id;
        const state = document.createElement("span");
        state.className = `status ${{worker.state}}`;
        state.textContent = worker.state;
        const lastSeen = document.createElement("div");
        lastSeen.textContent = `Последний сигнал: ${{worker.last_seen_at}}`;
        const run = document.createElement("div");
        run.textContent = `Текущий запуск: ${{worker.current_run_id || "—"}}`;
        const error = document.createElement("div");
        error.className = "error";
        error.textContent = worker.error || "";
        card.append(title, document.createTextNode(" · "), state, lastSeen, run, error);
        workers.append(card);
      }}
    }};
  </script>
</body>
</html>"""


def _run_payload(run: CrawlRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "trigger": run.trigger,
        "status": run.status,
        "day": run.day.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "planned": run.planned,
        "ok_count": run.ok_count,
        "failed_count": run.failed_count,
        "error": run.error,
        "items": [
            {
                "slug": item.slug,
                "source": item.source,
                "status": item.status,
                "error": item.error,
                "processed_at": item.processed_at.isoformat() if item.processed_at else None,
            }
            for item in sorted(run.items, key=lambda item: item.id)
        ],
    }


def _score(value: Any) -> str:
    return str(value) if value is not None else "—"


def _catalog_state_query(
    *, query: str, platform: str, sort: str, order: str, page: int
) -> str:
    return urlencode(
        {
            "q": query,
            "platform": platform,
            "sort": sort,
            "order": order,
            "page": page,
        }
    )


def _game_url(
    slug: str,
    *,
    query: str | None = None,
    platform: str = "",
    sort: str = "metascore",
    order: str = "desc",
    page: int = 1,
) -> str:
    url = f"/games/{quote(slug, safe='')}"
    if query is not None:
        url += "?" + _catalog_state_query(
            query=query, platform=platform, sort=sort, order=order, page=page
        )
    return url


def render_game_list(
    games: list[Game],
    platforms: list[tuple[str, str | None]],
    total: int,
    *,
    query: str,
    platform: str,
    sort: str,
    order: str,
    page: int,
    page_size: int,
) -> str:
    cards = []
    for game in games:
        image = (
            f"<img src='{html.escape(game.cover_url, quote=True)}' alt='' loading='lazy'>"
            if game.cover_url
            else "<div class='cover-placeholder'>Нет обложки</div>"
        )
        game_url = html.escape(
            _game_url(
                game.slug,
                query=query,
                platform=platform,
                sort=sort,
                order=order,
                page=page,
            ),
            quote=True,
        )
        cards.append(
            "<article class='game-card'>"
            f"<a class='cover' href='{game_url}'>{image}</a>"
            f"<h2><a href='{game_url}'>{html.escape(game.title)}</a></h2>"
            f"<p>Metascore: <strong>{html.escape(_score(game.best_metascore))}</strong>"
            f" · Userscore: <strong>{html.escape(_score(game.best_userscore))}</strong></p>"
            f"<p>{html.escape(', '.join(game.genres or []))}</p>"
            "</article>"
        )
    cards_html = "".join(cards) or "<p>Игры не найдены.</p>"
    platform_options = ["<option value=''>Все платформы</option>"]
    for slug, name in platforms:
        selected = " selected" if slug == platform else ""
        label = name or slug
        platform_options.append(
            f"<option value='{html.escape(slug, quote=True)}'{selected}>{html.escape(label)}</option>"
        )
    next_page = (
        f"<a href='/?q={quote(query)}&amp;platform={quote(platform)}&amp;sort={sort}"
        f"&amp;order={order}&amp;page={page + 1}'>Следующая →</a>"
        if page * page_size < total
        else ""
    )
    previous_page = (
        f"<a href='/?q={quote(query)}&amp;platform={quote(platform)}&amp;sort={sort}"
        f"&amp;order={order}&amp;page={page - 1}'>← Предыдущая</a>"
        if page > 1
        else ""
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Игры Metacritic — каталог</title>
  <style>
    body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #20242a; }}
    a {{ color: #164fc4; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
    header {{ display: flex; justify-content: space-between; align-items: baseline; }}
    form.filters {{ display: flex; flex-wrap: wrap; gap: .6rem; margin: 1.5rem 0; }}
    input, select, button {{ font: inherit; padding: .55rem; border: 1px solid #cbd1d9; border-radius: 5px; }}
    button {{ cursor: pointer; background: #2459d3; color: white; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 1rem; }}
    .game-card {{ border: 1px solid #e0e4e9; border-radius: 8px; padding: .8rem; }}
    .game-card h2 {{ font-size: 1.1rem; margin: .5rem 0; }}
    .cover img, .cover-placeholder {{ width: 100%; aspect-ratio: 3 / 4; object-fit: cover; background: #f1f3f5; }}
    .cover-placeholder {{ display: grid; place-items: center; color: #69717c; }}
    .pagination {{ display: flex; justify-content: space-between; margin: 1.5rem 0; }}
    .muted {{ color: #69717c; }}
  </style>
</head>
<body>
  <header><h1>Каталог игр</h1><a href="/monitor">Мониторинг обходов</a></header>
  <p class="muted">Найдено игр: {total}</p>
  <form class="filters" method="get" action="/">
    <input type="search" name="q" value="{html.escape(query, quote=True)}" placeholder="Название игры">
    <select name="platform">{''.join(platform_options)}</select>
    <select name="sort">
      <option value="metascore"{" selected" if sort == "metascore" else ""}>Metascore</option>
      <option value="userscore"{" selected" if sort == "userscore" else ""}>Userscore</option>
      <option value="title"{" selected" if sort == "title" else ""}>Название</option>
      <option value="release_date"{" selected" if sort == "release_date" else ""}>Дата выхода</option>
    </select>
    <select name="order">
      <option value="desc"{" selected" if order == "desc" else ""}>По убыванию</option>
      <option value="asc"{" selected" if order == "asc" else ""}>По возрастанию</option>
    </select>
    <button type="submit">Применить</button>
  </form>
  <main class="grid">{cards_html}</main>
  <nav class="pagination">{previous_page}{next_page}</nav>
</body>
</html>"""


def render_game_detail(
    game: Game,
    similar: list[Game],
    *,
    previous_slug: str | None = None,
    next_slug: str | None = None,
    query: str = "",
    platform: str = "",
    sort: str = "metascore",
    order: str = "desc",
    page: int = 1,
) -> str:
    cover = (
        f"<img class='hero-cover' src='{html.escape(game.cover_url, quote=True)}' alt='Обложка {html.escape(game.title, quote=True)}'>"
        if game.cover_url
        else ""
    )
    platforms = "".join(
        "<tr>"
        f"<td>{html.escape(item.name or item.platform)}</td>"
        f"<td>{html.escape(_score(item.metascore))}</td>"
        f"<td>{html.escape(_score(item.userscore))}</td>"
        f"<td>{html.escape(_score(item.critic_count))} / {html.escape(_score(item.user_count))}</td>"
        "</tr>"
        for item in game.platforms
    ) or "<tr><td colspan='4'>Данных о платформах пока нет.</td></tr>"
    summaries = {summary.audience: summary for summary in game.summaries}
    summary_sections = []
    for audience, label in (("critic", "Критики"), ("user", "Пользователи")):
        summary = summaries.get(audience)
        if summary is None:
            summary_sections.append(f"<section><h2>{label}</h2><p>Резюме пока нет.</p></section>")
            continue
        likes = "".join(f"<li>{html.escape(point)}</li>" for point in summary.likes) or "<li>Нет данных</li>"
        dislikes = "".join(f"<li>{html.escape(point)}</li>" for point in summary.dislikes) or "<li>Нет данных</li>"
        summary_sections.append(
            f"<section><h2>{label}</h2><p id='summary-{audience}-text' data-translation-field='summary' "
            f"data-audience='{audience}'>{html.escape(summary.summary)}</p>"
            f"<h3>Нравится</h3><ul id='summary-{audience}-likes' data-translation-field='likes' "
            f"data-audience='{audience}'>{likes}</ul><h3>Не нравится</h3>"
            f"<ul id='summary-{audience}-dislikes' data-translation-field='dislikes' "
            f"data-audience='{audience}'>{dislikes}</ul>"
            f"<p class='muted'>Отзывов в резюме: {summary.reviews_used}</p></section>"
        )
    similar_cards = "".join(
        f"<li><a href='{_game_url(other.slug)}'>{html.escape(other.title)}</a></li>"
        for other in similar
    ) or "<li>Похожие игры появятся после накопления данных о жанрах или эмбеддингах.</li>"
    previous_navigation = (
        f"<a class='game-navigation previous' href='{html.escape(_game_url(previous_slug, query=query, platform=platform, sort=sort, order=order, page=page), quote=True)}' aria-label='Предыдущая игра' title='Предыдущая игра'>←</a>"
        if previous_slug
        else "<span class='game-navigation previous disabled' aria-disabled='true' aria-label='Предыдущей игры нет'>←</span>"
    )
    next_navigation = (
        f"<a class='game-navigation next' href='{html.escape(_game_url(next_slug, query=query, platform=platform, sort=sort, order=order, page=page), quote=True)}' aria-label='Следующая игра' title='Следующая игра'>→</a>"
        if next_slug
        else "<span class='game-navigation next disabled' aria-disabled='true' aria-label='Следующей игры нет'>→</span>"
    )
    video = (
        f"<p><a href='{html.escape(game.video_url, quote=True)}' target='_blank' rel='noopener'>"
        f"{html.escape(game.video_title or 'Смотреть видео')}</a></p>"
        if game.video_url and game.video_url.startswith("https://")
        else ""
    )
    letsplay = ""
    if (
        game.letsplay_url
        and game.letsplay_url.startswith("https://www.youtube.com/")
        and game.letsplay_summary
    ):
        letsplay = (
            "<section><h2>Летсплей на YouTube</h2>"
            f"<p><a href='{html.escape(game.letsplay_url, quote=True)}' target='_blank' rel='noopener'>"
            f"{html.escape(game.letsplay_title or 'Смотреть летсплей')}</a>"
            f" · Просмотров: {html.escape(_score(game.letsplay_views))}</p>"
            f"<p id='letsplay-summary' data-translation-field='letsplay_summary'>"
            f"{html.escape(game.letsplay_summary)}</p></section>"
        )
    developers = ", ".join(game.developers or []) or "—"
    publishers = ", ".join(game.publishers or []) or "—"
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(game.title)} — Metacritic Games</title>
  <style>
    body {{ font: 15px/1.6 system-ui, sans-serif; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; color: #20242a; }}
    a {{ color: #164fc4; }} .hero {{ display: flex; gap: 2rem; align-items: flex-start; }}
    .hero-cover {{ width: min(280px, 40vw); object-fit: cover; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ text-align: left; border-bottom: 1px solid #dce0e5; padding: .6rem; }}
    section {{ border-top: 1px solid #e0e4e9; padding: .5rem 0; }}
    .muted {{ color: #69717c; }}
    .language-switch {{ display: flex; align-items: center; flex-wrap: wrap; gap: .5rem; margin: 1rem 0; }}
    .language-switch button {{ border: 1px solid #cbd1d9; border-radius: 5px; padding: .45rem .75rem; background: #fff; color: #164fc4; cursor: pointer; font: inherit; }}
    .language-switch button[aria-pressed="true"] {{ background: #2459d3; border-color: #2459d3; color: #fff; }}
    #language-status {{ color: #69717c; }}
    .game-navigation {{ position: fixed; z-index: 10; top: 50%; display: flex; align-items: center; justify-content: center; box-sizing: border-box; width: 3.25rem; height: 3.25rem; padding: 0 0 .12em; transform: translateY(-50%); border: 1px solid #d4dbe5; border-radius: 50%; background: #fff; color: #164fc4; box-shadow: 0 3px 14px #17243a33; font-family: Arial, sans-serif; font-size: 2rem; line-height: 1; text-align: center; text-decoration: none; transition: background .15s, color .15s, transform .15s; }}
    .game-navigation:hover {{ transform: translateY(-50%) scale(1.08); background: #2459d3; color: #fff; text-decoration: none; }}
    .game-navigation.previous {{ left: 1rem; }}
    .game-navigation.next {{ right: 1rem; }}
    .game-navigation.disabled {{ color: #aeb6c2; background: #f0f2f5; box-shadow: none; }}
    @media(max-width:600px) {{ .game-navigation {{ width: 2.7rem; height: 2.7rem; font-size: 1.6rem; }} }}
    @media(max-width:600px) {{ .hero {{ display: block; }} .hero-cover {{ width: min(100%, 280px); }} }}
  </style>
</head>
<body>
  {previous_navigation}
  {next_navigation}
  <p><a href='{html.escape("/?" + _catalog_state_query(query=query, platform=platform, sort=sort, order=order, page=page), quote=True)}'>← Каталог игр</a> · <a href="/monitor">Мониторинг</a></p>
  <div class="language-switch" role="group" aria-label="Язык описания и резюме">
    <button type="button" id="language-original" aria-pressed="true">Оригинал (Metacritic)</button>
    <button type="button" id="language-russian" aria-pressed="false">Русский</button>
    <span id="language-status" role="status" aria-live="polite"></span>
  </div>
  <header class="hero">{cover}<div><h1>{html.escape(game.title)}</h1>
    <p>Metascore: <strong>{html.escape(_score(game.best_metascore))}</strong>
    · Userscore: <strong>{html.escape(_score(game.best_userscore))}</strong></p>
    <p>Дата выхода: {html.escape(game.release_date.isoformat() if game.release_date else "—")}</p>
    <p>Жанры: {html.escape(", ".join(game.genres or []) or "—")}</p>
    <p>Разработчик: {html.escape(developers)} · Издатель: {html.escape(publishers)}</p>
    {video}
  </div></header>
  <section><h2>Описание</h2><p id="game-description" data-translation-field="description">{html.escape(game.description or "Описание отсутствует.")}</p></section>
  <section><h2>Оценки по платформам</h2>
    <table><thead><tr><th>Платформа</th><th>Metascore</th><th>Userscore</th><th>Отзывы критиков / пользователей</th></tr></thead>
    <tbody>{platforms}</tbody></table>
  </section>
  {''.join(summary_sections)}
  {letsplay}
  <section><h2>Похожие игры</h2><ul>{similar_cards}</ul></section>
  <script>
    const originalButton = document.getElementById("language-original");
    const russianButton = document.getElementById("language-russian");
    const languageStatus = document.getElementById("language-status");
    const translationUrl = "/api/games/{quote(game.slug, safe='')}/translation";
    const originalValues = new Map();
    for (const element of document.querySelectorAll("[data-translation-field]")) {{
      originalValues.set(element.id, element.matches("ul")
        ? Array.from(element.querySelectorAll("li"), item => item.textContent)
        : element.textContent);
    }}
    let russianContent = null;
    let translationPending = false;

    function setLanguage(language) {{
      const isRussian = language === "ru";
      originalButton.setAttribute("aria-pressed", String(!isRussian));
      russianButton.setAttribute("aria-pressed", String(isRussian));
      const translationByAudience = new Map(
        (russianContent?.summaries || []).map(item => [item.audience, item])
      );
      for (const element of document.querySelectorAll("[data-translation-field]")) {{
        const field = element.dataset.translationField;
        const audience = element.dataset.audience;
        const translated = audience ? translationByAudience.get(audience)?.[field]
          : russianContent?.[field];
        if (element.matches("ul")) {{
          const values = isRussian && Array.isArray(translated)
            ? translated
            : originalValues.get(element.id);
          element.replaceChildren(...values.map(value => {{
            const item = document.createElement("li");
            item.textContent = value;
            return item;
          }}));
        }} else {{
          element.textContent = isRussian && typeof translated === "string"
            ? translated
            : originalValues.get(element.id);
        }}
      }}
      languageStatus.textContent = "";
    }}

    originalButton.addEventListener("click", () => setLanguage("original"));
    russianButton.addEventListener("click", async () => {{
      if (translationPending) return;
      if (!russianContent) {{
        translationPending = true;
        russianButton.disabled = true;
        languageStatus.textContent = "Перевод загружается…";
        try {{
          const response = await fetch(translationUrl, {{ method: "POST" }});
          const payload = await response.json();
          if (!response.ok) {{
            throw new Error(payload.detail || "Не удалось загрузить перевод.");
          }}
          russianContent = payload;
        }} catch (error) {{
          languageStatus.textContent = error.message || "Не удалось загрузить перевод.";
          return;
        }} finally {{
          translationPending = false;
          russianButton.disabled = false;
        }}
      }}
      setLanguage("ru");
    }});
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/games", response_class=HTMLResponse, include_in_schema=False)
async def game_catalog(
    q: str = "",
    platform: str = "",
    sort: str = "metascore",
    order: str = "desc",
    page: int = Query(default=1, ge=1),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    if sort not in {"metascore", "userscore", "title", "release_date"}:
        raise HTTPException(status_code=422, detail="Unsupported sort field")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=422, detail="Unsupported sort order")
    page_size = 24
    games, total = await repo.list_games(
        session,
        query=q or None,
        platform=platform or None,
        sort=sort,
        order=order,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    platforms = await repo.available_platforms(session)
    return HTMLResponse(
        render_game_list(
            games,
            platforms,
            total,
            query=q,
            platform=platform,
            sort=sort,
            order=order,
            page=page,
            page_size=page_size,
        )
    )


@app.get("/games/{slug}", response_class=HTMLResponse, include_in_schema=False)
async def game_detail(
    slug: str,
    q: str = "",
    platform: str = "",
    sort: str = "metascore",
    order: str = "desc",
    page: int = Query(default=1, ge=1),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    if sort not in {"metascore", "userscore", "title", "release_date"}:
        raise HTTPException(status_code=422, detail="Unsupported sort field")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=422, detail="Unsupported sort order")
    game = await repo.game_by_slug(session, slug)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    similar = await repo.similar_games(session, game)
    previous_slug, next_slug = await repo.game_neighbors(
        session,
        slug,
        query=q or None,
        platform=platform or None,
        sort=sort,
        order=order,
    )
    return HTMLResponse(
        render_game_detail(
            game,
            similar,
            previous_slug=previous_slug,
            next_slug=next_slug,
            query=q,
            platform=platform,
            sort=sort,
            order=order,
            page=page,
        )
    )


def _game_translation_payload(game: Game) -> dict[str, Any]:
    return {
        "description": game.description_ru,
        "letsplay_summary": game.letsplay_summary_ru,
        "summaries": [
            {
                "audience": summary.audience,
                "summary": summary.summary_ru,
                "likes": summary.likes_ru,
                "dislikes": summary.dislikes_ru,
            }
            for summary in game.summaries
        ],
    }


@app.post("/api/games/{slug}/translation")
async def translate_game(
    slug: str,
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    game = await repo.game_by_slug(session, slug)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    summaries_to_translate = [
        summary
        for summary in game.summaries
        if summary.summary_ru is None
        or summary.likes_ru is None
        or summary.dislikes_ru is None
    ]
    description_to_translate = game.description if game.description_ru is None else None
    letsplay_to_translate = (
        game.letsplay_summary
        if game.letsplay_summary_ru is None
        else None
    )
    if not (
        description_to_translate
        or letsplay_to_translate
        or summaries_to_translate
    ):
        return JSONResponse(_game_translation_payload(game))

    summary_sources = [
        {
            "audience": summary.audience,
            "summary": summary.summary,
            "likes": summary.likes,
            "dislikes": summary.dislikes,
        }
        for summary in summaries_to_translate
    ]
    try:
        translation = await translate_game_content(
            description=description_to_translate,
            letsplay_summary=letsplay_to_translate,
            summaries=summary_sources,
        )
    except ReviewSummaryRateLimited as exc:
        headers = (
            {"Retry-After": str(exc.retry_after_seconds)}
            if exc.retry_after_seconds is not None
            else None
        )
        raise HTTPException(
            status_code=503,
            detail="Сервис перевода временно ограничил запросы. Попробуйте позже.",
            headers=headers,
        ) from exc
    except httpx.HTTPStatusError as exc:
        provider_status = exc.response.status_code
        if provider_status == 401:
            logger.error("Translation provider rejected its API key for game %s", slug)
            raise HTTPException(
                status_code=503,
                detail=(
                    "OpenRouter отклонил API-ключ (401). Проверьте OPENROUTER_API_KEY "
                    "и LLM_API_KEY в .env, затем пересоздайте контейнеры приложения."
                ),
            ) from exc
        if provider_status == 403:
            logger.error("Translation provider denied access for game %s", slug)
            raise HTTPException(
                status_code=503,
                detail=(
                    "OpenRouter запретил запрос (403). Проверьте доступ ключа и "
                    "доступность выбранных моделей в аккаунте."
                ),
            ) from exc
        logger.exception("Game translation failed for %s", slug)
        raise HTTPException(
            status_code=502,
            detail=f"Провайдер перевода вернул HTTP {provider_status}. Попробуйте позже.",
        ) from exc
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        logger.exception("Game translation failed for %s", slug)
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить перевод. Проверьте соединение и попробуйте позже.",
        ) from exc

    translated_summaries = {
        summary.audience: {
            "summary": summary.summary,
            "likes": summary.likes,
            "dislikes": summary.dislikes,
        }
        for summary in translation.summaries
    }
    game.description_ru = (
        translation.description
        if description_to_translate
        else game.description_ru
    )
    game.letsplay_summary_ru = (
        translation.letsplay_summary
        if letsplay_to_translate
        else game.letsplay_summary_ru
    )
    for summary in summaries_to_translate:
        translated = translated_summaries[summary.audience]
        summary.summary_ru = translated["summary"]
        summary.likes_ru = translated["likes"]
        summary.dislikes_ru = translated["dislikes"]

    await repo.save_game_translations(
        session,
        game.id,
        description_ru=game.description_ru,
        letsplay_summary_ru=game.letsplay_summary_ru,
        summaries=translated_summaries,
    )
    await session.commit()
    return JSONResponse(_game_translation_payload(game))


@app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(session: AsyncSession = Depends(get_session)) -> HTMLResponse:
    runs = await repo.recent_runs(session)
    workers = await repo.worker_statuses(session)
    logs = await repo.recent_worker_logs(session)
    summary_stats = await repo.pending_review_summary_stats(session)
    return HTMLResponse(render_dashboard(runs, workers, logs, summary_stats))


@app.get("/api/monitor/status")
async def monitor_status(
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    runs = await repo.recent_runs(session)
    workers = await repo.worker_statuses(session)
    logs = await repo.recent_worker_logs(session)
    summary_stats = await repo.pending_review_summary_stats(session)
    return _monitor_payload(runs, workers, logs, summary_stats)


@app.get("/api/monitor/events")
async def monitor_events(request: Request):
    async def event_stream():
        while not await request.is_disconnected():
            async with SessionLocal() as session:
                runs = await repo.recent_runs(session)
                workers = await repo.worker_statuses(session)
                logs = await repo.recent_worker_logs(session)
                summary_stats = await repo.pending_review_summary_stats(session)
                payload = _monitor_payload(runs, workers, logs, summary_stats)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/crawl-runs")
async def list_crawl_runs(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    runs = await repo.recent_runs(session, limit)
    return [_run_payload(run) for run in runs]


@app.post("/api/crawl-runs", status_code=status.HTTP_202_ACCEPTED)
async def trigger_crawl(
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    run = await repo.enqueue_run(session, current_day())
    await session.commit()
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "id": run.id,
            "status": "queued",
            "message": "Ручной запуск добавлен в очередь.",
        },
    )


@app.post("/crawl-runs", include_in_schema=False)
async def trigger_crawl_from_dashboard(
    session: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    await repo.enqueue_run(session, current_day())
    await session.commit()
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
