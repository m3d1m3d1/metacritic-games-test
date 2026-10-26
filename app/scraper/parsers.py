"""Parsers for Metacritic pages.

* Game and review pages use the embedded Nuxt payload (see nuxt.py) and schema.org JSON-LD.
* Home and browse listings use URL shapes and visible text rather than generated CSS classes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from app.scraper.nuxt import load_nuxt, page_components

SLUG_RE = re.compile(r"^(?:https?://www\.metacritic\.com)?/game/([^/?#]+)/?$")
DATE_RE = re.compile(r"[A-Z][a-z]{2} \d{1,2}, \d{4}")
IMG_BASE = "https://www.metacritic.com/a/img/catalog"


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------- listings
@dataclass
class ListingItem:
    slug: str
    release_date: str | None = None


def _slug(href: str | None) -> str | None:
    m = SLUG_RE.match(href or "")
    return m.group(1) if m else None


def parse_home_new_releases(html: str, limit: int = 20) -> list[str]:
    """Slugs from the 'New Releases' block of /game/ (first N, in page order).

    The section heading lives inside the same container as its cards (as in the
    'Related Games' carousel on game pages), so: find the heading, climb to the nearest
    ancestor that contains game links, and read the links from there.
    """
    soup = _soup(html)
    heading = next(
        (h for h in soup.find_all(["h2", "h3"]) if h.get_text(strip=True).lower() == "new releases"),
        None,
    )
    if heading is None:
        return []
    node = heading.parent
    while node is not None and not any(_slug(a.get("href")) for a in node.find_all("a", href=True)):
        node = node.parent
    if node is None:
        return []
    slugs: list[str] = []
    for a in node.find_all("a", href=True):
        slug = _slug(a["href"])
        if slug and slug not in slugs:
            slugs.append(slug)
            if len(slugs) == limit:
                break
    return slugs


def parse_browse(html: str) -> list[ListingItem]:
    """Cards on /browse/game/... pages.

    A card link contains the release date ("Oct 15, 2026") in its text, which nav/menu
    links to games do not - this keeps the header mega-menu out of the result.
    """
    items: dict[str, ListingItem] = {}
    for a in _soup(html).find_all("a", href=True):
        slug = _slug(a["href"])
        if not slug or slug in items:
            continue
        m = DATE_RE.search(a.get_text(" ", strip=True))
        if m:
            items[slug] = ListingItem(slug=slug, release_date=m.group(0))
    return list(items.values())


# --------------------------------------------------------------------------- game page
@dataclass
class PlatformScore:
    platform: str  # url slug, e.g. "xbox-series-x"
    name: str | None = None
    is_lead: bool = False
    release_date: str | None = None
    metascore: int | None = None  # None = "tbd" (fewer than 4 critic reviews)
    critic_count: int | None = None
    userscore: float | None = None  # filled from the platform's user-reviews page
    user_count: int | None = None


@dataclass
class GamePage:
    slug: str | None = None
    title: str | None = None
    description: str | None = None
    cover_url: str | None = None
    developers: list[str] = field(default_factory=list)
    publishers: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    release_date: str | None = None  # ISO, e.g. 2026-10-15
    metascore: int | None = None  # of the platform the page shows (lead platform)
    userscore: float | None = None
    shown_platform: str | None = None
    platforms: list[PlatformScore] = field(default_factory=list)
    video_id: str | None = None
    video_title: str | None = None
    video_url: str | None = None  # embeddable player page

    def coverage(self) -> dict[str, bool]:
        return {k: bool(v) for k, v in self.__dict__.items()}


def _jsonld(soup: BeautifulSoup) -> dict:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        for obj in data if isinstance(data, list) else [data]:
            if isinstance(obj, dict) and obj.get("@type") == "VideoGame":
                return obj
    return {}


def _platform_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return parse_qs(urlparse(url).query).get("platform", [None])[0]


def _score_or_none(summary: dict | None, user: bool = False) -> int | float | None:
    """Metacritic encodes 'tbd' as score null (critics) or score 0 with 0 reviews (users)."""
    if not summary:
        return None
    score, count = summary.get("score"), summary.get("reviewCount") or 0
    if score is None or (user and count == 0):
        return None
    return score


def _cover(item: dict, ld: dict) -> str | None:
    """Portrait card image if present, else main image. URLs come from the page's own JSON-LD
    (they carry a signed /resize/<hash>/ prefix we cannot build ourselves)."""
    urls = [s.get("contentUrl") for s in ld.get("screenshot", []) if isinstance(s, dict)]
    img = ld.get("image")
    urls += img if isinstance(img, list) else [img]
    images = item.get("images") or []
    by_type = {i.get("typeName"): i for i in images}
    for type_name in ("cardImage", "mainImage"):
        entry = by_type.get(type_name)
        fname = entry.get("filename") if entry else None
        match = next((u for u in urls if u and fname and fname in u), None)
        if match:
            return match
    for type_name in ("cardImage", "mainImage"):
        entry = by_type.get(type_name)
        if entry and entry.get("bucketPath"):
            return f"{IMG_BASE}{entry['bucketPath']}"
    return next((u for u in urls if u), None)


def parse_game_page(html: str) -> GamePage:
    comps = page_components(load_nuxt(html))
    item = (comps.get("product") or {}).get("data", {}).get("item") or {}
    ld = _jsonld(_soup(html))
    page = GamePage()

    page.slug = item.get("slug")
    page.title = item.get("title") or ld.get("name")
    page.description = (item.get("description") or ld.get("description") or "").strip() or None
    page.cover_url = _cover(item, ld)
    page.release_date = item.get("releaseDate") or ld.get("datePublished")
    page.genres = [g["name"] for g in item.get("genres", []) if g.get("name")]

    companies = (item.get("production") or {}).get("companies", [])
    page.developers = [c["name"] for c in companies if c.get("typeName") == "Developer" and c.get("name")]
    page.publishers = [c["name"] for c in companies if c.get("typeName") == "Publisher" and c.get("name")]

    crit = (comps.get("critic-score-summary") or {}).get("data", {}).get("item")
    user = (comps.get("user-score-summary") or {}).get("data", {}).get("item")
    page.metascore = _score_or_none(crit)
    page.userscore = _score_or_none(user, user=True)
    page.shown_platform = _platform_from_url((crit or {}).get("url"))

    for p in item.get("platforms", []):
        cs = p.get("criticScoreSummary") or {}
        page.platforms.append(
            PlatformScore(
                platform=p.get("slug"),
                name=p.get("name"),
                is_lead=bool(p.get("isLeadPlatform")),
                release_date=p.get("releaseDate"),
                metascore=_score_or_none(cs),
                critic_count=cs.get("reviewCount"),
            )
        )

    video = item.get("video") or {}
    trailer = ld.get("trailer") or {}
    page.video_id = video.get("jwPlayerId")
    page.video_title = video.get("title") or trailer.get("name")
    page.video_url = trailer.get("embedUrl") or (
        f"https://cdn.jwplayer.com/players/{page.video_id}.html" if page.video_id else None
    )
    return page


# --------------------------------------------------------------------------- review pages
@dataclass
class Review:
    kind: str  # "critic" | "user"
    text: str
    score: int | None = None  # critics: 0-100, users: 0-10
    author: str | None = None  # critic: publication name, user: nickname
    date: str | None = None
    url: str | None = None
    platform: str | None = None
    spoiler: bool = False


@dataclass
class ReviewsPage:
    kind: str
    platform: str | None = None  # slug
    score: int | float | None = None  # platform summary score (None = tbd)
    review_count: int = 0
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    reviews: list[Review] = field(default_factory=list)


def parse_reviews_page(html: str, kind: str) -> ReviewsPage:
    """`/game/<slug>/critic-reviews/?platform=..` or `/user-reviews/?platform=..`.

    The page embeds the score summary of that platform plus the first page of reviews
    (10 critic / 50 user). Item layouts verified on real pages (Gears of War: E-Day).
    """
    if kind not in ("critic", "user"):
        raise ValueError(kind)
    comps = page_components(load_nuxt(html))
    summary = (comps.get(f"{kind}-score-summary") or {}).get("data", {}).get("item") or {}
    block = (comps.get(f"{kind}-reviews") or {}).get("data", {}) or {}
    page = ReviewsPage(
        kind=kind,
        platform=_platform_from_url(summary.get("url")),
        score=_score_or_none(summary, user=(kind == "user")),
        review_count=summary.get("reviewCount") or 0,
        positive=summary.get("positiveCount") or 0,
        neutral=summary.get("neutralCount") or 0,
        negative=summary.get("negativeCount") or 0,
    )
    for it in block.get("items") or []:
        text = (it.get("quote") or it.get("body") or it.get("review") or "").strip()
        if not text:
            continue  # e.g. spoiler-hidden or score-only entries
        page.reviews.append(
            Review(
                kind=kind,
                text=text,
                score=it.get("score"),
                author=(
                    it.get("publicationName") or it.get("author") or it.get("username") or it.get("userName") or None
                ),
                date=it.get("date"),
                url=it.get("url"),
                platform=it.get("platform") if isinstance(it.get("platform"), str) else None,
                spoiler=bool(it.get("spoiler")),
            )
        )
    return page
