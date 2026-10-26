import json

import pytest

from app.scraper.parsers import (
    parse_browse,
    parse_game_page,
    parse_home_new_releases,
    parse_reviews_page,
)


def _nuxt_html(payload: dict, jsonld: dict | None = None) -> str:
    flat: list = []

    def encode(value):
        index = len(flat)
        flat.append(None)
        if isinstance(value, dict):
            flat[index] = {key: encode(item) for key, item in value.items()}
        elif isinstance(value, list):
            flat[index] = [encode(item) for item in value]
        elif isinstance(value, bool):
            flat[index] = "true" if value else None
        elif isinstance(value, int):
            flat[index] = float(value)
        else:
            flat[index] = value
        return index

    encode(payload)
    page = (
        '<script id="__NUXT_DATA__" type="application/json">'
        f"{json.dumps(flat)}</script>"
    )
    if jsonld is not None:
        page += (
            '<script type="application/ld+json">'
            f"{json.dumps(jsonld)}</script>"
        )
    return page


def _component(name: str, data: dict) -> dict:
    return {"meta": {"componentName": name}, "data": data}


def _review_page(kind: str, summary: dict, reviews: list[dict]) -> str:
    return _nuxt_html(
        {
            "data": {
                "loadPage:reviews": {
                    "components": [
                        _component(
                            f"{kind}-score-summary",
                            {"item": summary},
                        ),
                        _component(
                            f"{kind}-reviews",
                            {"items": reviews},
                        ),
                    ]
                }
            }
        }
    )


def test_game_page():
    platforms = [
        {"slug": "pc", "criticScoreSummary": {"score": None, "reviewCount": 2}},
        {
            "slug": "nintendo-switch-2",
            "name": "Switch 2",
            "isLeadPlatform": True,
            "releaseDate": "2026-10-15",
            "criticScoreSummary": {"score": 76, "reviewCount": 8},
        },
        {
            "slug": "playstation-5",
            "criticScoreSummary": {"score": None, "reviewCount": 3},
        },
        {
            "slug": "xbox-series-x",
            "criticScoreSummary": {"score": None, "reviewCount": 2},
        },
    ]
    html = _nuxt_html(
        {
            "data": {
                "loadPage:game": {
                    "components": [
                        _component(
                            "product",
                            {
                                "item": {
                                    "slug": "example-game",
                                    "title": "Example Game",
                                    "description": "A fictional game for parser tests.",
                                    "releaseDate": "2026-10-15",
                                    "genres": [{"name": "Adventure"}],
                                    "production": {
                                        "companies": [
                                            {
                                                "typeName": "Developer",
                                                "name": "Example Studio",
                                            },
                                            {
                                                "typeName": "Publisher",
                                                "name": "Sample Publisher",
                                            },
                                        ]
                                    },
                                    "images": [
                                        {
                                            "typeName": "cardImage",
                                            "filename": "portrait-test.jpg",
                                            "bucketPath": "/test/portrait-test.jpg",
                                        },
                                        {
                                            "typeName": "mainImage",
                                            "filename": "landscape-test.jpg",
                                            "bucketPath": "/test/landscape-test.jpg",
                                        },
                                    ],
                                    "platforms": platforms,
                                    "video": {
                                        "jwPlayerId": "testPlayer42",
                                        "title": "Example trailer",
                                    },
                                }
                            },
                        ),
                        _component(
                            "critic-score-summary",
                            {
                                "item": {
                                    "score": 76,
                                    "reviewCount": 8,
                                    "url": (
                                        "https://example.test/game/example-game/"
                                        "critic-reviews/?platform=nintendo-switch-2"
                                    ),
                                }
                            },
                        ),
                        _component(
                            "user-score-summary",
                            {"item": {"score": 0, "reviewCount": 0}},
                        ),
                    ]
                }
            }
        },
        {
            "@type": "VideoGame",
            "name": "Example Game",
            "screenshot": [
                {"contentUrl": "https://cdn.example.test/portrait-test.jpg"}
            ],
            "image": ["https://cdn.example.test/landscape-test.jpg"],
        },
    )

    game = parse_game_page(html)
    assert (game.slug, game.title) == ("example-game", "Example Game")
    assert game.developers == ["Example Studio"]
    assert game.publishers == ["Sample Publisher"]
    assert game.genres == ["Adventure"]
    assert game.release_date == "2026-10-15"
    assert game.description == "A fictional game for parser tests."
    assert game.metascore == 76
    assert game.userscore is None
    assert game.shown_platform == "nintendo-switch-2"
    assert game.video_id == "testPlayer42"
    assert game.video_url == "https://cdn.jwplayer.com/players/testPlayer42.html"
    assert game.cover_url == "https://cdn.example.test/portrait-test.jpg"

    got = {platform.platform: (platform.metascore, platform.critic_count) for platform in game.platforms}
    assert got == {
        "pc": (None, 2),
        "nintendo-switch-2": (76, 8),
        "playstation-5": (None, 3),
        "xbox-series-x": (None, 2),
    }
    assert [platform.platform for platform in game.platforms if platform.is_lead] == [
        "nintendo-switch-2"
    ]


def test_critic_reviews():
    reviews = [
        {
            "quote": f"Fictional critic review {index}.",
            "score": 90 if index == 0 else 75,
            "publicationName": "Example Journal" if index == 0 else "Sample Review",
            "url": f"https://example.test/review/{index}",
        }
        for index in range(8)
    ]
    html = _review_page(
        "critic",
        {
            "score": 76,
            "reviewCount": 8,
            "positiveCount": 6,
            "neutralCount": 2,
            "negativeCount": 0,
            "url": "https://example.test/reviews/?platform=nintendo-switch-2",
        },
        reviews,
    )

    result = parse_reviews_page(html, "critic")
    assert result.platform == "nintendo-switch-2"
    assert result.score == 76 and result.review_count == 8
    assert (result.positive, result.neutral, result.negative) == (6, 2, 0)
    assert len(result.reviews) == 8
    assert result.reviews[0].author == "Example Journal"
    assert result.reviews[0].text and result.reviews[0].url.startswith("http")
    assert all(0 <= review.score <= 100 for review in result.reviews)
    assert any(
        review.author == "Example Journal" and review.score == 90
        for review in result.reviews
    )


def test_empty_user_reviews():
    html = _review_page(
        "user",
        {
            "score": 0,
            "reviewCount": 0,
            "url": "https://example.test/reviews/?platform=nintendo-switch-2",
        },
        [],
    )
    result = parse_reviews_page(html, "user")
    assert result.platform == "nintendo-switch-2"
    assert result.score is None and result.review_count == 0 and result.reviews == []


def test_critic_review_counts_and_page_items_are_independent():
    reviews = [
        {
            "quote": f"Invented critic review {index}.",
            "score": 88,
            "publicationName": "Example Journal",
        }
        for index in range(10)
    ]
    html = _review_page(
        "critic",
        {
            "score": 88,
            "reviewCount": 62,
            "positiveCount": 57,
            "neutralCount": 5,
            "negativeCount": 0,
            "url": "https://example.test/reviews/?platform=xbox-series-x",
        },
        reviews,
    )
    result = parse_reviews_page(html, "critic")
    assert (result.platform, result.score, result.review_count) == (
        "xbox-series-x",
        88,
        62,
    )
    assert (result.positive, result.neutral, result.negative) == (57, 5, 0)
    assert len(result.reviews) == 10


def test_user_reviews():
    reviews = [
        {
            "body": f"Invented user review {index}.",
            "score": 8.6,
            "username": "Example Player",
            "spoiler": index == 12,
        }
        for index in range(50)
    ]
    html = _review_page(
        "user",
        {
            "score": 8.6,
            "reviewCount": 1588,
            "url": "https://example.test/reviews/?platform=xbox-series-x",
        },
        reviews,
    )
    result = parse_reviews_page(html, "user")
    assert (result.platform, result.score, result.review_count) == (
        "xbox-series-x",
        8.6,
        1588,
    )
    assert len(result.reviews) == 50
    assert all(
        review.text and review.author and 0 <= review.score <= 10
        for review in result.reviews
    )
    assert any(review.spoiler for review in result.reviews)


def test_home_new_releases():
    cards = "".join(
        f'<a href="/game/sample-{index}/"><h3>Sample {index}</h3>80</a>'
        for index in range(20)
    )
    html = f"""
    <nav><a href="/game/menu-entry/">Menu entry</a></nav>
    <section><h2>New Releases</h2><div data-testid="carousel">{cards}</div></section>
    """
    slugs = parse_home_new_releases(html)
    assert len(slugs) == 20 and len(set(slugs)) == 20
    assert slugs[0] == "sample-0"
    assert all("/" not in slug for slug in slugs)


def test_browse_pages():
    def browse_page(prefix: str) -> str:
        cards = "".join(
            f'<a href="/game/{prefix}-{index}/">'
            f'Sample {index}Oct 15, 2026</a>'
            for index in range(24)
        )
        return f'<nav><a href="/game/menu-entry/">Menu entry</a></nav>{cards}'

    first_page = parse_browse(browse_page("first"))
    second_page = parse_browse(browse_page("second"))
    assert len(first_page) == 24 and len(second_page) == 24
    assert all(item.release_date == "Oct 15, 2026" for item in first_page)
    assert not ({item.slug for item in first_page} & {item.slug for item in second_page})


def test_bad_review_kind():
    with pytest.raises(ValueError):
        parse_reviews_page("", "nope")


HOME = """
<nav><a href="/game/menu-entry/">Menu entry</a></nav>
<section><div><div><h2>New Releases</h2><a href="/browse/game/all/all/all-time/new/">See All</a></div>
<div data-testid="carousel">
 <a href="/game/alpha-adventure/"><h3>Alpha Adventure</h3>86</a>
 <a href="/game/beta-journey/"><h3>Beta Journey</h3>77</a>
 <a href="/game/alpha-adventure/">Alpha Adventure again</a>
</div></div></section>
<section><div><h2>Upcoming Games</h2><a href="/game/gamma-story/"><h3>Gamma Story</h3>tbd</a></div></section>
"""

BROWSE = """
<nav><a href="/game/menu-entry/">Menu entry</a></nav>
<a href="/game/ember-quest/">Ember QuestOct 15, 2026</a>
<a href="/game/glacier-maze/">Glacier MazeOct 15, 2026A fictional description</a>
"""


def test_home_synthetic_card_titles_do_not_stop_the_scan():
    assert parse_home_new_releases(HOME) == ["alpha-adventure", "beta-journey"]


def test_home_limit():
    assert parse_home_new_releases(HOME, limit=1) == ["alpha-adventure"]


def test_browse_synthetic():
    items = parse_browse(BROWSE)
    assert [item.slug for item in items] == ["ember-quest", "glacier-maze"]
    assert items[0].release_date == "Oct 15, 2026"
