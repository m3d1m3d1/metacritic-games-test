"""Probe: can we reach Metacritic from THIS machine, and do the parsers work on real pages?

    python -m scripts.probe                          # direct
    python -m scripts.probe --proxy http://user:pass@host:port
    python -m scripts.probe --game gears-of-war-e-day   # a popular game: many critic + user reviews

Saves raw HTML into tests/fixtures/ and prints a report. Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.scraper.client import BlockedError, MetacriticClient
from app.scraper.parsers import (
    parse_browse,
    parse_game_page,
    parse_home_new_releases,
    parse_reviews_page,
)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def save(name: str, html: str) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / name).write_text(html, encoding="utf-8")
    print(f"   saved tests/fixtures/{name} ({len(html):,} bytes)")


async def main(proxy: str | None, game_slug: str | None) -> int:
    ok = True
    async with MetacriticClient(proxy=proxy) as mc:
        print("1) GET /game/  (New Releases, expect 20)")
        home = await mc.home()
        save("home.html", home)
        new = parse_home_new_releases(home)
        print(f"   parsed {len(new)}: {new[:5]}...")
        ok &= len(new) == 20

        print("2) GET browse (newest) pages 1, 2, 3")
        pages = []
        for n in (1, 2, 3):
            html = await mc.browse_new(n)
            save(f"browse_p{n}.html", html)
            pages.append([i.slug for i in parse_browse(html)])
        print(f"   cards per page: {[len(p) for p in pages]}")
        for a, b in ((0, 1), (1, 2)):
            dup = [s for s in pages[a] if s in pages[b]]
            print(f"   page{a+1} vs page{b+1} overlap: {len(dup)}"
                  f"  positions in page{a+1}: {[pages[a].index(s) for s in dup]}"
                  f"  in page{b+1}: {[pages[b].index(s) for s in dup]}")
        ok &= all(pages)

        slug = game_slug or (new[0] if new else pages[0][0])
        print(f"3) GET /game/{slug}/")
        html = await mc.game(slug)
        save(f"game_{slug}.html", html)
        game = parse_game_page(html)
        for k, v in game.coverage().items():
            print(f"     {'OK     ' if v else 'MISSING'} {k}")
        print(f"   title={game.title!r} dev={game.developers} meta={game.metascore} "
              f"user={game.userscore} lead={game.shown_platform}")
        print(f"   cover={game.cover_url}\n   video={game.video_url}")

        print("4) per-platform pages (critic + user reviews carry that platform's scores)")
        saved = False
        for p in game.platforms:
            c = parse_reviews_page(await mc.critic_reviews(slug, p.platform), "critic")
            u = parse_reviews_page(await mc.user_reviews(slug, p.platform), "user")
            print(f"   {p.platform:18} critic: score={c.score} n={c.review_count} got={len(c.reviews)}"
                  f" | user: score={u.score} n={u.review_count} got={len(u.reviews)}")
            if p.is_lead and not saved:
                saved = True
                save(f"critic_reviews_{slug}.html", await mc.critic_reviews(slug, p.platform))
                save(f"user_reviews_{slug}.html", await mc.user_reviews(slug, p.platform))

    print("\nRESULT:", "OK" if ok else "CHECK OUTPUT ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--game", default=None)
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(main(args.proxy, args.game)))
    except BlockedError as exc:
        print(f"\nBLOCKED: {exc}\nTry --proxy, a residential IP, or Playwright.")
        sys.exit(1)
