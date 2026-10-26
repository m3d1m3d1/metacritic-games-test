"""Read structured data from the `<script id="__NUXT_DATA__">` block of a Metacritic page.

Metacritic is a Nuxt SSR site: every page ships the data it rendered from as a JSON array in
"devalue" format (a flat list where integers inside objects/lists are indexes into that list).
Parsing this is far more stable than scraping generated CSS classes, and it contains fields
that are not visible on the page (platform list with per-platform scores, review items, ...).
"""
from __future__ import annotations

import json
from typing import Any

from bs4 import BeautifulSoup

_WRAPPERS = {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "EmptyRef", "EmptyShallowRef"}
_SPECIAL = {"Date", "Set", "Map", "BigInt", "null", "undefined", "NaN", "Infinity", "-Infinity", "-0", "RegExp", "URL"}


class NuxtDataError(ValueError):
    """The page has no (parsable) __NUXT_DATA__ block."""


def decode(flat: list) -> Any:
    memo: dict[int, Any] = {}

    def res(i: int) -> Any:
        if i in memo:
            return memo[i]
        v = flat[i]
        if isinstance(v, list):
            if len(v) == 2 and isinstance(v[0], str) and v[0] in _WRAPPERS:
                memo[i] = None
                memo[i] = res(v[1])
                return memo[i]
            if v and isinstance(v[0], str) and v[0] in _SPECIAL:
                memo[i] = v
                return v
            out: list = []
            memo[i] = out
            out.extend(res(x) if isinstance(x, int) and x >= 0 else x for x in v)
            return out
        if isinstance(v, dict):
            out_d: dict = {}
            memo[i] = out_d
            for k, x in v.items():
                out_d[k] = res(x) if isinstance(x, int) and x >= 0 else x
            return out_d
        return v

    return res(0)


def load_nuxt(html: str) -> dict:
    tag = BeautifulSoup(html, "lxml").find("script", id="__NUXT_DATA__")
    if tag is None or not tag.string:
        raise NuxtDataError("no __NUXT_DATA__ block")
    try:
        return decode(json.loads(tag.string))
    except (ValueError, IndexError, TypeError) as exc:
        raise NuxtDataError(f"cannot decode __NUXT_DATA__: {exc}") from exc


def page_components(payload: dict) -> dict[str, dict]:
    """componentName -> component, for the page's own `loadPage:` entry (skeleton/header excluded)."""
    for key, value in (payload.get("data") or {}).items():
        if key.startswith("loadPage:") and "skeleton" not in key and isinstance(value, dict):
            return {
                c["meta"]["componentName"]: c
                for c in value.get("components", [])
                if isinstance(c, dict) and c.get("meta", {}).get("componentName")
            }
    return {}
