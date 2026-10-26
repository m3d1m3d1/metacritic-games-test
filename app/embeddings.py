from __future__ import annotations

import asyncio
import hashlib
import math
import threading

from fastembed import TextEmbedding

from app.db.models import EMBEDDING_DIM
from app.scraper.parsers import GamePage

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None
_model_lock = threading.Lock()


def embedding_text(page: GamePage) -> str:
    parts = [page.title or ""]
    if page.genres:
        parts.append("Genres: " + ", ".join(page.genres))
    if page.description:
        parts.append(page.description)
    return "\n".join(part.strip() for part in parts if part.strip())


def embedding_source_hash(page: GamePage) -> str:
    return hashlib.sha256(embedding_text(page).encode("utf-8")).hexdigest()


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def _embed(text: str) -> list[float]:
    vector = next(_get_model().embed([text]), None)
    if vector is None:
        raise RuntimeError("Embedding model returned no vector")
    values = [float(value) for value in vector]
    if len(values) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model returned {len(values)} dimensions; expected {EMBEDDING_DIM}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding model returned a non-finite vector value")
    return values


async def embed_game(page: GamePage) -> tuple[list[float], str]:
    text = embedding_text(page)
    if not text:
        raise ValueError("Cannot embed a game without title, genres, or description")
    source_hash = embedding_source_hash(page)
    return await asyncio.to_thread(_embed, text), source_hash
