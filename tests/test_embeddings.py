import pytest

from app import embeddings
from app.db.models import EMBEDDING_DIM
from app.scraper.parsers import GamePage


def test_embedding_text_combines_title_genres_and_description():
    page = GamePage(
        title="Example",
        genres=["Action", "Adventure"],
        description="A story-driven game.",
    )
    same_page = GamePage(
        title="Example",
        genres=["Action", "Adventure"],
        description="A story-driven game.",
    )
    text = embeddings.embedding_text(page)

    assert text == "Example\nGenres: Action, Adventure\nA story-driven game."
    assert embeddings.embedding_source_hash(page) == embeddings.embedding_source_hash(same_page)
    assert embeddings.embedding_source_hash(page) != embeddings.embedding_source_hash(
        GamePage(title="Example", genres=["Action"], description="A story-driven game.")
    )


@pytest.mark.asyncio
async def test_embed_game_returns_384_finite_values_without_loading_real_model(monkeypatch):
    class FakeModel:
        def embed(self, texts):
            assert texts == ["Example"]
            yield [0.25] * EMBEDDING_DIM

    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    vector, source_hash = await embeddings.embed_game(GamePage(title="Example"))

    assert len(vector) == EMBEDDING_DIM
    assert all(value == 0.25 for value in vector)
    assert source_hash == embeddings.embedding_source_hash(GamePage(title="Example"))


@pytest.mark.asyncio
async def test_embed_game_rejects_empty_source_text():
    with pytest.raises(ValueError, match="without title"):
        await embeddings.embed_game(GamePage())
