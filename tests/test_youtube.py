from app import youtube


def test_search_selects_most_viewed_non_live_result(monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            assert options["skip_download"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, query, download):
            assert query == "ytsearch10:Example Game gameplay let's play"
            assert download is False
            return {
                "entries": [
                    {"id": "lowview00001", "title": "Small channel", "view_count": 50},
                    {"id": "livevideo001", "title": "Live stream", "view_count": 5000, "is_live": True},
                    {"id": "highview0001", "title": "Most viewed", "view_count": 900},
                    {"id": "noview000001", "title": "No view metadata"},
                ]
            }

    monkeypatch.setattr(youtube, "YoutubeDL", FakeYoutubeDL)

    found = youtube._search_most_viewed("Example Game")

    assert found is not None
    assert found.video_id == "highview0001"
    assert found.views == 900
    assert found.url == "https://www.youtube.com/watch?v=highview0001"


def test_letsplay_source_hash_depends_on_video_and_transcript():
    original = youtube.letsplay_source_hash("video1", "Transcript text")

    assert original == youtube.letsplay_source_hash("video1", "Transcript text")
    assert original != youtube.letsplay_source_hash("video2", "Transcript text")
    assert original != youtube.letsplay_source_hash("video1", "Updated transcript")
