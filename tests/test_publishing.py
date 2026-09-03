"""Этапы публикации: spoof_metadata и telegram_notify."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from publishing import spoof_metadata as sm
from publishing import telegram_notifier as tn


# ============================================================
# spoof_metadata
# ============================================================

@pytest.mark.ffmpeg
class TestSpoofMetadata:
    def test_rewrites_container_and_keeps_stream(self, sample_video, tmp_path, ffprobe_bin):
        out = tmp_path / "spoofed.mp4"

        assert sm.spoof_metadata(Path(sample_video), out) is True
        assert out.exists() and out.stat().st_size > 0

        probe = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_format", "-show_streams",
             "-of", "json", str(out)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(probe.stdout)
        codecs = {s["codec_type"] for s in data["streams"]}
        assert {"video", "audio"} <= codecs
        tags = data["format"].get("tags", {})
        assert tags.get("encoder") or tags.get("comment"), "метаданные не подставлены"

    def test_broken_input_returns_false(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        out = tmp_path / "out.mp4"

        assert sm.spoof_metadata(broken, out) is False
        assert not out.exists(), "битый результат не должен оставаться на диске"

    def test_missing_ffmpeg_returns_false(self, monkeypatch, sample_video, tmp_path):
        def no_ffmpeg(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(sm.subprocess, "run", no_ffmpeg)

        assert sm.spoof_metadata(Path(sample_video), tmp_path / "o.mp4") is False


# ============================================================
# telegram_notify
# ============================================================

class TestNormalizeAnimeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Ад_1080p_серия", "Ад серия"),
            ("Наруто-720p", "Наруто"),
            ("Тайтл_2_серия", "Тайтл 2 серия"),
            ("clip_2", "clip 2"),
        ],
    )
    def test_cleans_filename_stems(self, raw, expected):
        assert tn.normalize_anime_name(raw) == expected

    def test_empty_name_has_fallback(self):
        assert tn.normalize_anime_name("---") == "anime"


class TestFormatClipTitle:
    def test_placeholder_is_replaced_by_fallback(self):
        assert tn.format_clip_title("moment_1", fallback="Episode_2") == "Episode 2"
        assert tn.format_clip_title("moment 12", fallback="Ep") == "Ep"

    def test_real_title_is_kept(self):
        assert tn.format_clip_title("ОН СКАЗАЛ ЭТО", fallback="Ep") == "ОН СКАЗАЛ ЭТО"

    def test_whitespace_is_collapsed(self):
        assert tn.format_clip_title("  a   b  ", fallback="x") == "a b"


class FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class TestTelegramNotifier:
    def _notifier(self):
        return tn.TelegramNotifier("token", "chat")

    def test_message_escapes_html(self):
        message = self._notifier()._build_message(["<b>злой</b>"], "Аниме & Co")

        assert "&lt;b&gt;" in message
        assert "&amp;" in message
        assert "Название аниме" in message

    def test_message_without_titles_has_placeholder(self):
        assert "Без названия" in self._notifier()._build_message([], "X")

    def test_no_files_returns_false(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tn.requests, "post", lambda *a, **k: calls.append(k))

        assert self._notifier().send_media_group(["T"], "Anime", []) is False
        assert calls == []

    def test_sends_single_group(self, monkeypatch, tmp_path):
        posts = []

        def fake_post(url, data=None, files=None, timeout=None):
            posts.append({"url": url, "data": data, "files": list(files or {})})
            return FakeResponse()

        monkeypatch.setattr(tn.requests, "post", fake_post)
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"x")

        assert self._notifier().send_media_group(["T"], "Anime", [clip]) is True
        assert len(posts) == 1
        media = json.loads(posts[0]["data"]["media"])
        assert media[0]["type"] == "video"
        assert media[0]["caption"]

    def test_splits_into_chunks_of_ten(self, monkeypatch, tmp_path):
        posts = []

        def fake_post(url, data=None, files=None, timeout=None):
            posts.append(json.loads(data["media"]))
            return FakeResponse()

        monkeypatch.setattr(tn.requests, "post", fake_post)
        clips = []
        for i in range(12):
            path = tmp_path / f"clip_{i}.mp4"
            path.write_bytes(b"x")
            clips.append(path)

        assert self._notifier().send_media_group(["T"], "Anime", clips) is True
        assert [len(chunk) for chunk in posts] == [10, 2]
        assert "caption" in posts[0][0]
        assert "caption" not in posts[1][0], "подпись должна быть только у первой группы"

    def test_http_error_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            tn.requests, "post", lambda *a, **k: FakeResponse(400, "bad request")
        )
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"x")

        assert self._notifier().send_media_group(["T"], "Anime", [clip]) is False

    def test_network_error_returns_false(self, monkeypatch, tmp_path):
        def boom(*args, **kwargs):
            raise tn.requests.RequestException("нет сети")

        monkeypatch.setattr(tn.requests, "post", boom)
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"x")

        assert self._notifier().send_media_group(["T"], "Anime", [clip]) is False

    def test_missing_file_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tn.requests, "post", lambda *a, **k: FakeResponse())

        assert self._notifier().send_media_group(["T"], "Anime", [tmp_path / "nope.mp4"]) is False
