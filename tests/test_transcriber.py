"""Этап transcribe_video: извлечение дорожки и нормализация текста."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import transcriber


class TestSelectWhisperModel:
    def test_defaults_to_base(self):
        assert transcriber.select_whisper_model({}) == "base"
        assert transcriber.select_whisper_model(None) == "base"

    def test_uses_configured_model(self):
        assert transcriber.select_whisper_model({"whisper_model": "large-v3"}) == "large-v3"


class TestSanitizeSubtitleText:
    def test_uppercases_and_strips_punctuation(self):
        assert transcriber.sanitize_subtitle_text("привет, мир!") == "ПРИВЕТ МИР"

    def test_keeps_highlight_tags(self):
        result = transcriber.sanitize_subtitle_text("это <hl>важно</hl>.")
        assert "<hl>ВАЖНО</hl>" in result
        assert result.startswith("ЭТО")

    def test_normalises_h1_and_slash_variants(self):
        result = transcriber.sanitize_subtitle_text("a <h1>b</h1> c")
        assert "<hl>B</hl>" in result

    def test_underscores_become_spaces(self):
        assert transcriber.sanitize_subtitle_text("a_b") == "A B"

    def test_empty_input(self):
        assert transcriber.sanitize_subtitle_text("") == ""

    def test_keeps_emoji(self):
        assert "🔥" in transcriber.sanitize_subtitle_text("огонь 🔥")


class TestRebuildWords:
    def test_words_cover_segment_timeline(self):
        segment = {"start": 10.0, "end": 12.0, "words": [{"word": "old", "start": 10.0, "end": 12.0}]}

        transcriber.rebuild_words_from_clean_text(segment, "ЭТО <hl>ВАЖНО</hl>")

        words = [w["word"] for w in segment["words"]]
        assert words == ["ЭТО", "<hl>", "ВАЖНО", "</hl>"]
        assert segment["words"][0]["start"] == pytest.approx(10.0)
        assert segment["words"][-1]["end"] == pytest.approx(12.0)
        for word in segment["words"]:
            assert 10.0 <= word["start"] <= word["end"] <= 12.0

    def test_empty_text_clears_words(self):
        segment = {"start": 0.0, "end": 1.0, "words": [{"word": "x", "start": 0, "end": 1}]}
        transcriber.rebuild_words_from_clean_text(segment, "   ")
        assert segment["words"] == []

    def test_non_dict_is_ignored(self):
        transcriber.rebuild_words_from_clean_text(None, "text")  # не должно падать


@pytest.mark.ffmpeg
class TestExtractAudioTrack:
    def test_produces_wav(self, sample_video):
        wav = transcriber.extract_audio_track(Path(sample_video))
        try:
            assert wav.exists()
            assert wav.suffix == ".wav"
            assert wav.stat().st_size > 1000
        finally:
            wav.unlink(missing_ok=True)

    def test_video_without_audio_raises(self, tmp_path, ffmpeg_bin):
        import subprocess

        silent = tmp_path / "silent.mp4"
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(silent),
            ],
            check=True,
            capture_output=True,
        )

        with pytest.raises(RuntimeError, match="нет аудиодорожек"):
            transcriber.extract_audio_track(silent)


@pytest.mark.ffmpeg
class TestTranscribeVideo:
    def test_uses_configured_model_and_language(self, sample_video, whisper_stub):
        cfg = {"subtitles": {"whisper_model": "small", "language": "ru"}}

        result = transcriber.transcribe_video(sample_video, cfg=cfg)

        assert whisper_stub.loaded_models == ["small"]
        assert whisper_stub.calls[0]["kwargs"]["language"] == "ru"
        assert result["segments"], "транскрипт должен содержать сегменты"

    def test_language_omitted_when_not_configured(self, sample_video, whisper_stub):
        transcriber.transcribe_video(sample_video, cfg={"subtitles": {}})

        assert "language" not in whisper_stub.calls[0]["kwargs"]

    def test_temp_audio_is_removed(self, sample_video):
        transcriber.transcribe_video(sample_video, cfg={})

        leftover = Path(transcriber.tempfile.gettempdir()) / f"{Path(sample_video).stem}_audio.wav"
        assert not leftover.exists()
