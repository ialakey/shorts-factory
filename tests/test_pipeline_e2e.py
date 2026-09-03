"""Сквозной прогон канала: от исходного видео до готового клипа.

Замокан только слой LLM и отправка в Telegram — всё остальное (ffmpeg,
транскрибация, рендер, подмена метаданных) выполняется по-настоящему.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from moviepy.editor import VideoFileClip

from core import channel_processor as cp

pytestmark = [pytest.mark.render, pytest.mark.ffmpeg]

PIPELINE = [
    "transcribe_video",
    "analyze_moment",
    "make_clips",
    "dynamic_shorts",
    "watermark",
    "enhance_audio",
    "video_effects",
    "music",
    "subtitles",
    "spoof_metadata",
    "telegram_notify",
]


@pytest.fixture
def sent_to_telegram(monkeypatch):
    sent = []

    class FakeNotifier:
        def __init__(self, token, chat_id):
            self.token = token
            self.chat_id = chat_id

        def send_media_group(self, titles, anime_name, paths):
            sent.append({"titles": list(titles), "anime": anime_name, "paths": [Path(p) for p in paths]})
            return True

    monkeypatch.setattr(cp, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(cp, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(cp, "TELEGRAM_CHAT_ID", "chat")
    return sent


@pytest.fixture
def channel(monkeypatch, tmp_path, channel_dir, render_config, sample_video):
    """Канал внутри временного CHANNELS_DIR с реальным исходником."""

    channels = tmp_path / "channels"
    channels.mkdir(exist_ok=True)
    target = channels / channel_dir.name
    shutil.move(str(channel_dir), str(target))
    shutil.copy(sample_video, target / "input_videos" / "episode.mp4")

    cfg = dict(render_config)
    cfg["channel_name"] = target.name
    cfg["pipeline"] = list(PIPELINE)
    (target / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    monkeypatch.setattr(cp, "CHANNELS_DIR", channels)
    return target


@pytest.fixture
def stub_llm(monkeypatch):
    """analyze_moment — единственное место, которое ходило бы в OpenAI."""

    calls = []

    def fake_analyze(transcript, video_path, cfg, api_key, output_dir=None):
        calls.append({"transcript": transcript, "video": Path(video_path)})
        return {
            "moment_1": {
                "title": "E2E CLIP",
                "duration": 1.2,
                "segments": {"segment_1": {"start": 0.2, "end": 1.4}},
            }
        }

    monkeypatch.setattr(cp, "analyze_moment", fake_analyze)
    return calls


def test_channel_runs_end_to_end(channel, stub_llm, sent_to_telegram, ffprobe_bin):
    cp.process_channel(channel.name)

    clips = sorted((channel / "output_clips").glob("*.mp4"))
    assert [c.name for c in clips] == ["E2E_CLIP.mp4"]

    with VideoFileClip(str(clips[0])) as clip:
        assert clip.duration > 0
        assert (clip.w, clip.h) == (216, 384)
        assert clip.audio is not None, "звуковая дорожка потерялась"

    # analyze_moment получил настоящий транскрипт с этапа transcribe_video
    assert stub_llm[0]["transcript"]["segments"]

    # spoof_metadata переписал контейнер
    probe = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_format", "-of", "json", str(clips[0])],
        capture_output=True, text=True, check=True,
    )
    tags = json.loads(probe.stdout)["format"].get("tags", {})
    assert tags.get("encoder") or tags.get("comment")

    # telegram получил ровно новый клип
    assert len(sent_to_telegram) == 1
    assert [p.name for p in sent_to_telegram[0]["paths"]] == ["E2E_CLIP.mp4"]
    assert sent_to_telegram[0]["titles"] == ["E2E CLIP"]
    assert sent_to_telegram[0]["anime"] == "episode"


def test_debug_mode_writes_artifacts(monkeypatch, tmp_path, channel, stub_llm, sent_to_telegram):
    """debug: вход из test_data, выход и дампы — в output_test_data."""

    test_data = tmp_path / "test_data"
    out_data = tmp_path / "output_test_data"
    test_data.mkdir(exist_ok=True)
    shutil.copy(channel / "input_videos" / "episode.mp4", test_data / "debug_episode.mp4")
    monkeypatch.setattr(cp, "TEST_DATA_DIR", test_data)
    monkeypatch.setattr(cp, "OUTPUT_TEST_DATA_DIR", out_data)

    cfg = yaml.safe_load((channel / "config.yaml").read_text(encoding="utf-8"))
    cfg["debug"] = True
    cfg["pipeline"] = ["transcribe_video", "analyze_moment", "make_clips"]
    (channel / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    cp.process_channel(channel.name)

    assert (out_data / "debug_episode_transcript.txt").exists()
    assert (out_data / "debug_episode_moments.json").exists()
    assert list(out_data.glob("*.mp4")), "клип должен лежать в output_test_data"
    assert not list((channel / "output_clips").glob("*.mp4")), "в debug боевую папку не трогаем"


def test_failing_channel_does_not_break_others(monkeypatch, channel, stub_llm):
    """app.py ловит падение канала — процесс продолжает работу."""

    def boom(*args, **kwargs):
        raise RuntimeError("этап упал")

    monkeypatch.setattr(cp, "make_clips", boom)

    with pytest.raises(RuntimeError):
        cp.process_channel(channel.name)
