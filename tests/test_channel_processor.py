"""Оркестрация пайплайна: какие этапы запускаются для данного ``pipeline``.

Сами этапы здесь замоканы — проверяется именно проводка между ними:
порядок, пропуски, автозапуск ``make_clips`` и обработка отсутствующих данных.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core import channel_processor as cp


class Recorder:
    """Собирает вызовы замоканных этапов."""

    def __init__(self):
        self.calls: dict[str, list] = {
            "kodik": [],
            "transcribe": [],
            "analyze": [],
            "make_clips": [],
            "spoof": [],
            "telegram": [],
        }

    def called(self, stage: str) -> bool:
        return bool(self.calls[stage])


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Изолированный канал + замоканные этапы пайплайна."""

    channels = tmp_path / "channels"
    (channels / "Chan" / "input_videos").mkdir(parents=True)
    (channels / "Chan" / "output_clips").mkdir(parents=True)
    monkeypatch.setattr(cp, "CHANNELS_DIR", channels)
    monkeypatch.setattr(cp, "TEST_DATA_DIR", tmp_path / "test_data")
    monkeypatch.setattr(cp, "OUTPUT_TEST_DATA_DIR", tmp_path / "output_test_data")
    monkeypatch.setattr(cp, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(cp, "TELEGRAM_CHAT_ID", "")

    rec = Recorder()

    def fake_kodik(titles, input_dir):
        rec.calls["kodik"].append((titles, Path(input_dir)))

    def fake_transcribe(video_path, cfg=None):
        rec.calls["transcribe"].append(Path(video_path))
        return {"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

    def fake_analyze(transcript, video_path, cfg, api_key, output_dir=None):
        rec.calls["analyze"].append(Path(video_path))
        return {"moment_1": {"title": "T", "segments": {"segment_1": {"start": 0, "end": 1}}}}

    def fake_make_clips(video_path, moments, output_folder, cfg, *, pipeline=None, channel_dir=None):
        rec.calls["make_clips"].append(
            {
                "video": Path(video_path),
                "moments": moments,
                "output": Path(output_folder),
                "pipeline": list(pipeline or []),
                "channel_dir": Path(channel_dir) if channel_dir else None,
            }
        )
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        (Path(output_folder) / "T.mp4").write_bytes(b"clip")

    def fake_spoof(src, dst):
        rec.calls["spoof"].append((Path(src), Path(dst)))
        Path(dst).write_bytes(b"spoofed")
        return True

    class FakeNotifier:
        def __init__(self, token, chat_id):
            self.token = token
            self.chat_id = chat_id

        def send_media_group(self, titles, anime_name, paths):
            rec.calls["telegram"].append((list(titles), anime_name, [Path(p) for p in paths]))
            return True

    monkeypatch.setattr(cp, "auto_download_titles", fake_kodik)
    monkeypatch.setattr(cp, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(cp, "analyze_moment", fake_analyze)
    monkeypatch.setattr(cp, "make_clips", fake_make_clips)
    monkeypatch.setattr(cp, "spoof_metadata", fake_spoof)
    monkeypatch.setattr(cp, "TelegramNotifier", FakeNotifier)

    def setup(pipeline, *, extra_cfg=None, with_video=True):
        cfg = {"channel_name": "Chan", "pipeline": list(pipeline)}
        cfg.update(extra_cfg or {})
        (channels / "Chan" / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8"
        )
        if with_video:
            (channels / "Chan" / "input_videos" / "ep1.mp4").write_bytes(b"fake")
        return channels / "Chan"

    env_obj = type("Env", (), {})()
    env_obj.rec = rec
    env_obj.setup = setup
    env_obj.channels = channels
    env_obj.monkeypatch = monkeypatch
    return env_obj


def test_core_pipeline_runs_every_requested_stage(env):
    base = env.setup(["transcribe_video", "analyze_moment", "make_clips"])

    cp.process_channel("Chan")

    rec = env.rec
    assert rec.called("transcribe")
    assert rec.called("analyze")
    assert rec.called("make_clips")
    assert not rec.called("kodik")
    assert not rec.called("spoof")
    assert not rec.called("telegram")

    call = rec.calls["make_clips"][0]
    assert call["output"] == base / "output_clips"
    assert call["channel_dir"] == base
    assert "moment_1" in call["moments"]


def test_stages_absent_from_pipeline_are_skipped(env):
    env.setup(["transcribe_video"])

    cp.process_channel("Chan")

    assert env.rec.called("transcribe")
    assert not env.rec.called("analyze")
    assert not env.rec.called("make_clips")


def test_analyze_without_transcript_does_not_run(env):
    env.setup(["analyze_moment"])

    cp.process_channel("Chan")

    assert not env.rec.called("transcribe")
    assert not env.rec.called("analyze")


def test_post_clip_stage_forces_make_clips(env):
    """subtitles без make_clips → make_clips запускается сам на весь ролик."""

    env.setup(["subtitles"])

    cp.process_channel("Chan")

    assert env.rec.called("make_clips")
    moments = env.rec.calls["make_clips"][0]["moments"]
    assert moments == [{"start": 0.0, "end": 1.0, "title": "ep1"}]


def test_kodik_stage_receives_config_section(env):
    env.setup(
        ["kodik_download", "transcribe_video"],
        extra_cfg={"kodik_download": [["Название", "1-2", "Студия"]]},
    )

    cp.process_channel("Chan")

    titles, input_dir = env.rec.calls["kodik"][0]
    assert titles == [["Название", "1-2", "Студия"]]
    assert input_dir.name == "input_videos"


def test_legacy_autodownload_stage_still_works(env):
    env.setup(["autodownload"], extra_cfg={"autodownload": [["X", "1", "Y"]]})

    cp.process_channel("Chan")

    assert env.rec.calls["kodik"][0][0] == [["X", "1", "Y"]]


def test_kodik_failure_does_not_break_pipeline(env, monkeypatch):
    env.setup(["kodik_download", "transcribe_video"], extra_cfg={"kodik_download": []})

    def boom(titles, input_dir):
        raise RuntimeError("kodik недоступен")

    monkeypatch.setattr(cp, "auto_download_titles", boom)

    cp.process_channel("Chan")

    # падение загрузки не должно ронять остальные этапы
    assert env.rec.called("transcribe")


def test_debug_mode_uses_test_folders_and_dumps_artifacts(env, tmp_path):
    env.setup(
        ["transcribe_video", "analyze_moment", "make_clips"],
        extra_cfg={"debug": True},
        with_video=False,
    )
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    (test_data / "debug_ep.mp4").write_bytes(b"fake")

    cp.process_channel("Chan")

    out = tmp_path / "output_test_data"
    assert env.rec.calls["make_clips"][0]["output"] == out
    assert (out / "debug_ep_transcript.txt").exists()
    assert (out / "debug_ep_moments.json").exists()


def test_debug_mode_skips_kodik(env, tmp_path):
    env.setup(
        ["kodik_download", "transcribe_video"],
        extra_cfg={"debug": True, "kodik_download": [["A", "1", "B"]]},
        with_video=False,
    )
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    (test_data / "debug_ep.mp4").write_bytes(b"fake")

    cp.process_channel("Chan")

    assert not env.rec.called("kodik")


def test_spoof_metadata_replaces_original_clip(env):
    env.setup(["transcribe_video", "analyze_moment", "make_clips", "spoof_metadata"])

    cp.process_channel("Chan")

    assert env.rec.called("spoof")
    clip = env.channels / "Chan" / "output_clips" / "T.mp4"
    assert clip.exists()
    assert clip.read_bytes() == b"spoofed"


def test_failed_spoof_keeps_original_clip(env, monkeypatch):
    env.setup(["transcribe_video", "analyze_moment", "make_clips", "spoof_metadata"])
    monkeypatch.setattr(cp, "spoof_metadata", lambda src, dst: False)

    cp.process_channel("Chan")

    clip = env.channels / "Chan" / "output_clips" / "T.mp4"
    assert clip.read_bytes() == b"clip"
    assert not (env.channels / "Chan" / "output_clips" / "T_spoofed.mp4").exists()


def test_telegram_skipped_without_credentials(env):
    env.setup(["transcribe_video", "analyze_moment", "make_clips", "telegram_notify"])

    cp.process_channel("Chan")

    assert not env.rec.called("telegram")


def test_telegram_sends_only_new_clips(env, monkeypatch):
    monkeypatch.setattr(cp, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(cp, "TELEGRAM_CHAT_ID", "chat")
    base = env.setup(["transcribe_video", "analyze_moment", "make_clips", "telegram_notify"])
    (base / "output_clips" / "old.mp4").write_bytes(b"old")

    cp.process_channel("Chan")

    titles, anime, paths = env.rec.calls["telegram"][0]
    assert [p.name for p in paths] == ["T.mp4"]
    assert anime == "ep1"
    assert titles == ["T"]


def test_missing_config_is_not_fatal(env):
    (env.channels / "NoConfig").mkdir()

    cp.process_channel("NoConfig")  # не должно бросать

    assert not env.rec.called("transcribe")


def test_no_input_videos_stops_early(env):
    env.setup(["transcribe_video"], with_video=False)

    cp.process_channel("Chan")

    assert not env.rec.called("transcribe")
