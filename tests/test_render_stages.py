"""Реальный рендер: каждый этап make_clips должен отдавать валидный клип.

Это самый «дорогой», но и самый честный слой тестов: сюда попадают ошибки,
которые не ловятся моками — несовместимость moviepy, битые слои, падение
кодека, неверная геометрия полотна.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from moviepy.editor import VideoFileClip

from rendering.video_editor import make_clips

pytestmark = [pytest.mark.render, pytest.mark.ffmpeg]

#: Этапы, которые применяются уже поверх нарезанного клипа.
POST_CLIP_STAGES = [
    "dynamic_shorts",
    "watermark",
    "enhance_audio",
    "video_effects",
    "music",
    "subtitles",
    "title",
]


@pytest.fixture
def source_video(channel_dir, sample_video) -> Path:
    target = channel_dir / "input_videos" / "episode.mp4"
    shutil.copy(sample_video, target)
    return target


def render(video, moments, channel_dir, cfg, pipeline):
    out = channel_dir / "output_clips"
    make_clips(video, moments, out, cfg, pipeline=pipeline, channel_dir=channel_dir)
    return sorted(out.glob("*.mp4"))


def assert_playable(path: Path, expected_size):
    assert path.stat().st_size > 0, "клип пустой"
    with VideoFileClip(str(path)) as clip:
        assert clip.duration > 0
        assert (clip.w, clip.h) == expected_size
        clip.get_frame(min(0.1, clip.duration / 2))


@pytest.mark.parametrize("stage", POST_CLIP_STAGES)
def test_each_stage_renders_valid_clip(stage, source_video, channel_dir, render_config, single_moment):
    clips = render(source_video, single_moment, channel_dir, render_config, ["make_clips", stage])

    assert len(clips) == 1, f"этап {stage} не отдал клип"
    assert clips[0].name == "TEST_CLIP.mp4"
    assert_playable(clips[0], (216, 384))


def test_full_pipeline_renders(source_video, channel_dir, render_config, single_moment):
    clips = render(
        source_video, single_moment, channel_dir, render_config,
        ["make_clips"] + POST_CLIP_STAGES,
    )

    assert len(clips) == 1
    assert_playable(clips[0], (216, 384))


def test_montage_from_several_segments(source_video, channel_dir, render_config):
    """Момент собирается из нескольких кусков — длительность складывается."""

    moments = {
        "moment_1": {
            "title": "MONTAGE",
            "segments": {
                "segment_1": {"start": 0.0, "end": 0.5},
                "segment_2": {"start": 1.0, "end": 1.5},
            },
        }
    }

    clips = render(source_video, moments, channel_dir, render_config, ["make_clips"])

    assert len(clips) == 1
    with VideoFileClip(str(clips[0])) as clip:
        assert clip.duration == pytest.approx(1.0, abs=0.25)


def test_flat_start_end_moment_is_supported(source_video, channel_dir, render_config):
    moments = {"moment_1": {"title": "FLAT", "start": 0.2, "end": 1.2}}

    clips = render(source_video, moments, channel_dir, render_config, ["make_clips"])

    assert [c.name for c in clips] == ["FLAT.mp4"]


def test_several_moments_produce_several_clips(source_video, channel_dir, render_config):
    moments = {
        "moment_1": {"title": "FIRST", "segments": {"segment_1": {"start": 0.0, "end": 0.8}}},
        "moment_2": {"title": "SECOND", "segments": {"segment_1": {"start": 1.0, "end": 1.8}}},
    }

    clips = render(source_video, moments, channel_dir, render_config, ["make_clips"])

    assert [c.name for c in clips] == ["FIRST.mp4", "SECOND.mp4"]


def test_static_background_mode_renders(source_video, channel_dir, render_config, single_moment):
    """Ветка со статичной картинкой вместо размытого фона."""

    render_config["background"]["use_blurred_background"] = False

    clips = render(source_video, single_moment, channel_dir, render_config, ["make_clips"])

    assert len(clips) == 1
    assert_playable(clips[0], (216, 384))


def test_subtitles_use_configured_whisper_model(source_video, channel_dir, render_config,
                                                single_moment, whisper_stub):
    render_config["subtitles"]["whisper_model"] = "small"

    render(source_video, single_moment, channel_dir, render_config, ["make_clips", "subtitles"])

    assert whisper_stub.loaded_models == ["small"]
    assert whisper_stub.calls[0]["kwargs"]["word_timestamps"] is True


def test_broken_source_video_is_skipped(channel_dir, render_config, single_moment):
    broken = channel_dir / "input_videos" / "broken.mp4"
    broken.write_bytes(b"not a video")

    clips = render(broken, single_moment, channel_dir, render_config, ["make_clips"])

    assert clips == [], "битый исходник не должен ронять весь канал"


def test_missing_music_folder_is_reported(source_video, channel_dir, render_config, single_moment):
    shutil.rmtree(channel_dir / "assets" / "musics")

    with pytest.raises(FileNotFoundError, match="musics"):
        render(source_video, single_moment, channel_dir, render_config, ["make_clips", "music"])
