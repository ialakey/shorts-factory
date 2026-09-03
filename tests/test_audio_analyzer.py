"""Сигнал «аудио»: RMS/ZCR/центроид и склейка пиков в события."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import audio_analyzer as aa


class TestFrameSignal:
    def test_splits_into_full_windows(self):
        frames = aa._frame_signal(np.arange(10, dtype=np.float32), 4)
        assert frames.shape == (2, 4)

    def test_short_signal_gives_no_frames(self):
        assert aa._frame_signal(np.zeros(3, dtype=np.float32), 8).size == 0


class TestSpectralFeatures:
    def test_rms_of_constant_signal(self):
        frames = np.full((3, 8), 0.5, dtype=np.float32)
        assert np.allclose(aa._rms(frames), 0.5)

    def test_silence_has_zero_rms(self):
        assert np.allclose(aa._rms(np.zeros((2, 8), dtype=np.float32)), 0.0)

    def test_zcr_higher_for_alternating_signal(self):
        alternating = np.tile([1.0, -1.0], (1, 16)).astype(np.float32)
        flat = np.ones((1, 32), dtype=np.float32)
        assert aa._zcr(alternating)[0] > aa._zcr(flat)[0]

    def test_centroid_higher_for_higher_tone(self):
        sr = 8000
        t = np.arange(sr // 10) / sr
        low = np.sin(2 * np.pi * 200 * t).astype(np.float32)[None, :]
        high = np.sin(2 * np.pi * 2000 * t).astype(np.float32)[None, :]

        assert aa._spectral_centroid(high, sr)[0] > aa._spectral_centroid(low, sr)[0]


class TestMergeEvents:
    def test_builds_event_from_mask(self):
        mask = np.array([False, True, True, True, False])
        strength = np.array([0.0, 0.5, 0.9, 0.4, 0.0])

        events = aa._merge_events_from_mask(mask, strength, frame_duration=0.1, min_event_ms=50)

        assert len(events) == 1
        assert events[0]["start"] == pytest.approx(0.1)
        assert events[0]["end"] == pytest.approx(0.4)
        assert events[0]["peak"] == pytest.approx(0.9)
        assert events[0]["frames"] == 3

    def test_short_events_are_dropped(self):
        mask = np.array([False, True, False])
        strength = np.array([0.0, 1.0, 0.0])

        assert aa._merge_events_from_mask(mask, strength, 0.01, min_event_ms=100) == []

    def test_event_open_at_end_is_closed(self):
        mask = np.array([False, True, True])
        strength = np.array([0.0, 0.7, 0.8])

        events = aa._merge_events_from_mask(mask, strength, 0.1, min_event_ms=50)

        assert len(events) == 1
        assert events[0]["end"] == pytest.approx(0.3)

    def test_empty_mask(self):
        assert aa._merge_events_from_mask(np.zeros(5, bool), np.zeros(5), 0.1, 10) == []


class TestLabelPeak:
    def test_labels_are_stable(self):
        assert isinstance(aa._label_peak(0.9, 1.0), str)
        assert aa._label_peak(1.0, 1.0) != aa._label_peak(0.05, 1.0)


@pytest.mark.ffmpeg
class TestAnalyzeAudioPeaks:
    def test_returns_stats_for_real_video(self, sample_video):
        result = aa.analyze_audio_peaks(sample_video, {})

        assert result["summary"] in {"ok", "no_peaks"}
        stats = result["stats"]
        assert stats["duration_s"] > 1.0
        assert stats["frames"] > 0
        assert stats["rms_max"] >= stats["rms_mean"] >= 0.0
        for event in result["events"]:
            assert event["end"] > event["start"]
            assert event["category"] in {"strong", "soft"}
            assert "hint" in event

    def test_broken_file_degrades_gracefully(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"definitely not a video")

        result = aa.analyze_audio_peaks(broken, {})

        assert result["events"] == []
        assert isinstance(result["summary"], str)

    def test_config_overrides_are_respected(self, sample_video):
        result = aa.analyze_audio_peaks(
            sample_video, {"audio_analysis": {"window_ms": 200, "max_events": 3}}
        )

        assert result["window_ms"] == 200
        assert len(result["events"]) <= 3
