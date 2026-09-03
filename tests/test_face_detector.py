"""Этап dynamic_shorts: детекция лиц и виртуальная камера."""

from __future__ import annotations

import math

import numpy as np
import pytest
from moviepy.editor import ColorClip

from rendering import face_detector as fd


class TestResolveCameraConfig:
    def test_defaults_are_complete(self):
        cfg = fd.resolve_camera_config(None)
        for key in ("target_width", "target_height", "analysis_fps", "follow_stiffness"):
            assert key in cfg

    def test_profile_overrides_base_defaults(self):
        static = fd.resolve_camera_config({"profile": "static"})
        assert static["profile"] == "static"
        assert static["follow_stiffness"] == fd.CAMERA_PROFILES["static"]["follow_stiffness"]

    def test_explicit_keys_win_over_profile(self):
        cfg = fd.resolve_camera_config({"profile": "static", "follow_stiffness": 99.0})
        assert cfg["follow_stiffness"] == 99.0

    def test_unknown_profile_falls_back_to_operator(self):
        cfg = fd.resolve_camera_config({"profile": "нет такого"})
        assert cfg["profile"] == "operator"
        assert cfg["follow_stiffness"] == fd.CAMERA_PROFILES["operator"]["follow_stiffness"]

    def test_none_values_do_not_erase_defaults(self):
        cfg = fd.resolve_camera_config({"follow_damping": None})
        assert cfg["follow_damping"] is not None


class TestFaceSaliency:
    def _face(self, center, size, score):
        return fd.FaceDetection(
            center=np.array(center, np.float32),
            size=np.array(size, np.float32),
            score=score,
            box=np.array([0, 0, 1, 1], np.float32),
        )

    def test_bigger_and_more_confident_face_wins(self):
        big = self._face((0.5, 0.5), (0.3, 0.3), 0.95)
        small = self._face((0.5, 0.5), (0.05, 0.05), 0.6)
        assert fd._face_saliency(big) > fd._face_saliency(small)

    def test_central_face_wins_over_edge(self):
        centre = self._face((0.5, 0.5), (0.2, 0.2), 0.9)
        edge = self._face((0.02, 0.02), (0.2, 0.2), 0.9)
        assert fd._face_saliency(centre) > fd._face_saliency(edge)


class TestSelectPrimaryFace:
    def _face(self, center, size=(0.2, 0.2), score=0.9):
        return fd.FaceDetection(
            center=np.array(center, np.float32),
            size=np.array(size, np.float32),
            score=score,
            box=np.array([0, 0, 1, 1], np.float32),
        )

    def test_no_detections(self):
        face, pending, switched = fd._select_primary_face(
            [], None, match_tolerance=0.2, switch_margin=0.15,
            switch_pending=0.0, switch_hold_s=0.5, dt=0.125,
        )
        assert face is None and switched is False

    def test_keeps_tracking_nearest_face(self):
        tracked = self._face((0.5, 0.5))
        nearby = self._face((0.52, 0.5))

        face, _, switched = fd._select_primary_face(
            [nearby], tracked, match_tolerance=0.2, switch_margin=0.15,
            switch_pending=0.0, switch_hold_s=0.5, dt=0.125,
        )

        assert face is nearby
        assert switched is False

    def test_switch_requires_holding_advantage(self):
        tracked = self._face((0.5, 0.5), size=(0.1, 0.1), score=0.7)
        same = self._face((0.5, 0.5), size=(0.1, 0.1), score=0.7)
        better = self._face((0.9, 0.5), size=(0.35, 0.35), score=0.99)

        face, pending, switched = fd._select_primary_face(
            [same, better], tracked, match_tolerance=0.3, switch_margin=0.1,
            switch_pending=0.0, switch_hold_s=0.5, dt=0.125,
        )
        assert switched is False, "мгновенный перескок на другого героя запрещён"
        assert pending > 0

        face, _, switched = fd._select_primary_face(
            [same, better], tracked, match_tolerance=0.3, switch_margin=0.1,
            switch_pending=0.5, switch_hold_s=0.5, dt=0.125,
        )
        assert switched is True
        assert face is better

    def test_lost_track_reacquires_best_face(self):
        tracked = self._face((0.1, 0.1))
        far = self._face((0.9, 0.9))

        face, _, switched = fd._select_primary_face(
            [far], tracked, match_tolerance=0.05, switch_margin=0.15,
            switch_pending=0.0, switch_hold_s=0.5, dt=0.125,
        )

        assert face is far
        assert switched is True


class TestKenBurns:
    def test_offset_is_bounded_by_amplitudes(self):
        for t in np.linspace(0, 24, 50):
            offset, zoom = fd._ken_burns_offset(
                float(t), 12.0, pan_amplitude=0.03, tilt_amplitude=0.02, zoom_amplitude=0.05
            )
            assert abs(offset[0]) <= 0.03 + 1e-6
            assert abs(offset[1]) <= 0.02 + 1e-6
            assert 0.8 <= zoom <= 1.05 + 1e-6

    def test_zero_amplitudes_freeze_the_frame(self):
        offset, zoom = fd._ken_burns_offset(
            5.0, 12.0, pan_amplitude=0.0, tilt_amplitude=0.0, zoom_amplitude=0.0
        )
        assert np.allclose(offset, 0.0)
        assert zoom == 1.0

    def test_motion_is_periodic(self):
        a = fd._ken_burns_offset(0.0, 10.0, pan_amplitude=0.1, tilt_amplitude=0.0, zoom_amplitude=0.0)[0]
        b = fd._ken_burns_offset(10.0, 10.0, pan_amplitude=0.1, tilt_amplitude=0.0, zoom_amplitude=0.0)[0]
        assert np.allclose(a, b, atol=1e-5)

    def test_zero_period_does_not_divide_by_zero(self):
        offset, zoom = fd._ken_burns_offset(
            1.0, 0.0, pan_amplitude=0.1, tilt_amplitude=0.1, zoom_amplitude=0.1
        )
        assert np.isfinite(offset).all()
        assert math.isfinite(zoom)


class TestInterpolateStates:
    def _states(self):
        return [
            fd.CameraState(time=0.0, center=np.array([0.0, 0.0], np.float32), zoom=1.0, has_face=True),
            fd.CameraState(time=1.0, center=np.array([1.0, 2.0], np.float32), zoom=2.0, has_face=True),
        ]

    def test_endpoints_are_exact(self):
        path = fd._interpolate_states(self._states())
        centre, zoom = path(0.0)
        assert np.allclose(centre, [0.0, 0.0]) and zoom == 1.0
        centre, zoom = path(1.0)
        assert np.allclose(centre, [1.0, 2.0]) and zoom == 2.0

    def test_midpoint_is_linear(self):
        centre, zoom = fd._interpolate_states(self._states())(0.5)
        assert np.allclose(centre, [0.5, 1.0])
        assert zoom == pytest.approx(1.5)

    def test_outside_range_is_clamped(self):
        path = fd._interpolate_states(self._states())
        assert np.allclose(path(-10.0)[0], [0.0, 0.0])
        assert np.allclose(path(99.0)[0], [1.0, 2.0])


class TestDetectorBackend:
    def test_blank_frame_has_no_faces(self):
        backend = fd._DetectorBackend(0.75)
        try:
            frame = np.full((180, 320, 3), 127, np.uint8)
            assert backend.detect(frame, min_size=16.0) == []
        finally:
            backend.close()

    def test_noise_frame_does_not_crash(self):
        backend = fd._DetectorBackend(0.75)
        try:
            frame = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
            detections = backend.detect(frame, min_size=16.0)
            assert isinstance(detections, list)
            for det in detections:
                assert 0.0 <= det.center[0] <= 1.0
                assert 0.0 <= det.center[1] <= 1.0
        finally:
            backend.close()


class TestBuildDynamicShortClip:
    def test_produces_vertical_clip(self):
        source = ColorClip(size=(320, 180), color=(40, 90, 140), duration=1.0).set_fps(12)

        result = fd.build_dynamic_short_clip(
            source, {"target_width": 72, "target_height": 128, "analysis_fps": 4}
        )

        assert result.analysis_points > 0
        assert (result.clip.w, result.clip.h) == (72, 128)
        frame = result.clip.get_frame(0.5)
        assert frame.shape == (128, 72, 3)

    def test_without_faces_falls_back_to_ken_burns(self):
        source = ColorClip(size=(320, 180), color=(10, 10, 10), duration=1.0).set_fps(12)

        result = fd.build_dynamic_short_clip(
            source, {"target_width": 72, "target_height": 128, "analysis_fps": 4}
        )

        assert result.used_face_track is False
        assert result.zoom_range[0] <= result.zoom_range[1]

    def test_missing_clip_raises(self):
        with pytest.raises(ValueError):
            fd.build_dynamic_short_clip(None)


@pytest.mark.ffmpeg
class TestAnalyzeFaceActivity:
    def test_returns_events_structure(self, sample_video):
        result = fd.analyze_face_activity(sample_video, {"gpt": {"face_analysis_interval_s": 1.0}})

        assert result["summary"] == "ok"
        assert isinstance(result["events"], list)
        assert result["analysis_interval_s"] == pytest.approx(1.0)
        for event in result["events"]:
            assert event["end"] > event["start"]
            assert event["faces"] >= 1

    def test_broken_video_degrades_gracefully(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")

        result = fd.analyze_face_activity(broken, {})

        assert result["events"] == []
        assert result["summary"].startswith("error") or result["summary"] == "no_video"
