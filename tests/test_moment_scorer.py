"""Детерминированный слой отбора кандидатов (analysis/moment_scorer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import moment_scorer as ms


def make_segments(duration: float = 60.0, step: float = 3.0):
    """Плотная речевая дорожка на весь ролик."""

    segments = []
    t = 0.0
    idx = 0
    while t + step <= duration:
        segments.append(
            {
                "start": t,
                "end": t + step * 0.8,
                "text": f"реплика номер {idx} и что дальше?",
                "words": 6,
            }
        )
        t += step
        idx += 1
    return segments


@pytest.fixture
def timeline():
    duration = 60.0
    segments = make_segments(duration)
    audio_summary = {
        "events": [
            {"start": 10.0, "end": 11.0, "peak": 0.9, "sharpness": 0.4, "category": "strong"},
            {"start": 40.0, "end": 41.0, "peak": 0.3, "sharpness": 0.1, "category": "soft"},
        ]
    }
    face_summary = {
        "events": [
            {"start": 5.0, "end": 25.0, "faces": 1, "max_face_ratio": 0.12},
        ]
    }
    cut_times = [4.0, 12.0, 18.0, 30.0, 44.0]
    motion_series = [(float(t), 0.1 + 0.01 * (t % 5)) for t in range(0, 60, 2)]
    hooks = [{"start": 9.5, "end": 11.0, "type": "question", "score": 0.8, "text": "что дальше?"}]

    return ms.build_timeline(
        segments, audio_summary, face_summary, cut_times, motion_series, duration, hooks, cell_s=0.5
    )


class TestScoringConfig:
    def test_defaults_are_merged(self):
        cfg = ms.resolve_scoring_cfg(None)
        assert cfg["cell_s"] == ms.DEFAULT_SCORING["cell_s"]
        assert set(cfg["weights"]) == set(ms.DEFAULT_WEIGHTS)

    def test_user_values_override_defaults(self):
        cfg = ms.resolve_scoring_cfg({"cell_s": 1.0, "max_candidates": 3})
        assert cfg["cell_s"] == 1.0
        assert cfg["max_candidates"] == 3
        assert cfg["step_s"] == ms.DEFAULT_SCORING["step_s"]

    def test_default_weights_sum_to_one(self):
        assert sum(ms.DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


class TestBuildTimeline:
    def test_grid_covers_duration(self, timeline):
        assert timeline is not None
        assert timeline.cells == 120  # 60 c / 0.5 c
        assert timeline.duration == 60.0

    def test_signals_are_filled(self, timeline):
        assert timeline.speech.max() > 0
        assert timeline.audio_energy.max() > 0
        assert timeline.face.max() > 0
        assert timeline.cuts.sum() == 5

    def test_face_window_matches_events(self, timeline):
        lo, hi = timeline.slice_bounds(5.0, 25.0)
        assert timeline.face[lo:hi].mean() > 0.9
        assert timeline.face[:lo].max() == 0.0

    def test_zero_duration_returns_none(self):
        assert ms.build_timeline([], {}, {}, [], [], 0.0, []) is None

    def test_slice_bounds_never_empty(self, timeline):
        lo, hi = timeline.slice_bounds(10.0, 10.0)
        assert hi > lo


class TestWindowSignals:
    def test_signals_are_normalised(self, timeline):
        cfg = ms.resolve_scoring_cfg(None)
        signals, stats = ms.window_signals(timeline, 8.0, 20.0, cfg)

        for key, value in signals.items():
            assert 0.0 <= value <= 1.0, f"сигнал {key} вне [0,1]: {value}"
        assert 0.0 <= stats["speech_density"] <= 1.0
        assert 0.0 <= stats["face_coverage"] <= 1.0

    def test_hook_signal_finds_question(self, timeline):
        cfg = ms.resolve_scoring_cfg(None)
        with_hook, evidence = ms.hook_signal(timeline, 9.5, float(cfg["hook_window_s"]))
        without_hook, _ = ms.hook_signal(timeline, 50.0, float(cfg["hook_window_s"]))

        assert with_hook > without_hook
        assert evidence.get("text")


class TestScoreWindow:
    def test_returns_full_structure(self, timeline):
        cfg = ms.resolve_scoring_cfg(None)
        window = ms.score_window(timeline, 5.0, 20.0, cfg)

        assert window["start"] == 5.0
        assert window["end"] == 20.0
        assert window["duration"] == pytest.approx(15.0)
        assert 0.0 <= window["score"] <= 1.0
        assert set(window["signals"]) == set(ms.DEFAULT_WEIGHTS)
        assert isinstance(window["warnings"], list)

    def test_window_with_speech_scores_higher_than_silence(self):
        duration = 40.0
        speech_only = ms.build_timeline(
            [{"start": 0.0, "end": 18.0, "text": "речь идёт всё время", "words": 40}],
            {}, {}, [], [], duration, [],
            cell_s=0.5,
        )
        cfg = ms.resolve_scoring_cfg(None)

        loud = ms.score_window(speech_only, 0.0, 15.0, cfg)
        silent = ms.score_window(speech_only, 22.0, 37.0, cfg)

        assert loud["score"] > silent["score"]


class TestBuildCandidates:
    def test_candidates_respect_duration_bounds(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)

        assert candidates
        for cand in candidates:
            assert 10.0 - 1e-6 <= cand["duration"] <= 20.0 + 1e-6
            assert cand["end"] <= timeline.duration + 1e-6

    def test_candidates_are_ranked_by_score(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)
        scores = [c["score"] for c in candidates]

        assert scores == sorted(scores, reverse=True)
        assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))

    def test_nms_removes_heavy_overlap(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0, {"nms_overlap": 0.2})

        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
                shortest = min(a["duration"], b["duration"])
                assert inter / shortest <= 0.2 + 1e-6

    def test_max_candidates_is_capped(self, timeline):
        assert len(ms.build_candidates(timeline, 10.0, 20.0, {"max_candidates": 2})) == 2

    def test_disabled_scoring_returns_nothing(self, timeline):
        assert ms.build_candidates(timeline, 10.0, 20.0, {"enabled": False}) == []

    def test_no_timeline_returns_nothing(self):
        assert ms.build_candidates(None, 10.0, 20.0) == []

    def test_episode_shorter_than_min_time_is_scored_whole(self):
        short = ms.build_timeline(
            [{"start": 0.0, "end": 4.0, "text": "коротко", "words": 3}],
            {}, {}, [], [], 5.0, [], cell_s=0.5,
        )

        candidates = ms.build_candidates(short, 30.0, 60.0)

        assert len(candidates) == 1
        assert candidates[0]["start"] == 0.0
        assert candidates[0]["end"] == pytest.approx(5.0)


class TestCandidatePresentation:
    def test_prompt_view_is_compact(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)
        compact = ms.candidates_for_prompt(candidates, limit=3)

        assert len(compact) <= 3
        assert set(compact[0]) >= {"rank", "start", "end", "duration", "score", "signals"}
        assert "stats" not in compact[0]

    def test_title_from_hook_text(self):
        cand = {"stats": {"hook": {"text": "что он сделал, серьёзно?"}}}
        assert ms.candidate_title(cand, "fallback") == "ЧТО ОН СДЕЛАЛ СЕРЬЁЗНО"

    def test_title_falls_back(self):
        assert ms.candidate_title({"stats": {}}, "ЗАПАСНОЕ") == "ЗАПАСНОЕ"

    def test_title_length_is_capped(self):
        cand = {"stats": {"hook": {"text": "слово " * 20}}}
        assert len(ms.candidate_title(cand, "x")) <= 45


class TestMomentsFromCandidates:
    def test_builds_pipeline_ready_moments(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)

        moments = ms.moments_from_candidates(candidates, count=2, fallback_title="ЭПИЗОД")

        assert list(moments) == ["moment_1", "moment_2"]
        first = moments["moment_1"]
        assert first["source"] == "heuristic_candidate"
        assert first["segments"]["segment_1"]["end"] > first["segments"]["segment_1"]["start"]
        assert first["title"]

    def test_empty_candidates(self):
        assert ms.moments_from_candidates([], count=3) == {}

    def test_count_zero(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)
        assert ms.moments_from_candidates(candidates, count=0) == {}


class TestMathHelpers:
    @pytest.mark.parametrize(
        "value,expected", [(-1.0, 0.0), (0.5, 0.5), (2.0, 1.0), (float("nan"), 0.0)]
    )
    def test_clip01(self, value, expected):
        assert ms._clip01(value) == expected

    def test_sweet_spot_peaks_inside_range(self):
        assert ms._sweet_spot(0.6, 0.45, 0.88) == 1.0
        assert ms._sweet_spot(0.1, 0.45, 0.88) < 1.0
        assert ms._sweet_spot(1.5, 0.45, 0.88) < 1.0

    def test_overlap(self):
        assert ms._overlap(0, 10, 5, 20) == 5
        assert ms._overlap(0, 10, 20, 30) == 0

    def test_percentile_of_empty_array(self):
        assert ms._percentile(np.array([]), 90) == 0.0
