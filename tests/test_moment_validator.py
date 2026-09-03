"""Валидация и «починка» ответа LLM (analysis/moment_validator.py)."""

from __future__ import annotations

import pytest

from analysis import moment_scorer as ms
from analysis import moment_validator as mv


@pytest.fixture
def cfg():
    return mv.resolve_validation_cfg(None)


@pytest.fixture
def timeline():
    duration = 60.0
    segments = [
        {"start": t, "end": t + 2.4, "text": "реплика и вопрос?", "words": 5}
        for t in range(0, 60, 3)
    ]
    audio = {"events": [{"start": 5.0, "end": 6.0, "peak": 1.0, "sharpness": 0.5, "category": "strong"}]}
    faces = {"events": [{"start": 0.0, "end": 60.0, "faces": 1, "max_face_ratio": 0.1}]}
    return ms.build_timeline(segments, audio, faces, [10.0, 20.0], [], duration, [], cell_s=0.5)


@pytest.fixture
def empty_timeline():
    """Ролик без речи и без звука — любой момент должен признаваться пустым."""

    return ms.build_timeline([], {}, {}, [], [], 60.0, [], cell_s=0.5)


class TestNormalizeSegments:
    def test_dict_is_ordered_by_segment_index(self):
        value = {
            "segment_2": {"start": 10, "end": 12},
            "segment_1": {"start": 1, "end": 3},
        }
        assert mv.normalize_segments(value) == [(1.0, 3.0), (10.0, 12.0)]

    def test_list_keeps_order(self):
        value = [{"start": 5, "end": 6}, {"start": 1, "end": 2}]
        assert mv.normalize_segments(value) == [(5.0, 6.0), (1.0, 2.0)]

    def test_invalid_pairs_are_dropped(self):
        value = [
            {"start": 5, "end": 5},          # нулевая длина
            {"start": "abc", "end": 3},      # не число
            {"start": 9, "end": 4},          # конец раньше начала
            {"start": 1, "end": 2},
        ]
        assert mv.normalize_segments(value) == [(1.0, 2.0)]

    @pytest.mark.parametrize("value", [None, "строка", 42])
    def test_unsupported_types(self, value):
        assert mv.normalize_segments(value) == []


class TestNormalizeLlmMoments:
    def test_dict_of_moments(self):
        data = {"moment_1": {"title": "A", "segments": {"segment_1": {"start": 0, "end": 5}}}}
        moments = mv.normalize_llm_moments(data)

        assert len(moments) == 1
        assert moments[0]["key"] == "moment_1"
        assert moments[0]["title"] == "A"
        assert moments[0]["segments"] == [(0.0, 5.0)]

    def test_list_of_moments_gets_synthetic_keys(self):
        moments = mv.normalize_llm_moments([{"segments": [{"start": 0, "end": 4}]}])
        assert moments[0]["key"] == "moment_1"

    def test_flat_start_end_without_segments(self):
        moments = mv.normalize_llm_moments({"moment_1": {"start": 2, "end": 9}})
        assert moments[0]["segments"] == [(2.0, 9.0)]

    def test_json_string_is_parsed(self):
        moments = mv.normalize_llm_moments('{"moment_1": {"start": 1, "end": 4}}')
        assert moments[0]["segments"] == [(1.0, 4.0)]

    def test_broken_json_returns_empty(self):
        assert mv.normalize_llm_moments("{not json") == []

    def test_moments_without_usable_segments_are_dropped(self):
        assert mv.normalize_llm_moments({"moment_1": {"title": "нет таймингов"}}) == []


class TestRepairSegments:
    def test_clamps_to_source_duration(self, cfg):
        result, notes = mv.repair_segments(
            [(50.0, 120.0)], source_duration=60.0, min_time=5.0, max_time=20.0, cfg=cfg
        )
        assert result
        assert result[-1][1] <= 60.0

    def test_removes_overlap_with_previous_segment(self, cfg):
        result, notes = mv.repair_segments(
            [(0.0, 10.0), (5.0, 15.0)],
            source_duration=60.0, min_time=5.0, max_time=30.0, cfg=cfg,
        )
        for i in range(len(result) - 1):
            assert result[i][1] <= result[i + 1][0] + 1e-6

    def test_trims_assembly_longer_than_max_time(self, cfg):
        result, notes = mv.repair_segments(
            [(0.0, 40.0)], source_duration=60.0, min_time=10.0, max_time=20.0, cfg=cfg
        )
        total = sum(end - start for start, end in result)
        assert total <= 20.0 + 1e-3
        assert "сборка подрезана под max_time" in notes

    def test_extends_assembly_shorter_than_min_time(self, cfg):
        result, notes = mv.repair_segments(
            [(10.0, 13.0)], source_duration=60.0, min_time=15.0, max_time=25.0, cfg=cfg
        )
        total = sum(end - start for start, end in result)
        assert total >= 15.0 - 1e-3

    def test_gives_up_when_source_is_too_short(self, cfg):
        result, notes = mv.repair_segments(
            [(0.0, 4.0)], source_duration=5.0, min_time=30.0, max_time=60.0, cfg=cfg
        )
        assert result == []
        assert "не удалось дотянуть до min_time" in notes

    def test_adjacent_segments_are_merged(self, cfg):
        result, _ = mv.repair_segments(
            [(0.0, 10.0), (10.0, 18.0)],
            source_duration=60.0, min_time=5.0, max_time=30.0, cfg=cfg,
        )
        assert result == [(0.0, 18.0)]

    def test_snaps_to_phrase_boundaries(self, cfg):
        snap_points = [0.0, 12.0, 30.0]
        result, _ = mv.repair_segments(
            [(12.3, 29.7)],
            source_duration=60.0, min_time=5.0, max_time=30.0, cfg=cfg,
            snap_points=snap_points,
        )
        assert result[0][0] == pytest.approx(12.0)
        assert result[0][1] == pytest.approx(30.0)


class TestBuildSnapPoints:
    def test_collects_speech_and_cut_boundaries(self, cfg):
        points = mv.build_snap_points(
            [{"start": 1.0, "end": 3.0}], [2.0, 9.5], cfg
        )
        assert points == [1.0, 2.0, 3.0, 9.5]

    def test_can_be_disabled(self):
        cfg = mv.resolve_validation_cfg({"snap_to_speech": False, "snap_to_cuts": False})
        assert mv.build_snap_points([{"start": 1.0, "end": 3.0}], [2.0], cfg) == []


class TestEvaluateMoment:
    def test_good_moment_has_score_and_no_fatal(self, timeline, cfg):
        scoring = ms.resolve_scoring_cfg(None)
        result = mv.evaluate_moment([(0.0, 20.0)], timeline, scoring, cfg)

        assert result["score"] > 0
        assert result["fatal"] == []
        assert set(result["signals"]) == set(ms.DEFAULT_WEIGHTS)

    def test_empty_fragment_is_fatal(self, empty_timeline, cfg):
        scoring = ms.resolve_scoring_cfg(None)
        result = mv.evaluate_moment([(0.0, 20.0)], empty_timeline, scoring, cfg)

        assert result["fatal"], "фрагмент без речи и звука должен отбраковываться"

    def test_no_timeline_is_neutral(self, cfg):
        result = mv.evaluate_moment([(0.0, 5.0)], None, ms.resolve_scoring_cfg(None), cfg)
        assert result["score"] == 0.0


class TestSelectMoments:
    def _select(self, data, timeline, candidates=(), **kwargs):
        params = dict(
            timeline=timeline,
            candidates=list(candidates),
            transcript_segments=[],
            cut_times=[],
            source_duration=60.0,
            min_time=10.0,
            max_time=20.0,
            min_count=1,
            max_count=2,
            scoring_cfg=ms.resolve_scoring_cfg(None),
            fallback_title="ЭПИЗОД",
            verbose=False,
        )
        params.update(kwargs)
        return mv.select_moments(data, **params)

    def test_valid_llm_moment_survives(self, timeline):
        data = {"moment_1": {"title": "ХОРОШИЙ", "segments": {"segment_1": {"start": 5, "end": 20}}}}

        result, _ = self._select(data, timeline)

        assert list(result) == ["moment_1"]
        assert result["moment_1"]["title"] == "ХОРОШИЙ"
        assert result["moment_1"]["source"] == "llm"
        assert result["moment_1"]["duration"] == pytest.approx(15.0, abs=0.5)
        assert result["moment_1"]["segments"]["segment_1"]["end"] > 0

    def test_out_of_range_moment_is_repaired_not_dropped(self, timeline):
        data = {"moment_1": {"title": "ДЛИННЫЙ", "segments": {"segment_1": {"start": 0, "end": 55}}}}

        result, _ = self._select(data, timeline)

        assert result
        assert result["moment_1"]["duration"] <= 20.0 + 1e-3

    def test_max_count_is_respected(self, timeline):
        data = {
            f"moment_{i}": {"title": f"M{i}", "segments": {"segment_1": {"start": i * 15, "end": i * 15 + 12}}}
            for i in range(1, 5)
        }

        result, _ = self._select(data, timeline, max_count=2)

        assert len(result) == 2

    def test_garbage_is_filled_from_candidates(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)
        assert candidates

        result, log = self._select("полный мусор", timeline, candidates=candidates, min_count=1)

        assert result
        assert result["moment_1"]["source"] == "heuristic_candidate"

    def test_fill_can_be_disabled(self, timeline):
        candidates = ms.build_candidates(timeline, 10.0, 20.0)

        result, _ = self._select(
            {}, timeline, candidates=candidates,
            validation_cfg={"fill_from_candidates": False},
        )

        assert result == {}

    def test_empty_timeline_rejects_everything(self, empty_timeline):
        data = {"moment_1": {"title": "ПУСТО", "segments": {"segment_1": {"start": 0, "end": 15}}}}

        result, log = self._select(data, empty_timeline)

        assert result == {}
        assert any("отброшен" in line for line in log)

    def test_result_shape_matches_pipeline_contract(self, timeline):
        data = {"moment_1": {"title": "OK", "segments": {"segment_1": {"start": 5, "end": 20}}}}

        result, _ = self._select(data, timeline)
        moment = result["moment_1"]

        assert set(moment) >= {"title", "duration", "segments", "score", "signals", "source"}
        for name, seg in moment["segments"].items():
            assert name.startswith("segment_")
            assert seg["end"] > seg["start"]
