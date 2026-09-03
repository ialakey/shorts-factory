"""Этап analyze_moment: сборка сигналов и работа с LLM (сеть замокана)."""

from __future__ import annotations

import json

import numpy as np
import openai
import pytest
from openai.error import AuthenticationError, RateLimitError

from analysis import gpt_analyzer as ga


# ============================================================
# Предобработка сигналов
# ============================================================

class TestExtractTranscriptSegments:
    def test_keeps_valid_segments_and_counts_words(self):
        result = ga._extract_transcript_segments(
            {"segments": [{"start": 0, "end": 2, "text": "  привет   мир "}]}
        )

        assert result == [{"start": 0.0, "end": 2.0, "text": "привет мир", "words": 2}]

    def test_drops_broken_segments(self):
        result = ga._extract_transcript_segments(
            {
                "segments": [
                    {"start": 5, "end": 5, "text": "нулевая длина"},
                    {"start": 9, "end": 4, "text": "конец раньше начала"},
                    {"start": "x", "end": 4, "text": "не число"},
                    {"start": 0, "end": 1, "text": "ок"},
                ]
            }
        )

        assert [seg["text"] for seg in result] == ["ок"]

    def test_no_segments_key(self):
        assert ga._extract_transcript_segments({}) == []


class TestExtractHooks:
    def test_finds_question(self):
        hooks = ga._extract_hooks([{"start": 0, "end": 1, "text": "что он сделал?", "words": 3}])
        assert hooks and hooks[0]["type"] == "question"

    def test_finds_promise(self):
        hooks = ga._extract_hooks([{"start": 0, "end": 1, "text": "сейчас покажу кое-что", "words": 3}])
        assert hooks[0]["type"] == "promise"

    def test_finds_interruption(self):
        hooks = ga._extract_hooks([{"start": 0, "end": 1, "text": "я хотел сказать...", "words": 3}])
        assert hooks[0]["type"] == "interruption"

    def test_plain_text_is_not_a_hook(self):
        assert ga._extract_hooks([{"start": 0, "end": 1, "text": "он пошёл домой", "words": 3}]) == []

    def test_score_is_bounded(self):
        hooks = ga._extract_hooks([{"start": 0, "end": 1, "text": "обещаю, почему...?", "words": 3}])
        assert 0.0 < hooks[0]["score"] <= 0.95


class TestTempoWindows:
    def test_splits_duration_into_windows(self):
        result = ga._build_tempo_windows(
            [{"start": 0.0, "end": 5.0, "text": "речь", "words": 2}],
            duration=12.0, cut_times=[1.0, 7.0], motion_series=[(0.0, 0.2), (6.0, 0.4)],
        )

        assert result["window_s"] == 5.0
        assert [w["start"] for w in result["windows"]] == [0.0, 5.0, 10.0]
        assert result["windows"][0]["speech_density"] == pytest.approx(1.0)
        assert result["windows"][0]["silence_ratio"] == pytest.approx(0.0)
        assert result["windows"][0]["cuts_per_5s"] == 1
        assert result["windows"][1]["cuts_per_5s"] == 1

    def test_zero_duration(self):
        assert ga._build_tempo_windows([], 0.0, [], [])["windows"] == []


class TestEstimateDuration:
    def test_takes_maximum_across_signals(self):
        duration = ga._estimate_duration(
            [{"start": 0, "end": 10, "text": "x", "words": 1}],
            {"stats": {"duration_s": 42.0}},
            {"events": [{"start": 0, "end": 30}]},
        )
        assert duration == 42.0

    def test_no_signals(self):
        assert ga._estimate_duration([], {}, {}) == 0.0


class TestVisualSummary:
    def test_summarises_faces_and_cuts(self):
        summary = ga._build_visual_summary(
            {"events": [{"start": 0.0, "end": 5.0, "faces": 2, "max_face_ratio": 0.2}]},
            cut_times=[1.0, 2.0],
            motion_series=[(0.0, 0.1), (1.0, 0.5)],
            duration=10.0,
        )
        assert isinstance(summary, dict)
        assert summary


class TestDownsampleGray:
    def test_rgb_frame_becomes_square_gray(self):
        frame = (np.random.rand(90, 160, 3) * 255).astype(np.uint8)
        out = ga._downsample_gray(frame, size=16)

        assert out.shape == (16, 16)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_empty_frame(self):
        assert ga._downsample_gray(np.zeros((0, 0, 3), dtype=np.uint8), size=8).shape == (8, 8)


# ============================================================
# analyze_moment: обвязка вокруг LLM
# ============================================================

@pytest.fixture
def analysis_cfg():
    return {
        "gpt": {
            "model": "test-model",
            "min_time": 10,
            "max_time": 20,
            "min_count": 1,
            "max_count": 2,
            "tone": "тест",
            "platforms": "Shorts",
            "audience_age": "19-26",
            "prompt": "{transcript_text}|{candidates}|{min_time}-{max_time}",
        }
    }


@pytest.fixture
def transcript():
    return {
        "segments": [
            {"start": t, "end": t + 2.4, "text": f"реплика {i}, что дальше?"}
            for i, t in enumerate(range(0, 60, 3))
        ]
    }


@pytest.fixture
def stub_signals(monkeypatch):
    """Отключает тяжёлые сигналы: аудио/лица/склейки считаются отдельно."""

    monkeypatch.setattr(
        ga, "analyze_audio_peaks",
        lambda video_path, cfg: {
            "events": [{"start": 5.0, "end": 6.0, "peak": 1.0, "sharpness": 0.4, "category": "strong"}],
            "stats": {"duration_s": 60.0},
            "summary": "ok",
        },
    )
    monkeypatch.setattr(
        ga, "analyze_face_activity",
        lambda video_path, cfg: {
            "events": [{"start": 0.0, "end": 60.0, "faces": 1, "max_face_ratio": 0.1}],
            "summary": "ok",
        },
    )
    monkeypatch.setattr(ga, "_detect_cuts", lambda path, duration, sample_fps=2.0: ([5.0, 25.0], []))
    monkeypatch.setattr(ga.time, "sleep", lambda *_: None)


class FakeCompletion:
    """Отдаёт заранее заданные ответы модели по очереди."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return {"choices": [{"message": {"content": item}}]}


def install_llm(monkeypatch, responses):
    fake = FakeCompletion(responses)
    monkeypatch.setattr(openai.ChatCompletion, "create", staticmethod(fake))
    return fake


VALID_ANSWER = json.dumps(
    {
        "moment_1": {
            "title": "ЧТО ОН СДЕЛАЛ",
            "duration": 15.0,
            "segments": {"segment_1": {"start": 5.0, "end": 20.0}},
        }
    },
    ensure_ascii=False,
)


@pytest.mark.usefixtures("stub_signals")
class TestAnalyzeMoment:
    def test_valid_answer_is_returned(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [VALID_ANSWER])

        result = ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

        assert "moment_1" in result
        assert result["moment_1"]["title"] == "ЧТО ОН СДЕЛАЛ"
        segments = result["moment_1"]["segments"]
        assert segments["segment_1"]["end"] > segments["segment_1"]["start"]

    def test_fenced_json_is_parsed(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [f"вот ответ:\n```json\n{VALID_ANSWER}\n```"])

        result = ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

        assert "moment_1" in result

    def test_model_name_from_config_is_used(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        fake = install_llm(monkeypatch, [VALID_ANSWER])

        ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

        assert fake.calls[0]["model"] == "test-model"

    def test_intermediate_artifacts_are_written(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [VALID_ANSWER])
        out = tmp_path / "out"

        ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key", output_dir=out)

        assert (out / "ep_chatgpt_payload.txt").exists()
        assert (out / "ep_candidates.json").exists()
        signals = json.loads((out / "ep_signals.json").read_text(encoding="utf-8"))
        assert signals["duration_s"] > 0
        assert signals["cuts"] == [5.0, 25.0]

    def test_retry_after_unusable_answer(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        fake = install_llm(monkeypatch, ["{}", VALID_ANSWER])

        result = ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

        assert len(fake.calls) == 2
        assert "moment_1" in result

    def test_broken_json_falls_back_to_heuristics(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        fake = install_llm(monkeypatch, ["не json совсем"])

        result = ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

        assert len(fake.calls) == 3, "должно быть 3 попытки перед fallback"
        assert result, "при живых эвристических кандидатах пайплайн не должен падать"
        assert result["moment_1"]["source"] == "heuristic_candidate"

    def test_auth_error_is_explicit(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [AuthenticationError("bad token")])

        with pytest.raises(ValueError, match="авторизации"):
            ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

    def test_quota_error_is_explicit(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [RateLimitError("You exceeded your current quota")])

        with pytest.raises(ValueError, match="[Кк]вота"):
            ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key")

    def test_payload_contains_every_signal_block(self, monkeypatch, transcript, analysis_cfg, tmp_path):
        install_llm(monkeypatch, [VALID_ANSWER])
        out = tmp_path / "out"

        ga.analyze_moment(transcript, tmp_path / "ep.mp4", analysis_cfg, "key", output_dir=out)

        payload = (out / "ep_chatgpt_payload.txt").read_text(encoding="utf-8")
        for block in ("ТРАНСКРИПТ", "АУДИО", "ЛИЦА", "ВИЗУАЛ", "ТЕМП", "ЭМОЦИИ", "HOOKS", "КАНДИДАТЫ"):
            assert block in payload, f"в payload нет блока {block}"
