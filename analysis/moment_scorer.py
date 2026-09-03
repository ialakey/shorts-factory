"""Детерминированный многослойный отбор кандидатов на вирусные моменты.

Идея: каждый отдельный сигнал — речь, звук, лица, смены сцен, темп, hook —
шумный и сам по себе ненадёжный. Но их линейная композиция резко снижает
вероятность системной ошибки: LLM получает не «сырое» видео, а уже
отранжированные окна-кандидаты с расшифровкой сигналов и работает как слой
агрегации, а не как единственный мозг пайплайна.

Модуль полностью детерминирован и не ходит в сеть: его результат используется
и как подсказка для LLM, и как fallback, если модель вернула мусор.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Веса сигналов по умолчанию (в сумме 1.0) ------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "transcript": 0.24,
    "audio": 0.20,
    "face": 0.14,
    "scene": 0.12,
    "pacing": 0.14,
    "hook": 0.16,
}

DEFAULT_SCORING: Dict[str, object] = {
    "enabled": True,
    "cell_s": 0.5,               # разрешение временной сетки сигналов
    "step_s": 1.0,               # шаг скользящего окна кандидатов
    "length_steps": 3,           # сколько длительностей пробуем в диапазоне min..max
    "hook_window_s": 2.0,        # окно анализа «входа» в клип
    "max_candidates": 12,
    "nms_overlap": 0.45,         # порог подавления пересекающихся кандидатов
    "min_speech_density": 0.18,  # ниже — считаем окно «пустым» и штрафуем
    "min_face_coverage": 0.0,
    "speech_sweet_spot": [0.45, 0.88],
    "cuts_per_min_sweet_spot": [6.0, 40.0],
    "max_silence_gap_s": 2.5,
    "target_face_ratio": 0.10,   # площадь лица, при которой face-сигнал = 1.0
    "words_per_s_ref": 3.0,
    "empty_window_penalty": 0.55,
    "low_face_penalty": 0.85,
    "weights": dict(DEFAULT_WEIGHTS),
}


# ============================================================
# Мелкие математические помощники
# ============================================================

def _clip01(value: float) -> float:
    value = float(value)
    if value != value:  # NaN
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _sweet_spot(value: float, low: float, high: float, *, falloff: float = 1.0) -> float:
    """1.0 внутри «полезного» диапазона, плавный спад за его пределами."""
    if high < low:
        low, high = high, low
    if low <= value <= high:
        return 1.0
    span = max(1e-6, (high - low) * falloff)
    if value < low:
        return _clip01(1.0 - (low - value) / span)
    return _clip01(1.0 - (value - high) / span)


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size else 0.0


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ============================================================
# Временная сетка сигналов
# ============================================================

@dataclass
class SignalTimeline:
    """Поячеечная развёртка всех сигналов эпизода."""

    cell_s: float
    duration: float
    speech: np.ndarray           # доля ячейки, занятая речью
    words: np.ndarray            # слов в ячейке
    audio_energy: np.ndarray     # нормированная энергия аудиособытий
    audio_sharpness: np.ndarray
    face: np.ndarray             # доля ячейки с лицом в кадре
    face_ratio: np.ndarray       # максимальная площадь лица
    cuts: np.ndarray             # количество склеек
    motion: np.ndarray           # интенсивность движения
    hooks: List[dict] = field(default_factory=list)
    motion_ref: float = 0.15

    @property
    def cells(self) -> int:
        return int(self.speech.size)

    def slice_bounds(self, start: float, end: float) -> Tuple[int, int]:
        lo = int(max(0, math.floor(start / self.cell_s)))
        hi = int(min(self.cells, math.ceil(end / self.cell_s)))
        if hi <= lo:
            hi = min(self.cells, lo + 1)
        return lo, hi


def _fill_from_events(
        target: np.ndarray,
        events: Sequence[dict],
        cell_s: float,
        value_getter,
        *,
        mode: str = "max",
) -> None:
    cells = target.size
    for event in events or []:
        try:
            start = float(event.get("start", 0.0) or 0.0)
            end = float(event.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        value = float(value_getter(event) or 0.0)
        lo = int(max(0, math.floor(start / cell_s)))
        hi = int(min(cells, math.ceil(end / cell_s)))
        for idx in range(lo, hi):
            cell_start = idx * cell_s
            covered = _overlap(cell_start, cell_start + cell_s, start, end) / cell_s
            if covered <= 0:
                continue
            if mode == "max":
                target[idx] = max(target[idx], value)
            elif mode == "coverage":
                target[idx] = min(1.0, target[idx] + covered)
            else:
                target[idx] += value * covered


def build_timeline(
        segments: Sequence[dict],
        audio_summary: dict,
        face_summary: dict,
        cut_times: Sequence[float],
        motion_series: Sequence[Tuple[float, float]],
        duration: float,
        hooks: Sequence[dict],
        cell_s: float = 0.5,
) -> Optional[SignalTimeline]:
    """Сводит все сигналы эпизода в единую временную сетку."""

    duration = float(duration or 0.0)
    cell_s = max(0.1, float(cell_s))
    if duration <= 0:
        return None

    cells = int(math.ceil(duration / cell_s))
    if cells <= 0:
        return None

    speech = np.zeros(cells, dtype=np.float32)
    words = np.zeros(cells, dtype=np.float32)
    audio_energy = np.zeros(cells, dtype=np.float32)
    audio_sharpness = np.zeros(cells, dtype=np.float32)
    face = np.zeros(cells, dtype=np.float32)
    face_ratio = np.zeros(cells, dtype=np.float32)
    cuts = np.zeros(cells, dtype=np.float32)
    motion = np.zeros(cells, dtype=np.float32)

    # --- речь ---
    for seg in segments or []:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        seg_words = float(seg.get("words", 0) or 0)
        seg_len = end - start
        lo = int(max(0, math.floor(start / cell_s)))
        hi = int(min(cells, math.ceil(end / cell_s)))
        for idx in range(lo, hi):
            cell_start = idx * cell_s
            covered = _overlap(cell_start, cell_start + cell_s, start, end)
            if covered <= 0:
                continue
            speech[idx] = min(1.0, speech[idx] + covered / cell_s)
            words[idx] += seg_words * (covered / seg_len)

    # --- аудиособытия ---
    audio_events = audio_summary.get("events", []) if isinstance(audio_summary, dict) else []
    peaks = [float(e.get("peak", 0.0) or 0.0) for e in audio_events]
    max_peak = max(peaks) if peaks else 0.0
    if max_peak > 0:
        def _energy(event: dict) -> float:
            base = float(event.get("peak", 0.0) or 0.0) / max_peak
            if event.get("category") != "strong":
                base *= 0.6
            return base

        _fill_from_events(audio_energy, audio_events, cell_s, _energy, mode="max")
        _fill_from_events(
            audio_sharpness,
            audio_events,
            cell_s,
            lambda e: float(e.get("sharpness", 0.0) or 0.0),
            mode="max",
        )

    # --- лица ---
    face_events = face_summary.get("events", []) if isinstance(face_summary, dict) else []
    _fill_from_events(face, face_events, cell_s, lambda e: 1.0, mode="coverage")
    _fill_from_events(
        face_ratio,
        face_events,
        cell_s,
        lambda e: float(e.get("max_face_ratio", 0.0) or 0.0),
        mode="max",
    )

    # --- склейки ---
    for t in cut_times or []:
        idx = int(min(cells - 1, max(0, math.floor(float(t) / cell_s))))
        cuts[idx] += 1.0

    # --- движение ---
    for t, value in motion_series or []:
        idx = int(min(cells - 1, max(0, math.floor(float(t) / cell_s))))
        motion[idx] = max(motion[idx], float(value))

    motion_values = motion[motion > 0]
    motion_ref = float(np.percentile(motion_values, 90)) if motion_values.size else 0.15
    motion_ref = max(1e-3, motion_ref)

    return SignalTimeline(
        cell_s=cell_s,
        duration=duration,
        speech=speech,
        words=words,
        audio_energy=audio_energy,
        audio_sharpness=audio_sharpness,
        face=face,
        face_ratio=face_ratio,
        cuts=cuts,
        motion=motion,
        hooks=list(hooks or []),
        motion_ref=motion_ref,
    )


# ============================================================
# Сигналы окна
# ============================================================

def _longest_silence_run(speech_cells: np.ndarray, cell_s: float, threshold: float = 0.2) -> float:
    longest = 0
    current = 0
    for value in speech_cells:
        if value < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * cell_s


def hook_signal(timeline: SignalTimeline, start: float, window_s: float) -> Tuple[float, dict]:
    end = min(timeline.duration, start + window_s)
    lo, hi = timeline.slice_bounds(start, end)

    text_score = 0.0
    hook_text = ""
    hook_type = ""
    for hook in timeline.hooks:
        h_start = float(hook.get("start", 0.0) or 0.0)
        h_end = float(hook.get("end", 0.0) or 0.0)
        if _overlap(h_start, h_end, start, end) <= 0:
            continue
        score = float(hook.get("score", 0.0) or 0.0)
        if score > text_score:
            text_score = score
            hook_text = str(hook.get("text", ""))
            hook_type = str(hook.get("type", ""))

    audio_start = _percentile(timeline.audio_energy[lo:hi], 90)
    face_start = _mean(timeline.face[lo:hi])
    speech_start = _mean(timeline.speech[lo:hi])

    signal = 0.5 * text_score + 0.3 * audio_start + 0.2 * face_start
    if speech_start < 0.1 and audio_start < 0.2:
        # вход в клип «немой»: ни реплики, ни звукового акцента
        signal *= 0.4

    evidence = {
        "text_score": round(text_score, 3),
        "type": hook_type,
        "text": hook_text[:160],
        "audio": round(audio_start, 3),
        "face": round(face_start, 3),
        "speech": round(speech_start, 3),
    }
    return _clip01(signal), evidence


def window_signals(timeline: SignalTimeline, start: float, end: float, cfg: dict) -> Tuple[Dict[str, float], dict]:
    """Считает пять базовых сигналов + hook для окна [start, end)."""

    lo, hi = timeline.slice_bounds(start, end)
    span = max(1e-6, end - start)

    speech_cells = timeline.speech[lo:hi]
    speech_density = _mean(speech_cells)
    words_total = float(np.sum(timeline.words[lo:hi]))
    words_per_s = words_total / span

    speech_lo, speech_hi = cfg.get("speech_sweet_spot", [0.45, 0.88])
    transcript_signal = (
            0.6 * _sweet_spot(speech_density, float(speech_lo), float(speech_hi))
            + 0.4 * _clip01(words_per_s / float(cfg.get("words_per_s_ref", 3.0) or 3.0))
    )

    energy_cells = timeline.audio_energy[lo:hi]
    audio_signal = (
            0.55 * _percentile(energy_cells, 90)
            + 0.30 * _mean(energy_cells)
            + 0.15 * _max(timeline.audio_sharpness[lo:hi])
    )

    face_coverage = _mean(timeline.face[lo:hi])
    target_ratio = float(cfg.get("target_face_ratio", 0.10) or 0.10)
    face_signal = (
            0.65 * face_coverage
            + 0.35 * _clip01(_max(timeline.face_ratio[lo:hi]) / max(1e-6, target_ratio))
    )

    cuts_total = float(np.sum(timeline.cuts[lo:hi]))
    cuts_per_min = cuts_total / span * 60.0
    cuts_lo, cuts_hi = cfg.get("cuts_per_min_sweet_spot", [6.0, 40.0])
    motion_mean = _mean(timeline.motion[lo:hi])
    scene_signal = (
            0.6 * _sweet_spot(cuts_per_min, float(cuts_lo), float(cuts_hi), falloff=1.5)
            + 0.4 * _clip01(motion_mean / timeline.motion_ref)
    )

    silence_ratio = 1.0 - speech_density
    longest_silence = _longest_silence_run(speech_cells, timeline.cell_s)
    max_gap = float(cfg.get("max_silence_gap_s", 2.5) or 2.5)
    pacing_signal = (
            0.45 * (1.0 - _clip01(longest_silence / max(1e-6, max_gap)))
            + 0.35 * (1.0 - _clip01(silence_ratio))
            + 0.20 * _clip01(cuts_per_min / 30.0)
    )

    hook_signal_value, hook_evidence = hook_signal(
        timeline, start, float(cfg.get("hook_window_s", 2.0) or 2.0)
    )

    signals = {
        "transcript": _clip01(transcript_signal),
        "audio": _clip01(audio_signal),
        "face": _clip01(face_signal),
        "scene": _clip01(scene_signal),
        "pacing": _clip01(pacing_signal),
        "hook": hook_signal_value,
    }
    stats = {
        "speech_density": round(speech_density, 3),
        "words_per_s": round(words_per_s, 3),
        "silence_ratio": round(silence_ratio, 3),
        "longest_silence_s": round(longest_silence, 2),
        "face_coverage": round(face_coverage, 3),
        "cuts_per_min": round(cuts_per_min, 2),
        "motion_mean": round(motion_mean, 4),
        "hook": hook_evidence,
    }
    return signals, stats


def resolve_scoring_cfg(scoring_cfg: Optional[dict]) -> dict:
    """Мержит пользовательский конфиг с дефолтами и нормализует веса."""
    cfg = dict(DEFAULT_SCORING)
    weights = dict(DEFAULT_WEIGHTS)
    if isinstance(scoring_cfg, dict):
        for key, value in scoring_cfg.items():
            if key == "weights":
                continue
            cfg[key] = value
        user_weights = scoring_cfg.get("weights")
        if isinstance(user_weights, dict):
            for key, value in user_weights.items():
                if key in weights:
                    try:
                        weights[key] = float(value)
                    except (TypeError, ValueError):
                        continue
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        weights = dict(DEFAULT_WEIGHTS)
        total = sum(weights.values())
    cfg["weights"] = {key: max(0.0, value) / total for key, value in weights.items()}
    return cfg


def score_window(timeline: SignalTimeline, start: float, end: float, cfg: dict) -> dict:
    signals, stats = window_signals(timeline, start, end, cfg)
    weights = cfg.get("weights") or DEFAULT_WEIGHTS
    score = sum(weights.get(key, 0.0) * value for key, value in signals.items())

    reasons: List[str] = []
    min_speech = float(cfg.get("min_speech_density", 0.18) or 0.0)
    if stats["speech_density"] < min_speech:
        score *= float(cfg.get("empty_window_penalty", 0.55) or 1.0)
        reasons.append("мало речи")

    min_face = float(cfg.get("min_face_coverage", 0.0) or 0.0)
    if min_face > 0 and stats["face_coverage"] < min_face:
        score *= float(cfg.get("low_face_penalty", 0.85) or 1.0)
        reasons.append("лиц почти нет")

    if signals["hook"] < 0.2:
        reasons.append("слабый вход")
    if stats["longest_silence_s"] > float(cfg.get("max_silence_gap_s", 2.5) or 2.5):
        reasons.append("провисание в середине")

    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "duration": round(float(end - start), 3),
        "score": round(_clip01(score), 4),
        "signals": {key: round(value, 3) for key, value in signals.items()},
        "stats": stats,
        "warnings": reasons,
    }


def build_candidates(
        timeline: Optional[SignalTimeline],
        min_time: float,
        max_time: float,
        scoring_cfg: Optional[dict] = None,
) -> List[dict]:
    """Скользящим окном строит и ранжирует кандидатов на клип."""

    if timeline is None or timeline.duration <= 0:
        return []

    cfg = resolve_scoring_cfg(scoring_cfg)
    if not cfg.get("enabled", True):
        return []

    min_time = max(1.0, float(min_time))
    max_time = max(min_time, float(max_time))

    length_steps = max(1, int(cfg.get("length_steps", 3) or 1))
    if length_steps == 1 or max_time <= min_time:
        lengths = [min_time]
    else:
        lengths = [
            min_time + (max_time - min_time) * i / (length_steps - 1)
            for i in range(length_steps)
        ]

    step = max(0.25, float(cfg.get("step_s", 1.0) or 1.0))
    raw: List[dict] = []
    for length in lengths:
        if length > timeline.duration:
            continue
        start = 0.0
        while start + length <= timeline.duration + 1e-6:
            raw.append(score_window(timeline, start, start + length, cfg))
            start += step

    if not raw:
        # эпизод короче минимальной длительности — оцениваем его целиком
        raw.append(score_window(timeline, 0.0, timeline.duration, cfg))

    raw.sort(key=lambda c: (-c["score"], c["start"]))

    nms_overlap = float(cfg.get("nms_overlap", 0.45) or 0.0)
    max_candidates = int(cfg.get("max_candidates", 12) or 12)
    selected: List[dict] = []
    for cand in raw:
        keep = True
        for chosen in selected:
            inter = _overlap(cand["start"], cand["end"], chosen["start"], chosen["end"])
            shortest = min(cand["duration"], chosen["duration"]) or 1.0
            if inter / shortest > nms_overlap:
                keep = False
                break
        if keep:
            selected.append(cand)
        if len(selected) >= max_candidates:
            break

    for rank, cand in enumerate(selected, start=1):
        cand["rank"] = rank
    return selected


# ============================================================
# Представление кандидатов
# ============================================================

_TITLE_STOP = re.compile(r"[^\w\s\-]", re.UNICODE)


def candidate_title(candidate: dict, fallback: str = "") -> str:
    """Заголовок-заглушка для fallback-моментов (LLM обычно даёт свой)."""
    stats = candidate.get("stats", {}) if isinstance(candidate, dict) else {}
    hook = stats.get("hook", {}) if isinstance(stats, dict) else {}
    text = str(hook.get("text", "")).strip()
    if not text:
        return fallback
    text = _TITLE_STOP.sub("", text).strip()
    if not text:
        return fallback
    words = text.split()
    title = " ".join(words[:6]).upper()
    return title[:45].strip() or fallback


def candidates_for_prompt(candidates: Sequence[dict], limit: int = 8) -> List[dict]:
    """Компактное представление кандидатов для промпта LLM."""
    compact = []
    for cand in list(candidates)[:limit]:
        stats = cand.get("stats", {})
        hook = stats.get("hook", {})
        compact.append(
            {
                "rank": cand.get("rank"),
                "start": cand.get("start"),
                "end": cand.get("end"),
                "duration": cand.get("duration"),
                "score": cand.get("score"),
                "signals": cand.get("signals"),
                "speech_density": stats.get("speech_density"),
                "face_coverage": stats.get("face_coverage"),
                "cuts_per_min": stats.get("cuts_per_min"),
                "longest_silence_s": stats.get("longest_silence_s"),
                "hook_type": hook.get("type"),
                "hook_text": hook.get("text"),
                "warnings": cand.get("warnings"),
            }
        )
    return compact


def moments_from_candidates(
        candidates: Sequence[dict],
        *,
        count: int,
        fallback_title: str = "",
) -> Dict[str, dict]:
    """Собирает fallback-моменты, если LLM не дала валидного результата."""
    moments: Dict[str, dict] = {}
    for idx, cand in enumerate(list(candidates)[:max(0, count)], start=1):
        moments[f"moment_{idx}"] = {
            "title": candidate_title(cand, fallback_title),
            "duration": cand.get("duration"),
            "segments": {
                "segment_1": {"start": cand.get("start"), "end": cand.get("end")}
            },
            "source": "heuristic_candidate",
            "score": cand.get("score"),
            "signals": cand.get("signals"),
        }
    return moments
