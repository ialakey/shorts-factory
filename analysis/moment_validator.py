"""Валидация и «починка» моментов, которые вернула LLM.

Модель почти всегда ошибается в мелочах: выходит за длительность, склеивает
пересекающиеся отрезки, режет фразу на полуслове или предлагает окно, где нет
ни речи, ни звука. Раньше такие моменты просто выбрасывались (а весь эпизод мог
упасть с ошибкой). Здесь они сначала чинятся детерминированно, затем
проверяются по сигналам, и только безнадёжные отбрасываются — с добиванием
списка эвристическими кандидатами из :mod:`analysis.moment_scorer`.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from analysis import moment_scorer

DEFAULT_VALIDATION: Dict[str, object] = {
    "enabled": True,
    "min_segment_s": 1.2,        # короче — это «моргание», а не монтажный кусок
    "snap_tolerance_s": 0.6,     # притягиваем границы к паузам/склейкам
    "snap_to_speech": True,
    "snap_to_cuts": True,
    "min_speech_density": 0.05,  # жёсткий минимум: иначе клип пустой
    "min_audio_energy": 0.15,    # ...если речи нет, спасти может только звук
    "fill_from_candidates": True,
    "hook_warn_threshold": 0.2,
}


# ============================================================
# Нормализация того, что вернула модель
# ============================================================

def _coerce_pair(value: dict) -> Optional[Tuple[float, float]]:
    try:
        start = float(value.get("start"))
        end = float(value.get("end"))
    except (TypeError, ValueError, AttributeError):
        return None
    if end <= start:
        return None
    return start, end


def normalize_segments(segments_value) -> List[Tuple[float, float]]:
    """Приводит segments (dict/list) к упорядоченному списку пар."""
    if segments_value is None:
        return []
    if isinstance(segments_value, dict):
        items = list(segments_value.items())
    elif isinstance(segments_value, list):
        items = list(enumerate(segments_value, start=1))
    else:
        return []

    indexed: List[Tuple[int, dict]] = []
    for idx, (key, value) in enumerate(items, start=1):
        order = idx
        if isinstance(key, str) and key.startswith("segment_"):
            suffix = key.split("_", 1)[-1]
            if suffix.isdigit():
                order = int(suffix)
        if isinstance(value, dict):
            indexed.append((order, value))

    indexed.sort(key=lambda item: item[0])
    pairs = []
    for _, value in indexed:
        pair = _coerce_pair(value)
        if pair:
            pairs.append(pair)
    return pairs


def normalize_llm_moments(data) -> List[dict]:
    """Разворачивает ответ модели в единый список моментов."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(f"moment_{idx}", value) for idx, value in enumerate(data, start=1)]
    else:
        return []

    moments = []
    for key, value in items:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if not isinstance(value, dict):
            continue

        segments = normalize_segments(value.get("segments"))
        if not segments:
            pair = _coerce_pair(value)
            if pair:
                segments = [pair]
        if not segments:
            continue

        moments.append(
            {
                "key": str(key),
                "title": str(value.get("title", "") or ""),
                "segments": segments,
                "raw": value,
            }
        )
    return moments


# ============================================================
# Геометрия отрезков
# ============================================================

def _subtract(interval: Tuple[float, float], occupied: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Вырезает из отрезка уже занятые куски, возвращает свободные части."""
    pieces = [interval]
    for occ_start, occ_end in occupied:
        next_pieces: List[Tuple[float, float]] = []
        for start, end in pieces:
            if occ_end <= start or occ_start >= end:
                next_pieces.append((start, end))
                continue
            if start < occ_start:
                next_pieces.append((start, min(end, occ_start)))
            if end > occ_end:
                next_pieces.append((max(start, occ_end), end))
        pieces = [p for p in next_pieces if p[1] > p[0]]
    return pieces


def _free_gap(point: float, occupied: Sequence[Tuple[float, float]], duration: float) -> Tuple[float, float]:
    """Границы свободного промежутка вокруг точки."""
    lower = 0.0
    upper = duration
    for occ_start, occ_end in occupied:
        if occ_end <= point:
            lower = max(lower, occ_end)
        elif occ_start >= point:
            upper = min(upper, occ_start)
    return lower, upper


def _snap(value: float, snap_points: Sequence[float], tolerance: float, low: float, high: float) -> float:
    if tolerance <= 0 or not snap_points:
        return value
    best = value
    best_dist = tolerance
    for point in snap_points:
        if point < low or point > high:
            continue
        dist = abs(point - value)
        if dist < best_dist:
            best_dist = dist
            best = point
    return best


def build_snap_points(
        transcript_segments: Sequence[dict],
        cut_times: Sequence[float],
        cfg: dict,
) -> List[float]:
    """Границы фраз и склеек — «естественные» точки реза."""
    points: List[float] = []
    if cfg.get("snap_to_speech", True):
        for seg in transcript_segments or []:
            try:
                points.append(float(seg.get("start", 0.0) or 0.0))
                points.append(float(seg.get("end", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
    if cfg.get("snap_to_cuts", True):
        for t in cut_times or []:
            try:
                points.append(float(t))
            except (TypeError, ValueError):
                continue
    return sorted({round(p, 3) for p in points if p >= 0})


# ============================================================
# Починка момента
# ============================================================

def repair_segments(
        segments: Sequence[Tuple[float, float]],
        *,
        source_duration: float,
        min_time: float,
        max_time: float,
        cfg: dict,
        snap_points: Sequence[float] = (),
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Чинит монтажную сборку: клампы, пересечения, притяжка к паузам, длительность."""

    notes: List[str] = []
    min_segment_s = float(cfg.get("min_segment_s", 1.2) or 0.0)
    tolerance = float(cfg.get("snap_tolerance_s", 0.6) or 0.0)
    source_duration = float(source_duration or 0.0)

    occupied: List[Tuple[float, float]] = []
    result: List[Tuple[float, float]] = []

    for start, end in segments:
        start = max(0.0, float(start))
        end = float(end)
        if source_duration > 0:
            end = min(end, source_duration)
        if end - start <= 0:
            notes.append("segment вне длительности исходника")
            continue

        pieces = _subtract((start, end), occupied)
        if not pieces:
            notes.append("segment полностью пересекался с предыдущим")
            continue
        piece = max(pieces, key=lambda p: p[1] - p[0])
        if piece[1] - piece[0] < min_segment_s and result:
            notes.append("segment короче минимальной длины")
            continue

        low, high = _free_gap((piece[0] + piece[1]) / 2.0, occupied, source_duration or piece[1])
        snapped_start = _snap(piece[0], snap_points, tolerance, low, piece[1] - 0.2)
        snapped_end = _snap(piece[1], snap_points, tolerance, piece[0] + 0.2, high)
        if snapped_end - snapped_start >= max(0.4, min_segment_s * 0.5):
            piece = (snapped_start, snapped_end)

        result.append(piece)
        occupied.append(piece)
        occupied.sort()

    if not result:
        return [], notes

    total = sum(end - start for start, end in result)

    # --- слишком длинно: подрезаем с конца сборки ---
    if total > max_time:
        excess = total - max_time
        for idx in range(len(result) - 1, -1, -1):
            if excess <= 1e-6:
                break
            start, end = result[idx]
            length = end - start
            spare = length - min_segment_s
            if spare <= 0 and len(result) > 1:
                result.pop(idx)
                excess -= length
                notes.append("лишний segment убран под лимит длительности")
                continue
            cut = min(excess, max(0.0, spare))
            if cut > 0:
                result[idx] = (start, end - cut)
                excess -= cut
        notes.append("сборка подрезана под max_time")
        total = sum(end - start for start, end in result)

    # --- слишком коротко: расширяем в пределах свободных промежутков ---
    if total < min_time:
        deficit = min_time - total
        occupied = sorted(result)
        for idx in range(len(result) - 1, -1, -1):
            if deficit <= 1e-6:
                break
            start, end = result[idx]
            others = [seg for j, seg in enumerate(result) if j != idx]
            low, high = _free_gap((start + end) / 2.0, others, source_duration or end)
            grow_forward = min(deficit, max(0.0, high - end))
            end += grow_forward
            deficit -= grow_forward
            if deficit > 1e-6:
                grow_back = min(deficit, max(0.0, start - low))
                start -= grow_back
                deficit -= grow_back
            result[idx] = (start, end)
        total = sum(end - start for start, end in result)
        if total + 1e-3 < min_time:
            notes.append("не удалось дотянуть до min_time")
            return [], notes
        notes.append("сборка расширена до min_time")

    # --- склеиваем стыкующиеся куски: лишний рез не нужен ---
    merged: List[Tuple[float, float]] = []
    for start, end in result:
        if merged and abs(start - merged[-1][1]) < 0.05:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) != len(result):
        notes.append("стыкующиеся segments склеены")
    result = merged

    result = [(round(start, 3), round(end, 3)) for start, end in result if end > start]
    return result, notes


# ============================================================
# Оценка качества момента
# ============================================================

def evaluate_moment(
        segments: Sequence[Tuple[float, float]],
        timeline: Optional[moment_scorer.SignalTimeline],
        scoring_cfg: dict,
        validation_cfg: dict,
) -> dict:
    """Считает сигналы по собранному моменту и решает, жизнеспособен ли он."""

    if timeline is None or not segments:
        return {"score": 0.0, "signals": {}, "stats": {}, "fatal": [], "warnings": []}

    total = sum(end - start for start, end in segments) or 1.0
    weights = scoring_cfg.get("weights") or moment_scorer.DEFAULT_WEIGHTS

    aggregated: Dict[str, float] = {key: 0.0 for key in weights}
    speech_density = 0.0
    face_coverage = 0.0
    audio_energy = 0.0

    for start, end in segments:
        signals, stats = moment_scorer.window_signals(timeline, start, end, scoring_cfg)
        share = (end - start) / total
        for key in aggregated:
            aggregated[key] += signals.get(key, 0.0) * share
        speech_density += stats["speech_density"] * share
        face_coverage += stats["face_coverage"] * share
        audio_energy = max(audio_energy, signals.get("audio", 0.0))

    # hook считается только по входу в клип — по первому сегменту
    hook_value, hook_evidence = moment_scorer.hook_signal(
        timeline,
        segments[0][0],
        float(scoring_cfg.get("hook_window_s", 2.0) or 2.0),
    )
    aggregated["hook"] = hook_value

    score = sum(weights.get(key, 0.0) * value for key, value in aggregated.items())

    fatal: List[str] = []
    warnings: List[str] = []

    if (
            speech_density < float(validation_cfg.get("min_speech_density", 0.05) or 0.0)
            and audio_energy < float(validation_cfg.get("min_audio_energy", 0.15) or 0.0)
    ):
        fatal.append("пустой фрагмент: ни речи, ни звуковых акцентов")

    if hook_value < float(validation_cfg.get("hook_warn_threshold", 0.2) or 0.0):
        warnings.append("слабый hook на входе")

    return {
        "score": round(float(score), 4),
        "signals": {key: round(value, 3) for key, value in aggregated.items()},
        "stats": {
            "speech_density": round(speech_density, 3),
            "face_coverage": round(face_coverage, 3),
            "hook": hook_evidence,
        },
        "fatal": fatal,
        "warnings": warnings,
    }


# ============================================================
# Итоговый отбор
# ============================================================

def _segments_to_dict(segments: Sequence[Tuple[float, float]]) -> Dict[str, dict]:
    return {
        f"segment_{idx}": {"start": round(float(start), 3), "end": round(float(end), 3)}
        for idx, (start, end) in enumerate(segments, start=1)
    }


def _overlap_ratio(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    total_a = sum(end - start for start, end in a) or 1.0
    inter = 0.0
    for a_start, a_end in a:
        for b_start, b_end in b:
            inter += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return inter / total_a


def resolve_validation_cfg(cfg: Optional[dict]) -> dict:
    resolved = dict(DEFAULT_VALIDATION)
    if isinstance(cfg, dict):
        resolved.update(cfg)
    return resolved


def select_moments(
        llm_data,
        *,
        timeline: Optional[moment_scorer.SignalTimeline],
        candidates: Sequence[dict],
        transcript_segments: Sequence[dict],
        cut_times: Sequence[float],
        source_duration: float,
        min_time: float,
        max_time: float,
        min_count: int,
        max_count: int,
        scoring_cfg: dict,
        validation_cfg: Optional[dict] = None,
        fallback_title: str = "",
        verbose: bool = True,
) -> Tuple[Dict[str, dict], List[str]]:
    """Чинит и валидирует моменты, добивая недостачу эвристическими кандидатами."""

    vcfg = resolve_validation_cfg(validation_cfg)
    log: List[str] = []

    def _log(message: str) -> None:
        log.append(message)
        if verbose:
            print(message)

    snap_points = build_snap_points(transcript_segments, cut_times, vcfg) if vcfg.get("enabled", True) else []

    prepared: List[dict] = []
    for moment in normalize_llm_moments(llm_data):
        if vcfg.get("enabled", True):
            segments, notes = repair_segments(
                moment["segments"],
                source_duration=source_duration,
                min_time=min_time,
                max_time=max_time,
                cfg=vcfg,
                snap_points=snap_points,
            )
        else:
            segments, notes = list(moment["segments"]), []

        if not segments:
            _log(f"⚠️ Момент {moment['key']} отброшен: {', '.join(notes) or 'нет валидных сегментов'}")
            continue

        evaluation = evaluate_moment(segments, timeline, scoring_cfg, vcfg)
        if evaluation["fatal"]:
            _log(f"⚠️ Момент {moment['key']} отброшен: {', '.join(evaluation['fatal'])}")
            continue

        duration = round(sum(end - start for start, end in segments), 3)
        prepared.append(
            {
                "key": moment["key"],
                "title": moment["title"],
                "segments": segments,
                "duration": duration,
                "score": evaluation["score"],
                "signals": evaluation["signals"],
                "stats": evaluation["stats"],
                "notes": notes + evaluation["warnings"],
                "source": "llm",
            }
        )
        if notes:
            _log(f"🔧 Момент {moment['key']} починен: {', '.join(notes)} → {duration:.1f}s")

    prepared.sort(key=lambda m: -m["score"])
    selected = prepared[: max(1, int(max_count))]

    # --- добиваем эвристикой, если моделью выбрано слишком мало ---
    if vcfg.get("fill_from_candidates", True) and len(selected) < int(min_count):
        for cand in candidates or []:
            if len(selected) >= int(min_count):
                break
            cand_segments = [(float(cand["start"]), float(cand["end"]))]
            if any(_overlap_ratio(cand_segments, item["segments"]) > 0.5 for item in selected):
                continue
            segments, _ = repair_segments(
                cand_segments,
                source_duration=source_duration,
                min_time=min_time,
                max_time=max_time,
                cfg=vcfg,
                snap_points=snap_points,
            )
            if not segments:
                continue
            evaluation = evaluate_moment(segments, timeline, scoring_cfg, vcfg)
            if evaluation["fatal"]:
                continue
            selected.append(
                {
                    "key": f"candidate_{cand.get('rank', len(selected) + 1)}",
                    "title": moment_scorer.candidate_title(cand, fallback_title),
                    "segments": segments,
                    "duration": round(sum(end - start for start, end in segments), 3),
                    "score": evaluation["score"],
                    "signals": evaluation["signals"],
                    "stats": evaluation["stats"],
                    "notes": ["добавлен из эвристических кандидатов"],
                    "source": "heuristic_candidate",
                }
            )
            _log(
                "➕ Добавлен эвристический кандидат "
                f"{segments[0][0]:.1f}-{segments[-1][1]:.1f}s (score={evaluation['score']:.3f})"
            )

    result: Dict[str, dict] = {}
    for idx, item in enumerate(selected, start=1):
        result[f"moment_{idx}"] = {
            "title": item["title"],
            "duration": item["duration"],
            "segments": _segments_to_dict(item["segments"]),
            "score": item["score"],
            "signals": item["signals"],
            "stats": item["stats"],
            "source": item["source"],
            "notes": item["notes"],
        }
    return result, log
