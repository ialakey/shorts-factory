import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import openai
from openai.error import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from moviepy.editor import VideoFileClip

from analysis import moment_scorer, moment_validator
from analysis.audio_analyzer import analyze_audio_peaks
from rendering.face_detector import analyze_face_activity


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _dump_artifact(path: Path, data) -> None:
    """Сохраняет промежуточный артефакт анализа (кандидаты, сигналы)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"⚠️ Не удалось сохранить артефакт {path.name}: {exc}")


def _extract_transcript_segments(transcript_result):
    segments = []
    for seg in transcript_result.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = _normalize_text(seg.get("text", ""))
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        word_count = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": word_count,
            }
        )
    return segments


def _estimate_duration(segments, audio_summary, face_summary):
    duration = 0.0
    if segments:
        duration = max(duration, max(seg["end"] for seg in segments))
    audio_stats = audio_summary.get("stats", {}) if isinstance(audio_summary, dict) else {}
    duration = max(duration, float(audio_stats.get("duration_s", 0.0) or 0.0))
    face_events = face_summary.get("events", []) if isinstance(face_summary, dict) else []
    if face_events:
        duration = max(duration, max(float(e.get("end", 0.0) or 0.0) for e in face_events))
    return duration


def _downsample_gray(frame: np.ndarray, size: int = 32) -> np.ndarray:
    if frame.ndim == 3:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = frame.astype(np.float32, copy=False)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.float32)
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return gray[np.ix_(ys, xs)].astype(np.float32, copy=False) / 255.0


def _detect_cuts(video_path: str, duration: float, sample_fps: float = 2.0):
    if duration <= 0:
        return [], []

    diffs = []
    times = []
    try:
        with VideoFileClip(str(video_path)) as clip:
            fps = max(0.5, float(sample_fps))
            frame_count = int(max(1, duration * fps))
            prev = None
            for idx in range(frame_count + 1):
                t = min(duration, idx / fps)
                frame = clip.get_frame(t)
                small = _downsample_gray(frame)
                if prev is not None:
                    diff = float(np.mean(np.abs(small - prev)))
                    diffs.append(diff)
                    times.append(t)
                prev = small
    except Exception:
        return [], []

    if not diffs:
        return [], []

    threshold = max(0.12, float(np.percentile(diffs, 90)))
    cut_times = [times[i] for i, d in enumerate(diffs) if d >= threshold]
    motion_series = list(zip(times, diffs))
    return cut_times, motion_series


def _build_tempo_windows(segments, duration, cut_times, motion_series, window_s: float = 5.0):
    windows = []
    if duration <= 0:
        return {"window_s": window_s, "windows": []}

    cut_times = cut_times or []
    motion_series = motion_series or []

    total_windows = int(np.ceil(duration / window_s))
    for idx in range(total_windows):
        w_start = idx * window_s
        w_end = min(duration, w_start + window_s)
        if w_end <= w_start:
            continue

        speech_time = 0.0
        for seg in segments:
            overlap = max(0.0, min(w_end, seg["end"]) - max(w_start, seg["start"]))
            if overlap > 0:
                speech_time += overlap
        speech_density = min(1.0, speech_time / (w_end - w_start))
        silence_ratio = max(0.0, 1.0 - speech_density)

        cuts = sum(1 for t in cut_times if w_start <= t < w_end)
        motion_values = []
        if motion_series:
            for t, diff in motion_series:
                if w_start <= t < w_end:
                    motion_values.append(diff)
        motion_intensity = round(float(np.mean(motion_values)), 4) if motion_values else None

        item = {
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "speech_density": round(speech_density, 4),
            "silence_ratio": round(silence_ratio, 4),
            "cuts_per_5s": int(cuts),
        }
        if motion_intensity is not None:
            item["motion_intensity"] = motion_intensity
        windows.append(item)

    return {"window_s": window_s, "duration_s": round(duration, 3), "windows": windows}


def _extract_hooks(segments):
    hooks = []
    q_words = re.compile(r"\b(что|почему|как|зачем|кто|когда|где|какой|что-то|why|what|how|who|where|when)\b", re.I)
    confession_words = re.compile(
        r"\b(признаюсь|признаю|если честно|честно говоря|мне стыдно|я боюсь|я никогда|я всегда)\b",
        re.I,
    )
    promise_words = re.compile(
        r"\b(обещаю|мы обещаем|я покажу|покажу дальше|сейчас покажу|скоро будет|в следующем)\b",
        re.I,
    )
    interruption_tokens = re.compile(r"(\.\.\.|—|–|\u2026)")

    for seg in segments:
        text = seg["text"]
        if not text:
            continue

        kind = None
        score = 0.0
        cue = ""

        if "?" in text or q_words.search(text):
            kind = "question"
            score = 0.65
            cue = "question"
        if confession_words.search(text):
            kind = "confession"
            score = max(score, 0.7)
            cue = "confession"
        if promise_words.search(text):
            kind = "promise"
            score = max(score, 0.75)
            cue = "promise"
        if interruption_tokens.search(text):
            kind = "interruption"
            score = max(score, 0.6)
            cue = "interruption"

        if kind:
            if text.endswith("?") or text.endswith("!"):
                score += 0.05
            hooks.append(
                {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "type": kind,
                    "score": round(min(score, 0.95), 3),
                    "text": text,
                    "evidence": cue,
                }
            )

    hooks.sort(key=lambda h: (-h["score"], h["start"]))
    return hooks[:50]


def _extract_emotion_peaks(segments, audio_summary):
    events = audio_summary.get("events", []) if isinstance(audio_summary, dict) else []
    if not events:
        return {"peaks": []}

    max_peak = max(float(e.get("peak", 0.0) or 0.0) for e in events) or 1.0
    emotional_words = re.compile(r"\b(вау|ого|ужас|страшно|люблю|ненавижу|боже|нет|да)\b", re.I)

    peaks = []
    for e in events:
        start = float(e.get("start", 0.0) or 0.0)
        end = float(e.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        intensity = float(e.get("peak", 0.0) or 0.0) / max_peak
        transcript_match = None
        marker = 0.0
        for seg in segments:
            if seg["end"] < start or seg["start"] > end:
                continue
            if "!" in seg["text"] or emotional_words.search(seg["text"]):
                transcript_match = seg["text"]
                marker = 1.0
                break
        score = 0.6 * intensity + 0.4 * marker
        if score < 0.55:
            continue
        peaks.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(min(score, 0.98), 3),
                "evidence": {
                    "audio": {
                        "peak": round(float(e.get("peak", 0.0) or 0.0), 6),
                        "hint": e.get("hint", ""),
                        "category": e.get("category", ""),
                    },
                    "transcript": transcript_match or "",
                },
            }
        )

    peaks.sort(key=lambda p: (-p["score"], p["start"]))
    return {"peaks": peaks[:12]}


def _build_visual_summary(face_summary, cut_times, motion_series, duration):
    face_events = face_summary.get("events", []) if isinstance(face_summary, dict) else []
    face_time = 0.0
    weighted_ratio = 0.0
    for e in face_events:
        start = float(e.get("start", 0.0) or 0.0)
        end = float(e.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        dur = end - start
        face_time += dur
        weighted_ratio += dur * float(e.get("max_face_ratio", 0.0) or 0.0)
    coverage = face_time / duration if duration > 0 else 0.0
    avg_ratio = weighted_ratio / face_time if face_time > 0 else 0.0

    motion_values = [d for _, d in motion_series] if motion_series else []
    motion_summary = {
        "mean_intensity": round(float(np.mean(motion_values)), 4) if motion_values else None,
        "p90_intensity": round(float(np.percentile(motion_values, 90)), 4) if motion_values else None,
    }

    return {
        "faces": {
            "event_count": len(face_events),
            "face_time_s": round(face_time, 3),
            "coverage_ratio": round(coverage, 4),
            "avg_max_face_ratio": round(avg_ratio, 4),
        },
        "cuts": {
            "count": len(cut_times or []),
            "rate_per_min": round((len(cut_times or []) / duration) * 60.0, 3) if duration > 0 else 0.0,
        },
        "motion": motion_summary,
    }


def analyze_moment(transcript_result, video_path, cfg, api_key, output_dir=None):
    openai.api_key = api_key

    segments = _extract_transcript_segments(transcript_result)

    transcript_text = ""
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        transcript_text += f"[{start:.2f}-{end:.2f}] {text}\n"

    gpt_cfg = cfg.get("gpt", {})
    prompt_template = gpt_cfg.get("prompt", "")
    model = gpt_cfg.get("model", "gpt-5.1-2025-11-13")

    min_time = float(gpt_cfg.get("min_time", 45))
    max_time = float(gpt_cfg.get("max_time", 60))
    min_count = int(gpt_cfg.get("min_count", 2) or 1)
    max_count = int(gpt_cfg.get("max_count", 3) or 1)

    audio_summary = analyze_audio_peaks(video_path, cfg)
    face_summary = analyze_face_activity(video_path, cfg)

    duration = _estimate_duration(segments, audio_summary, face_summary)
    cut_times, motion_series = _detect_cuts(video_path, duration)

    tempo = _build_tempo_windows(segments, duration, cut_times, motion_series)
    hook_events = _extract_hooks(segments)
    hooks = {"unresolved": hook_events}
    emotion = _extract_emotion_peaks(segments, audio_summary)
    visual = _build_visual_summary(face_summary, cut_times, motion_series, duration)

    # --- детерминированный слой отбора кандидатов -------------------------
    scoring_cfg = moment_scorer.resolve_scoring_cfg(cfg.get("moment_scoring"))
    timeline = moment_scorer.build_timeline(
        segments,
        audio_summary,
        face_summary,
        cut_times,
        motion_series,
        duration,
        hook_events,
        cell_s=float(scoring_cfg.get("cell_s", 0.5) or 0.5),
    )
    candidates = moment_scorer.build_candidates(
        timeline, min_time, max_time, cfg.get("moment_scoring")
    )
    if candidates:
        print(
            f"🧮 Эвристический отбор: {len(candidates)} кандидатов, "
            f"лучший score={candidates[0]['score']:.3f} "
            f"({candidates[0]['start']:.1f}-{candidates[0]['end']:.1f}s)"
        )
    else:
        print("ℹ️ Эвристические кандидаты не построены — решение полностью за LLM.")

    audio_json = json.dumps(audio_summary, ensure_ascii=False, indent=2)
    face_json = json.dumps(face_summary, ensure_ascii=False, indent=2)
    tempo_json = json.dumps(tempo, ensure_ascii=False, indent=2)
    hooks_json = json.dumps(hooks, ensure_ascii=False, indent=2)
    emotion_json = json.dumps(emotion, ensure_ascii=False, indent=2)
    visual_json = json.dumps(visual, ensure_ascii=False, indent=2)
    candidates_json = json.dumps(
        moment_scorer.candidates_for_prompt(
            candidates, limit=int(scoring_cfg.get("prompt_limit", 8) or 8)
        ),
        ensure_ascii=False,
        indent=2,
    )

    payload = (
        "---\n"
        "ТРАНСКРИПТ ДЛЯ АНАЛИЗА:\n"
        f"{transcript_text}\n"
        "АУДИО ДЛЯ АНАЛИЗА:\n"
        f"{audio_json}\n"
        "ЛИЦА ДЛЯ АНАЛИЗА:\n"
        f"{face_json}\n"
        "ВИЗУАЛ ДЛЯ АНАЛИЗА:\n"
        f"{visual_json}\n"
        "ТЕМП ДЛЯ АНАЛИЗА:\n"
        f"{tempo_json}\n"
        "ЭМОЦИИ ДЛЯ АНАЛИЗА:\n"
        f"{emotion_json}\n"
        "HOOKS ДЛЯ АНАЛИЗА:\n"
        f"{hooks_json}\n"
        "КАНДИДАТЫ ДЛЯ АНАЛИЗА:\n"
        f"{candidates_json}\n"
        "---"
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        with open(output_path / f"{stem}_chatgpt_payload.txt", "w", encoding="utf-8") as f:
            f.write(payload)
        # промежуточные артефакты: их можно переиспользовать и разбирать отдельно
        _dump_artifact(output_path / f"{stem}_candidates.json", candidates)
        _dump_artifact(
            output_path / f"{stem}_signals.json",
            {
                "duration_s": round(duration, 3),
                "cuts": [round(float(t), 3) for t in cut_times],
                "tempo": tempo,
                "hooks": hook_events,
                "emotion": emotion,
                "visual": visual,
                "scoring_weights": scoring_cfg.get("weights"),
            },
        )

    prompt = prompt_template.format(
        transcript_text=transcript_text,
        audio=audio_json,
        face=face_json,
        visual=visual_json,
        tempo=tempo_json,
        emotion=emotion_json,
        hooks=hooks_json,
        candidates=candidates_json,
        tone=gpt_cfg.get("tone", ""),
        platforms=gpt_cfg.get("platforms", ""),
        audience_age=gpt_cfg.get("audience_age", ""),
        min_count=gpt_cfg.get("min_count", 2),
        max_count=gpt_cfg.get("max_count", 3),
        min_time=gpt_cfg.get("min_time", 45),
        max_time=gpt_cfg.get("max_time", 60),
    )

    fallback_title = Path(video_path).stem

    def _validate_llm_moments(data, *, fill_missing):
        """Чинит и проверяет ответ модели по сигналам эпизода."""
        validation_cfg = dict(cfg.get("moment_validation") or {})
        if not fill_missing:
            # на промежуточных попытках не подмешиваем эвристику: даём модели шанс
            validation_cfg["fill_from_candidates"] = False
        moments, _ = moment_validator.select_moments(
            data,
            timeline=timeline,
            candidates=candidates,
            transcript_segments=segments,
            cut_times=cut_times,
            source_duration=duration,
            min_time=min_time,
            max_time=max_time,
            min_count=min_count,
            max_count=max_count,
            scoring_cfg=scoring_cfg,
            validation_cfg=validation_cfg,
            fallback_title=fallback_title,
        )
        return moments

    def _heuristic_fallback():
        if not candidates:
            return {}
        moments = moment_scorer.moments_from_candidates(
            candidates, count=max(1, min_count), fallback_title=fallback_title
        )
        print(
            f"🛟 Fallback: собираем {len(moments)} момент(ов) по эвристическим "
            "кандидатам вместо ответа модели."
        )
        return moments

    max_retries = 3
    reprompt_suffix = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}{reprompt_suffix}"}],
                timeout=90,
            )
            raw_json_text = response["choices"][0]["message"]["content"].strip()

            fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_json_text, re.DOTALL)
            json_text = fenced_match.group(1).strip() if fenced_match else raw_json_text

            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                if attempt == max_retries:
                    fallback = _heuristic_fallback()
                    if fallback:
                        return fallback
                    raise ValueError(f"❌ Модель вернула невалидный JSON:\n{json_text}")
                print(
                    f"⚠️ Модель вернула невалидный JSON (попытка {attempt}/{max_retries}). "
                    "Повторяем запрос..."
                )
                reprompt_suffix = (
                    "\n\nВАЖНО: верни ТОЛЬКО валидный JSON без пояснений и без markdown."
                )
                continue

            is_last_attempt = attempt == max_retries
            filtered_data = _validate_llm_moments(data, fill_missing=is_last_attempt)

            if filtered_data:
                return filtered_data

            if is_last_attempt:
                fallback = _heuristic_fallback()
                if fallback:
                    return fallback
                raise ValueError(
                    "❌ Модель не смогла вернуть валидные моменты "
                    f"({min_time}-{max_time} секунд) за {max_retries} попытки, "
                    "а эвристических кандидатов нет."
                )

            print(
                "⚠️ Ни один момент не прошёл валидацию. "
                f"Повторяем запрос с уточнением... (попытка {attempt}/{max_retries})"
            )
            reprompt_suffix = (
                "\n\nВАЖНО: Верни ТОЛЬКО моменты, где сумма длительностей segments строго в "
                f"диапазоне {min_time}-{max_time} секунд, segments не пересекаются, каждый "
                "segment длиннее 1.5 секунды, а в первые 1-2 секунды есть речь или звуковой "
                "акцент. Опирайся на список КАНДИДАТЫ ДЛЯ АНАЛИЗА. "
                "Верни валидный JSON без пояснений."
            )
            continue

        except AuthenticationError:
            raise ValueError("❌ Ошибка авторизации: токен OpenAI недействителен или истёк. Проверь OPENAI_API_KEY.")

        except (APIConnectionError, Timeout, ConnectionError) as e:
            print(f"⚠️ Потеря соединения с OpenAI (попытка {attempt}/{max_retries}): {e}")
            time.sleep(5 * attempt)
            continue

        except RateLimitError as e:
            message = str(e)
            if "quota" in message.lower():
                raise ValueError(
                    "❌ Превышена квота OpenAI. Проверьте план/биллинг или уменьшите количество запросов."
                )

            print(
                f"⚠️ Лимит скорости OpenAI, повтор через {5 * attempt} секунд... (попытка {attempt}/{max_retries})"
            )
            time.sleep(5 * attempt)
            continue

        except (APIError, ServiceUnavailableError) as e:
            code = getattr(e, "http_status", None)
            if code == 502 or "502" in str(e):
                print(f"⚠️ Ошибка 502, повтор через 15 секунд... (попытка {attempt}/{max_retries})")
                time.sleep(15)
                continue
            else:
                raise ValueError(f"❌ Ошибка API OpenAI ({code}): {e}")

        except Exception as e:
            err_type = type(e).__name__
            if "RemoteDisconnected" in str(e) or "ProtocolError" in str(e):
                print(f"⚠️ Ошибка сети ({err_type}), попытка {attempt}/{max_retries}...")
                time.sleep(5 * attempt)
                continue
            traceback.print_exc()
            raise ValueError(f"❌ Неожиданная ошибка при обращении к OpenAI API: {e}")

    fallback = _heuristic_fallback()
    if fallback:
        return fallback

    raise RuntimeError("🚫 Не удалось подключиться к OpenAI API после нескольких попыток.")
