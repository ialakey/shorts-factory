from __future__ import annotations

from typing import Dict, List, Optional

import contextlib
import os
import subprocess
import tempfile
import wave

import numpy as np
from moviepy.editor import VideoFileClip


def _label_peak(peak_value: float, max_value: float) -> str:
    if max_value <= 0:
        return "пик"
    strength = peak_value / max_value
    if strength >= 0.9:
        return "взрыв"
    if strength >= 0.75:
        return "крик"
    if strength >= 0.6:
        return "удар"
    return "пик"


def _safe_soundarray(audio_clip, sample_rate: int) -> np.ndarray:
    """
    Robust soundarray extraction from MoviePy AudioClip.
    Always returns ndarray of shape (samples, channels) with dtype float32.
    Fixes MoviePy/NumPy stack issues like:
    'arrays to stack must be passed as a "sequence" type such as list or tuple.'
    """
    # Fast path
    try:
        arr = audio_clip.to_soundarray(fps=sample_rate)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.size > 0:
            return arr
    except Exception:
        pass

    # Fallback: manual chunking
    chunks: List[np.ndarray] = []
    try:
        for chunk in audio_clip.iter_chunks(fps=sample_rate):
            if chunk is None:
                continue
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.size == 0:
                continue
            if chunk.ndim == 1:
                chunk = chunk[:, None]
            chunks.append(chunk)
    except Exception:
        return np.empty((0, 1), dtype=np.float32)

    if not chunks:
        return np.empty((0, 1), dtype=np.float32)

    # Normalize channel count across chunks
    min_channels = min((c.shape[1] for c in chunks if c.ndim == 2), default=0)
    if min_channels <= 0:
        return np.empty((0, 1), dtype=np.float32)

    normalized = [c[:, :min_channels] for c in chunks]
    try:
        return np.concatenate(normalized, axis=0).astype(np.float32, copy=False)
    except Exception:
        return np.empty((0, min_channels), dtype=np.float32)


def _frame_signal(mono: np.ndarray, window_size: int) -> np.ndarray:
    usable = (len(mono) // window_size) * window_size
    if usable <= 0:
        return np.empty((0, window_size), dtype=np.float32)
    mono = mono[:usable]
    return mono.reshape(-1, window_size).astype(np.float32, copy=False)


def _rms(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32, copy=False)


def _zcr(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)
    signs = np.sign(frames)
    signs[signs == 0] = 1.0
    zc = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    return zc.astype(np.float32, copy=False)


def _spectral_centroid(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)

    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32, copy=False)
    if mag.size == 0:
        return np.empty((0,), dtype=np.float32)

    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate).astype(np.float32)
    denom = np.sum(mag, axis=1)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    centroid = np.sum(mag * freqs[None, :], axis=1) / denom
    return centroid.astype(np.float32, copy=False)


def _merge_events_from_mask(
        mask: np.ndarray,
        strength: np.ndarray,
        frame_duration: float,
        min_event_ms: float,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    start: Optional[int] = None
    peak = 0.0
    sum_strength = 0.0
    count = 0

    for i, on in enumerate(mask):
        if on:
            if start is None:
                start = i
                peak = float(strength[i])
                sum_strength = float(strength[i])
                count = 1
            else:
                v = float(strength[i])
                peak = max(peak, v)
                sum_strength += v
                count += 1
        elif start is not None:
            end = i
            dur_s = (end - start) * frame_duration
            if dur_s * 1000.0 >= min_event_ms:
                events.append(
                    {
                        "start": round(start * frame_duration, 3),
                        "end": round(end * frame_duration, 3),
                        "peak": round(peak, 6),
                        "mean": round((sum_strength / max(1, count)), 6),
                        "frames": int(end - start),
                    }
                )
            start = None
            peak = 0.0
            sum_strength = 0.0
            count = 0

    if start is not None:
        end = len(strength)
        dur_s = (end - start) * frame_duration
        if dur_s * 1000.0 >= min_event_ms:
            events.append(
                {
                    "start": round(start * frame_duration, 3),
                    "end": round(end * frame_duration, 3),
                    "peak": round(peak, 6),
                    "mean": round((sum_strength / max(1, count)), 6),
                    "frames": int(end - start),
                }
            )

    return events


def _compute_event_features(
        events: List[Dict[str, object]],
        rms: np.ndarray,
        zcr: np.ndarray,
        centroid: np.ndarray,
        frame_duration: float,
) -> None:
    if not events:
        return

    n = len(rms)
    for e in events:
        s = int(round(float(e["start"]) / frame_duration))
        t = int(round(float(e["end"]) / frame_duration))
        s = max(0, min(n, s))
        t = max(0, min(n, t))
        if t <= s:
            e["rms_mean"] = 0.0
            e["rms_p95"] = 0.0
            e["zcr_mean"] = 0.0
            e["centroid_mean"] = 0.0
            e["sharpness"] = 0.0
            continue

        seg_rms = rms[s:t]
        seg_zcr = zcr[s:t] if zcr.size == rms.size else np.empty((0,), dtype=np.float32)
        seg_cen = centroid[s:t] if centroid.size == rms.size else np.empty((0,), dtype=np.float32)

        e["rms_mean"] = round(float(np.mean(seg_rms)), 6)
        e["rms_p95"] = round(float(np.percentile(seg_rms, 95)), 6)

        z_mean = float(np.mean(seg_zcr)) if seg_zcr.size else 0.0
        e["zcr_mean"] = round(z_mean, 6)

        c_mean = float(np.mean(seg_cen)) if seg_cen.size else 0.0
        e["centroid_mean"] = round(c_mean, 3)

        cen_norm = min(1.0, max(0.0, c_mean / 8000.0))
        sharp = 0.65 * z_mean + 0.35 * cen_norm
        e["sharpness"] = round(float(sharp), 6)


def _ffmpeg_extract_wav(video_path: str, out_wav: str, sample_rate: int) -> bool:
    """
    Extract mono PCM wav via ffmpeg. Returns True on success.
    Requires ffmpeg to be available (MoviePy usually has it).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 44
    except Exception:
        return False


def _read_wav_pcm16(wav_path: str) -> np.ndarray:
    """
    Read PCM 16-bit WAV into float32 ndarray shape (samples, 1) in [-1, 1].
    """
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            if n_frames <= 0 or n_channels <= 0:
                return np.empty((0, 1), dtype=np.float32)

            if sampwidth != 2:
                # We requested pcm_s16le, but handle defensively.
                raw = wf.readframes(n_frames)
                if not raw:
                    return np.empty((0, 1), dtype=np.float32)
                # best-effort: treat as int16
                data = np.frombuffer(raw, dtype=np.int16)
            else:
                raw = wf.readframes(n_frames)
                if not raw:
                    return np.empty((0, 1), dtype=np.float32)
                data = np.frombuffer(raw, dtype=np.int16)

            if data.size == 0:
                return np.empty((0, 1), dtype=np.float32)

            # If multi-channel slipped in, reshape & average
            if n_channels > 1:
                usable = (data.size // n_channels) * n_channels
                data = data[:usable].reshape(-1, n_channels).mean(axis=1).astype(np.int16, copy=False)

            audio = (data.astype(np.float32) / 32768.0).reshape(-1, 1)
            return audio
    except Exception:
        return np.empty((0, 1), dtype=np.float32)


def analyze_audio_peaks(video_path, cfg) -> Dict[str, object]:
    audio_cfg = cfg.get("audio_analysis", {})
    sample_rate = int(audio_cfg.get("sample_rate", 22050))
    window_ms = float(audio_cfg.get("window_ms", 80))
    min_event_ms = float(audio_cfg.get("min_event_ms", 60))
    peak_percentile = float(audio_cfg.get("peak_percentile", 92))
    max_events = int(audio_cfg.get("max_events", 120))
    min_threshold = float(audio_cfg.get("min_threshold", 0.02))

    allow_soft_events = bool(audio_cfg.get("allow_soft_events", True))
    soft_percentile = float(audio_cfg.get("soft_percentile", 80))
    min_gap_ms = float(audio_cfg.get("min_gap_ms", 80))

    debug: Dict[str, object] = {}

    try:
        with VideoFileClip(str(video_path)) as clip:
            debug["video_duration"] = float(getattr(clip, "duration", 0.0) or 0.0)

            if clip.audio is None:
                return {"events": [], "summary": "no_audio", "debug": debug}

            debug["audio_duration"] = float(getattr(clip.audio, "duration", 0.0) or 0.0)
            debug["audio_fps"] = getattr(clip.audio, "fps", None)

            audio = _safe_soundarray(clip.audio, sample_rate)
    except Exception as exc:
        return {"events": [], "summary": f"error: {exc}", "debug": debug}

    # If MoviePy gave nothing, try ffmpeg WAV extraction (common on “weird” mp4/webm)
    if audio.size == 0:
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            ok = _ffmpeg_extract_wav(str(video_path), wav_path, sample_rate)
            debug["ffmpeg_wav"] = bool(ok)
            if ok:
                audio = _read_wav_pcm16(wav_path)

    if audio.size == 0:
        # Still empty => likely truly no decodable audio
        return {
            "events": [],
            "summary": "empty_audio",
            "debug": debug,
            "stats": {
                "duration_s": round(float(debug.get("audio_duration", 0.0) or 0.0), 3),
                "sample_rate": sample_rate,
                "window_ms": window_ms,
            },
        }

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    mono = np.mean(audio, axis=1).astype(np.float32, copy=False)

    window_size = max(1, int(sample_rate * (window_ms / 1000.0)))
    frames = _frame_signal(mono, window_size)
    if frames.size == 0:
        return {"events": [], "summary": "audio_too_short", "debug": debug}

    rms = _rms(frames)
    if rms.size == 0:
        return {"events": [], "summary": "rms_empty", "debug": debug}

    zcr = _zcr(frames)
    centroid = _spectral_centroid(frames, sample_rate)
    frame_duration = window_ms / 1000.0

    threshold = max(float(np.percentile(rms, peak_percentile)), min_threshold)
    strong_mask = rms >= threshold

    soft_threshold = max(float(np.percentile(rms, soft_percentile)), min_threshold) if allow_soft_events else threshold
    soft_mask = (rms >= soft_threshold) & ~strong_mask if allow_soft_events else np.zeros_like(strong_mask, dtype=bool)

    strong_events = _merge_events_from_mask(strong_mask, rms, frame_duration, min_event_ms)
    soft_events = _merge_events_from_mask(soft_mask, rms, frame_duration, min_event_ms) if allow_soft_events else []

    for e in strong_events:
        e["category"] = "strong"
    for e in soft_events:
        e["category"] = "soft"

    events: List[Dict[str, object]] = strong_events + soft_events

    # Always return global stats (even if no peaks) so LLM has context
    stats = {
        "rms_mean": round(float(np.mean(rms)), 6),
        "rms_std": round(float(np.std(rms)), 6),
        "rms_p90": round(float(np.percentile(rms, 90)), 6),
        "rms_p95": round(float(np.percentile(rms, 95)), 6),
        "rms_max": round(float(np.max(rms)), 6),
        "zcr_mean": round(float(np.mean(zcr)) if zcr.size else 0.0, 6),
        "centroid_mean": round(float(np.mean(centroid)) if centroid.size else 0.0, 3),
        "duration_s": round(len(mono) / float(sample_rate), 3),
        "frames": int(rms.size),
        "strong_threshold": round(threshold, 6),
        "soft_threshold": round(float(soft_threshold), 6) if allow_soft_events else round(threshold, 6),
    }

    if not events:
        return {
            "events": [],
            "threshold": round(threshold, 6),
            "window_ms": window_ms,
            "sample_rate": sample_rate,
            "summary": "no_peaks",
            "stats": stats,
            "debug": debug,
        }

    # Merge close events
    events.sort(key=lambda e: e["start"])
    if min_gap_ms > 0:
        gap_s = min_gap_ms / 1000.0
        merged: List[Dict[str, object]] = []
        cur = events[0].copy()
        for nxt in events[1:]:
            if float(nxt["start"]) - float(cur["end"]) <= gap_s:
                cur["end"] = max(cur["end"], nxt["end"])
                cur["peak"] = round(max(float(cur["peak"]), float(nxt["peak"])), 6)
                cur["mean"] = round((float(cur["mean"]) + float(nxt["mean"])) / 2.0, 6)
                cur["frames"] = int(cur.get("frames", 0)) + int(nxt.get("frames", 0))
                if cur.get("category") != "strong" and nxt.get("category") == "strong":
                    cur["category"] = "strong"
            else:
                merged.append(cur)
                cur = nxt.copy()
        merged.append(cur)
        events = merged

    _compute_event_features(events, rms, zcr, centroid, frame_duration)

    max_peak = max(float(e.get("peak", 0.0)) for e in events) or 0.0
    for e in events:
        e["type"] = _label_peak(float(e.get("peak", 0.0)), max_peak)
        e["duration"] = round(float(e["end"]) - float(e["start"]), 3)

        sharp = float(e.get("sharpness", 0.0))
        dur = float(e.get("duration", 0.0))
        if e["category"] == "strong":
            if sharp >= 0.35 and dur <= 0.35:
                e["hint"] = "резкий_пик"
            elif sharp >= 0.25 and dur >= 0.35:
                e["hint"] = "крик_или_шум"
            else:
                e["hint"] = "громкий_момент"
        else:
            if sharp >= 0.30:
                e["hint"] = "интересный_шум"
            else:
                e["hint"] = "фон_или_музыка"

    events.sort(key=lambda e: (0 if e.get("category") == "strong" else 1, -float(e.get("peak", 0.0))))
    events = events[:max_events]
    events.sort(key=lambda e: e["start"])

    return {
        "events": events,
        "threshold": round(threshold, 6),
        "window_ms": window_ms,
        "sample_rate": sample_rate,
        "summary": "ok",
        "stats": stats,
        "debug": debug,
    }