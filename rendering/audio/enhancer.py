"""Audio enhancement utilities for dialogue tracks."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from moviepy.audio.AudioClip import AudioClip


def _compress_frame(
    frame: np.ndarray,
    *,
    threshold: float,
    ratio: float,
    soft_knee: float,
    makeup_gain: float,
    noise_floor: float,
    output_ceiling: float,
) -> np.ndarray:
    """Apply a soft-knee compressor to a single audio frame."""

    if frame.size == 0:
        return frame

    original_dtype = frame.dtype if isinstance(frame, np.ndarray) else np.float32
    samples = np.asarray(frame, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)

    magnitudes = np.abs(samples)
    compressed = samples.copy()

    if soft_knee > 0.0:
        knee_start = max(threshold - soft_knee / 2.0, 0.0)
        knee_end = min(threshold + soft_knee / 2.0, 1.0)

        above = magnitudes > knee_end
        if np.any(above):
            compressed[above] = np.sign(compressed[above]) * (
                knee_end + (magnitudes[above] - knee_end) / ratio
            )

        knee_region = (magnitudes >= knee_start) & (magnitudes <= knee_end)
        if np.any(knee_region):
            x = magnitudes[knee_region]
            y = knee_end + (x - knee_end) / ratio
            blend = (x - knee_start) / max(soft_knee, 1e-6)
            interpolated = (1.0 - blend) * x + blend * y
            compressed[knee_region] = np.sign(compressed[knee_region]) * interpolated
    else:
        above = magnitudes > threshold
        if np.any(above):
            compressed[above] = np.sign(compressed[above]) * (
                threshold + (magnitudes[above] - threshold) / ratio
            )

    if noise_floor > 0.0:
        quiet = np.abs(compressed) < noise_floor
        if np.any(quiet):
            scaled = np.sqrt(np.clip(np.abs(compressed[quiet]) / noise_floor, 0.0, 1.0))
            compressed[quiet] = np.sign(compressed[quiet]) * noise_floor * scaled

    compressed *= makeup_gain

    peak = np.max(np.abs(compressed))
    ceiling = min(max(output_ceiling, 1e-6), 0.999)
    if peak > ceiling:
        compressed *= ceiling / peak

    return compressed.astype(original_dtype, copy=False)


def enhance_dialogue_audio(
    audio_clip: AudioClip,
    settings: Dict[str, Any] | None = None,
) -> AudioClip:
    """Return a new clip with dynamic range compression applied."""

    if audio_clip is None:
        raise ValueError("audio_clip must be provided")

    cfg = settings or {}
    threshold = float(np.clip(cfg.get("threshold", 0.35), 0.05, 0.95))
    ratio = float(max(cfg.get("ratio", 4.0), 1.0))
    soft_knee = float(np.clip(cfg.get("soft_knee", 0.08), 0.0, 1.0))
    makeup_gain = float(max(cfg.get("makeup_gain", 1.6), 0.0))
    noise_floor = float(np.clip(cfg.get("noise_floor", 0.02), 0.0, threshold))
    output_ceiling = float(np.clip(cfg.get("output_ceiling", 0.98), 0.1, 1.0))

    def _apply(get_frame, t):
        frame = get_frame(t)
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.float32)
        return _compress_frame(
            frame,
            threshold=threshold,
            ratio=ratio,
            soft_knee=soft_knee,
            makeup_gain=makeup_gain,
            noise_floor=noise_floor,
            output_ceiling=output_ceiling,
        )

    return audio_clip.fl(_apply)
