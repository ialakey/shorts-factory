from __future__ import annotations

import math

import numpy as np
from moviepy.editor import ImageClip

from rendering.layers.subtitle_renderer import render_text_layer


def _wrap_by_words(text: str, max_words_per_line: int | None) -> str:
    if not text:
        return ""
    if not max_words_per_line or max_words_per_line <= 0:
        return text

    words = text.split()
    if not words:
        return text

    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i : i + max_words_per_line]))
    return "\n".join(lines)


def _scale_title(
    *,
    text: str,
    font_path: str,
    font_size: int,
    min_font_size: int,
    max_width: int,
    max_height: int | None,
    color: str,
    stroke_color: str,
    stroke_width: int,
    align: str,
    emoji_font_size: int | None,
    bg_color: str | None,
    bg_opacity: float | None,
    shadow_color: str | None,
    shadow_opacity: float | None,
    shadow_offset_x: float | None,
    shadow_offset_y: float | None,
    shadow_blur: float | None,
    emoji_font_path: str,
) -> tuple[np.ndarray, int]:
    """Render and downscale the title until it fits the target box."""

    current_size = max(1, int(font_size))
    min_size = max(1, int(min_font_size))
    attempts = 0
    last_img: np.ndarray | None = None

    while attempts < 20 and current_size >= min_size:
        img = render_text_layer(
            text=text,
            font_path=font_path,
            font_size=current_size,
            emoji_font_size=emoji_font_size,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            max_width=max_width,
            align=align,
            bg_color=bg_color,
            bg_opacity=bg_opacity,
            shadow_color=shadow_color,
            shadow_opacity=shadow_opacity,
            shadow_offset_x=shadow_offset_x,
            shadow_offset_y=shadow_offset_y,
            shadow_blur=shadow_blur,
            emoji_font_path=emoji_font_path,
        )
        last_img = img

        h, w = img.shape[0], img.shape[1]
        fits_width = w <= max_width
        fits_height = True if max_height is None else h <= max_height
        if fits_width and fits_height:
            return img, current_size

        current_size = max(min_size, int(math.floor(current_size * 0.92)))
        attempts += 1

    return last_img if last_img is not None else np.zeros((1, 1, 4), dtype=np.uint8), current_size


def make_title_clip(
    *,
    text: str,
    font_path: str,
    font_size: int,
    color: str,
    stroke_color: str,
    stroke_width: int,
    max_width: int,
    video_width: int,
    top_padding: int = 80,
    max_height: int | None = None,
    max_words_per_line: int | None = None,
    min_font_size: int | None = None,
    emoji_font_size: int | None = None,
    bg_color: str | None = None,
    bg_opacity: float | None = 0.0,
    shadow_color: str | None = None,
    shadow_opacity: float | None = None,
    shadow_offset_x: float | None = None,
    shadow_offset_y: float | None = None,
    shadow_blur: float | None = None,
    fade_in: float | None = 0.25,
    fade_out: float | None = 0.25,
    start: float = 0.0,
    duration: float | None = None,
    align: str = "center",
    emoji_font_path: str = "NotoColorEmoji.ttf",
) -> ImageClip:
    """Create a stylised title clip with emoji/background support."""

    prepared_text = _wrap_by_words(text, max_words_per_line)
    min_size = min_font_size or max(12, int(font_size * 0.6))

    img, resolved_font_size = _scale_title(
        text=prepared_text,
        font_path=font_path,
        font_size=font_size,
        min_font_size=min_size,
        max_width=max_width,
        max_height=max_height,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        align=align,
        emoji_font_size=emoji_font_size,
        bg_color=bg_color,
        bg_opacity=bg_opacity,
        shadow_color=shadow_color,
        shadow_opacity=shadow_opacity,
        shadow_offset_x=shadow_offset_x,
        shadow_offset_y=shadow_offset_y,
        shadow_blur=shadow_blur,
        emoji_font_path=emoji_font_path,
    )

    clip = ImageClip(img, ismask=False).set_start(start)
    if duration is not None:
        clip = clip.set_duration(duration)

    x = (video_width - clip.w) / 2
    y = top_padding
    clip = clip.set_position((x, y))

    if fade_in and fade_in > 0:
        clip = clip.crossfadein(fade_in)
    if fade_out and fade_out > 0:
        clip = clip.crossfadeout(fade_out)

    clip.title_font_size = resolved_font_size  # type: ignore[attr-defined]
    return clip
