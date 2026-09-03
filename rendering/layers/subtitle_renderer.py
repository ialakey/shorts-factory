from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from moviepy.editor import ImageClip
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
from pilmoji import Pilmoji


@dataclass
class TextSegment:
    text: str
    highlighted: bool = False


@dataclass
class LineLayout:
    segments: List[TextSegment]
    width: int


def _parse_rgb(color: str) -> Tuple[int, int, int]:
    return ImageColor.getrgb(color)


_HL_TOKEN_PATTERN = re.compile(r"<\s*([\\/]*)\s*hl\s*>", re.IGNORECASE)
_HL_OPEN_PATTERN = re.compile(r"<\s*hl\s*>", re.IGNORECASE)
_HL_CLOSE_PATTERN = re.compile(r"<\s*[\\/]\s*hl\s*>", re.IGNORECASE)
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0001F1E6-\U0001F1FF]"
)


def _normalise_highlight_tags(text: str) -> str:
    """Unify highlight markers so parsing and stripping behave consistently."""

    def _replace(match: re.Match) -> str:
        prefix = (match.group(1) or "").strip()
        return "</hl>" if "/" in prefix else "<hl>"

    text = re.sub(r"<\s*h1\s*>", "<hl>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*h1\s*>", "</hl>", text, flags=re.IGNORECASE)
    text = _HL_TOKEN_PATTERN.sub(_replace, text)
    return text


def _strip_highlight_tags(text: str) -> str:
    """Remove all <hl> markers from text while keeping content intact."""

    text = _normalise_highlight_tags(text)
    text = _HL_OPEN_PATTERN.sub("", text)
    text = _HL_CLOSE_PATTERN.sub("", text)
    return text


def parse_highlighted_spans(text: str) -> List[TextSegment]:
    """Split a string into spans, preserving <hl>...</hl> markers."""

    text = _normalise_highlight_tags(text)
    pattern = re.compile(r"<\s*hl\s*>(.*?)<\s*[\\/]\s*hl\s*>", re.IGNORECASE | re.DOTALL)
    segments: List[TextSegment] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            plain = _strip_highlight_tags(text[last : match.start()])
            if plain:
                segments.append(TextSegment(text=plain, highlighted=False))
        highlighted_text = match.group(1)
        if highlighted_text:
            segments.append(TextSegment(text=highlighted_text, highlighted=True))
        last = match.end()
    if last < len(text):
        tail = _strip_highlight_tags(text[last:])
        if tail:
            segments.append(TextSegment(text=tail, highlighted=False))
    if not segments:
        stripped = _strip_highlight_tags(text)
        return [TextSegment(text=stripped, highlighted=False)] if stripped else []
    return segments


def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), int(font_size))


def _text_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int
) -> Tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)


def _measure_segment(
    draw: ImageDraw.ImageDraw,
    segment: TextSegment,
    base_font: ImageFont.FreeTypeFont,
    highlight_font: ImageFont.FreeTypeFont,
    stroke_width: int,
    highlight_stroke_width: int,
    emoji_scale: float,
    emoji_font_path: str,
) -> int:
    """Width measurement that honours emoji scaling and font fallbacks."""

    font = highlight_font if segment.highlighted else base_font
    stroke = highlight_stroke_width if segment.highlighted else stroke_width

    if _has_emoji(segment.text):
        return _measure_with_pilmoji(
            segment.text, font, stroke, emoji_font_path, emoji_scale
        )

    bbox = _text_bbox(draw, segment.text, font, stroke)
    return bbox[2] - bbox[0]


def _measure_with_pilmoji(
    text: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int,
    emoji_font_path: str,
    emoji_scale: float,
) -> int:
    """Draw to a tiny buffer with Pilmoji to get an accurate bbox for emojis."""

    # First, get a rough size estimate to pick a safe canvas.
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    rough_bbox = _text_bbox(ImageDraw.Draw(dummy), text, font, stroke_width)
    rough_w = max(rough_bbox[2] - rough_bbox[0], font.size)
    rough_h = max(rough_bbox[3] - rough_bbox[1], font.size)

    scale_hint = max(1.0, float(emoji_scale or 1.0))
    canvas_w = int(rough_w * scale_hint * 2 + 32)
    canvas_h = int(rough_h * scale_hint * 2 + 32)

    probe = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    with _pilmoji_context(probe, emoji_font_path, emoji_scale) as pilmoji:
        _draw_with_emoji_scale(
            pilmoji,
            (16, 16),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(255, 255, 255, 255),
            emoji_scale=emoji_scale,
        )

    bbox = probe.getbbox()
    if not bbox:
        return 0
    return bbox[2] - bbox[0]


def _split_tokens(segments: Iterable[TextSegment]) -> List[TextSegment]:
    tokens: List[TextSegment] = []
    for seg in segments:
        for chunk in re.findall(r"\S+\s*", seg.text):
            tokens.append(TextSegment(text=chunk, highlighted=seg.highlighted))
    return tokens


def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def _layout_lines(
    text: str,
    base_font: ImageFont.FreeTypeFont,
    highlight_font: ImageFont.FreeTypeFont,
    max_width: int,
    base_stroke_width: int,
    highlight_stroke_width: int,
    emoji_scale: float,
    emoji_font_path: str,
) -> Tuple[List[LineLayout], int]:
    dummy = Image.new("RGBA", (max_width, 2048), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(dummy)

    spans = parse_highlighted_spans(text)
    tokens = _split_tokens(spans)
    if not tokens:
        return [], 0

    lines: List[LineLayout] = []
    current: List[TextSegment] = []

    def line_width(segments: List[TextSegment]) -> int:
        width = 0
        for seg in segments:
            width += _measure_segment(
                drawer,
                seg,
                base_font,
                highlight_font,
                base_stroke_width,
                highlight_stroke_width,
                emoji_scale,
                emoji_font_path,
            )
        return width

    for token in tokens:
        probe = current + [token]
        if line_width(probe) <= max_width or not current:
            current = probe
        else:
            lines.append(LineLayout(segments=current, width=line_width(current)))
            current = [token]
    if current:
        lines.append(LineLayout(segments=current, width=line_width(current)))

    base_metr = drawer.textbbox(
        (0, 0),
        "Ay",
        font=base_font,
        stroke_width=base_stroke_width,
    )
    hl_metr = drawer.textbbox(
        (0, 0),
        "Ay",
        font=highlight_font,
        stroke_width=highlight_stroke_width,
    )
    emoji_factor = 1.0
    if any(_has_emoji(tok.text) for tok in tokens):
        emoji_factor = max(1.0, float(emoji_scale or 1.0))
    line_height = int(max(base_metr[3] - base_metr[1], hl_metr[3] - hl_metr[1]) * emoji_factor) + 2
    return lines, line_height


def _pilmoji_context(
    image: Image.Image, emoji_font_path: str, emoji_scale: float | None = None
) -> Pilmoji:
    """Return a Pilmoji instance while being backwards compatible with older versions.

    Some Pilmoji releases do not accept ``emoji_font_path`` in the constructor, so we
    try the modern signature first and gracefully fall back to the legacy one.
    """

    try:
        pilmoji = Pilmoji(image, emoji_font_path=emoji_font_path)
    except TypeError:
        pilmoji = Pilmoji(image)
        if emoji_font_path and hasattr(pilmoji, "emoji_font_path"):
            pilmoji.emoji_font_path = emoji_font_path
    if emoji_scale is not None:
        for attr in ("emoji_scale", "emoji_scale_factor"):
            if hasattr(pilmoji, attr):
                try:
                    setattr(pilmoji, attr, emoji_scale)
                except Exception:
                    pass
    return pilmoji


def _draw_with_emoji_scale(
    pilmoji: Pilmoji,
    position: Tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int, int] | Tuple[int, int, int],
    stroke_width: int,
    stroke_fill: Tuple[int, int, int, int] | Tuple[int, int, int],
    emoji_scale: float,
) -> None:
    tried_kwargs = [
        {"emoji_scale": emoji_scale},
        {"emoji_scale_factor": emoji_scale},
    ]

    for extra in tried_kwargs:
        try:
            pilmoji.text(
                position,
                text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
                emoji_position_offset=(0, 0),
                **extra,
            )
            return
        except TypeError:
            continue

    if emoji_scale is not None:
        for attr in ("emoji_scale", "emoji_scale_factor"):
            if hasattr(pilmoji, attr):
                try:
                    setattr(pilmoji, attr, emoji_scale)
                except Exception:
                    pass

    pilmoji.text(
        position,
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        emoji_position_offset=(0, 0),
    )


def apply_shadow(
    base: Image.Image,
    lines: List[LineLayout],
    *,
    base_font: ImageFont.FreeTypeFont,
    highlight_font: ImageFont.FreeTypeFont,
    emoji_font_path: str,
    emoji_scale: float,
    align: str,
    line_height: int,
    default_style: dict,
    highlight_style: dict,
    padding: int = 0,
) -> Image.Image:
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(shadow)
    with _pilmoji_context(shadow, emoji_font_path, emoji_scale) as pilmoji:
        for idx, line in enumerate(lines):
            x = _align_x(base.width - 2 * padding, line.width, align) + padding
            y = padding + idx * line_height
            for seg in line.segments:
                style = highlight_style if seg.highlighted else default_style
                draw_width = _measure_segment(
                    drawer,
                    seg,
                    base_font,
                    highlight_font,
                    default_style.get("stroke_width", 0),
                    style.get("stroke_width", default_style.get("stroke_width", 0)),
                    emoji_scale,
                    emoji_font_path,
                )
                if _has_emoji(seg.text):
                    x += draw_width
                    continue
                color = style.get("shadow_color") or default_style.get("shadow_color")
                opacity = style.get("shadow_opacity", 0.0)
                offset_x = style.get("shadow_offset_x", 0)
                offset_y = style.get("shadow_offset_y", 0)
                stroke_width = style.get("stroke_width", default_style.get("stroke_width", 0))
                font = highlight_font if seg.highlighted else base_font
                if not color or opacity <= 0:
                    draw_width = _text_bbox(ImageDraw.Draw(shadow), seg.text, font, stroke_width)
                    x += draw_width[2] - draw_width[0]
                    continue
                fill = (*_parse_rgb(color), int(255 * float(opacity)))
                _draw_with_emoji_scale(
                    pilmoji,
                    (x + int(offset_x), y + int(offset_y)),
                    seg.text,
                    font=font,
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=fill,
                    emoji_scale=emoji_scale,
                )
                x += draw_width
    blur = float(default_style.get("shadow_blur") or 0.0)
    if blur > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(base, shadow)


def apply_glow(
    base: Image.Image,
    lines: List[LineLayout],
    *,
    base_font: ImageFont.FreeTypeFont,
    highlight_font: ImageFont.FreeTypeFont,
    emoji_font_path: str,
    emoji_scale: float,
    align: str,
    line_height: int,
    default_style: dict,
    highlight_style: dict,
    padding: int = 0,
) -> Image.Image:
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(glow_layer)
    with _pilmoji_context(glow_layer, emoji_font_path, emoji_scale) as pilmoji:
        for idx, line in enumerate(lines):
            x = _align_x(base.width - 2 * padding, line.width, align) + padding
            y = padding + idx * line_height
            for seg in line.segments:
                style = highlight_style if seg.highlighted else default_style
                font = highlight_font if seg.highlighted else base_font
                draw_width = _measure_segment(
                    drawer,
                    seg,
                    base_font,
                    highlight_font,
                    default_style.get("stroke_width", 0),
                    style.get("stroke_width", default_style.get("stroke_width", 0)),
                    emoji_scale,
                    emoji_font_path,
                )
                if _has_emoji(seg.text):
                    x += draw_width
                    continue
                color = style.get("glow_color") or default_style.get("glow_color")
                radius = float(style.get("glow_radius") or default_style.get("glow_radius") or 0.0)
                stroke_width = style.get("stroke_width", default_style.get("stroke_width", 0))
                if not color or radius <= 0:
                    bbox = _text_bbox(ImageDraw.Draw(glow_layer), seg.text, font, stroke_width)
                    x += bbox[2] - bbox[0]
                    continue
                fill = (*_parse_rgb(color), 255)
                _draw_with_emoji_scale(
                    pilmoji,
                    (x, y),
                    seg.text,
                    font=font,
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=fill,
                    emoji_scale=emoji_scale,
                )
                x += draw_width
    max_radius = float(default_style.get("glow_radius") or 0.0)
    max_radius = max(max_radius, float(highlight_style.get("glow_radius") or 0.0))
    if max_radius > 0:
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(max_radius))
    return Image.alpha_composite(base, glow_layer)


def apply_stroke(
    base: Image.Image,
    lines: List[LineLayout],
    *,
    base_font: ImageFont.FreeTypeFont,
    highlight_font: ImageFont.FreeTypeFont,
    emoji_font_path: str,
    emoji_scale: float,
    align: str,
    line_height: int,
    default_style: dict,
    highlight_style: dict,
    padding: int = 0,
) -> Image.Image:
    stroke_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(stroke_layer)
    with _pilmoji_context(stroke_layer, emoji_font_path, emoji_scale) as pilmoji:
        for idx, line in enumerate(lines):
            x = _align_x(base.width - 2 * padding, line.width, align) + padding
            y = padding + idx * line_height
            for seg in line.segments:
                style = highlight_style if seg.highlighted else default_style
                font = highlight_font if seg.highlighted else base_font
                stroke_width = int(style.get("stroke_width", default_style.get("stroke_width", 0)))
                stroke_color = style.get("stroke_color") or default_style.get("stroke_color")
                fill_color = style.get("color") or default_style.get("color")
                if _has_emoji(seg.text):
                    stroke_width = 0
                    stroke_color = None
                    fill_color = "#FFFFFF"
                if not fill_color:
                    fill_color = "#FFFFFF"
                _draw_with_emoji_scale(
                    pilmoji,
                    (x, y),
                    seg.text,
                    font=font,
                    fill=_parse_rgb(fill_color),
                    stroke_width=stroke_width,
                    stroke_fill=_parse_rgb(stroke_color) if stroke_color else None,
                    emoji_scale=emoji_scale,
                )
                draw_width = _measure_segment(
                    drawer,
                    seg,
                    base_font,
                    highlight_font,
                    default_style.get("stroke_width", 0),
                    style.get("stroke_width", default_style.get("stroke_width", 0)),
                    emoji_scale,
                    emoji_font_path,
                )
                x += draw_width
    return Image.alpha_composite(base, stroke_layer)


def _align_x(total_width: int, line_width: int, align: str) -> int:
    align = (align or "center").lower()
    if align == "left":
        return 0
    if align == "right":
        return total_width - line_width
    return (total_width - line_width) // 2


def render_text_layer(
    *,
    text: str,
    font_path: str,
    font_size: int,
    highlight_font_size: int | None = None,
    emoji_font_size: int | None = None,
    color: str,
    stroke_color: str,
    stroke_width: int,
    max_width: int,
    align: str = "center",
    highlight_style: dict | None = None,
    emoji_font_path: str = "NotoColorEmoji.ttf",
    shadow_color: str | None = None,
    shadow_opacity: float | None = 0.0,
    shadow_offset_x: int | float = 0,
    shadow_offset_y: int | float = 1,
    shadow_blur: float | None = 0.0,
    glow_color: str | None = None,
    glow_radius: float | None = 0.0,
    bg_color: str | None = None,
    bg_opacity: float | None = 0.0,
) -> np.ndarray:
    """Render stylised text layer with emoji and highlight support."""

    text = (text or "").strip()
    if not text:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    base_font = _load_font(font_path, font_size)
    highlight_font = _load_font(font_path, int(highlight_font_size or font_size))
    emoji_scale = 1.0
    if emoji_font_size:
        try:
            emoji_scale = max(0.1, float(emoji_font_size) / float(font_size or 1))
        except Exception:
            emoji_scale = 1.0
    default_style = {
        "color": color,
        "stroke_color": stroke_color,
        "stroke_width": stroke_width,
        "shadow_color": shadow_color,
        "shadow_opacity": shadow_opacity or 0.0,
        "shadow_offset_x": shadow_offset_x or 0,
        "shadow_offset_y": shadow_offset_y or 0,
        "shadow_blur": shadow_blur or 0.0,
        # "glow_color": glow_color,
        # "glow_radius": glow_radius or 0.0,
    }
    highlight_defaults = {
        "color": "#FF4B4B",
        "stroke_color": stroke_color,
        "stroke_width": stroke_width + 1,
        "font_size": highlight_font_size or font_size,
        "shadow_color": shadow_color,
        "shadow_opacity": shadow_opacity or 0.0,
        "shadow_offset_x": shadow_offset_x or 0,
        "shadow_offset_y": shadow_offset_y or 0,
        "shadow_blur": shadow_blur or 0.0,
        # "glow_color": "#FF6666",
        # "glow_radius": max(float(glow_radius or 0.0), 4.0),
    }
    highlight_style = {**highlight_defaults, **(highlight_style or {})}

    stroke_pad = max(
        int(default_style["stroke_width"]),
        int(highlight_style.get("stroke_width", default_style["stroke_width"])),
    )
    shadow_pad = 0
    if shadow_color and (shadow_opacity or 0) > 0:
        shadow_pad = int(
            max(abs(shadow_offset_x or 0), abs(shadow_offset_y or 0))
            + float(shadow_blur or 0.0)
        )
    padding = max(stroke_pad, shadow_pad) + 2

    layout_width = max(10, int(max_width) - 2 * padding)
    lines, line_height = _layout_lines(
        text,
        base_font,
        highlight_font,
        layout_width,
        int(default_style["stroke_width"]),
        int(highlight_style.get("stroke_width", default_style["stroke_width"])),
        emoji_scale,
        emoji_font_path,
    )
    if not lines:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    img_height = line_height * len(lines) + 2 * padding
    content_width = max(line.width for line in lines)
    base_width = max(content_width + 2 * padding, 1)
    if base_width > layout_width + 2 * padding:
        base_width = layout_width + 2 * padding

    background_color = (0, 0, 0, 0)
    if bg_color and str(bg_color).lower() != "transparent" and (bg_opacity or 0) > 0:
        rgb = _parse_rgb(bg_color)
        background_color = (*rgb, int(255 * max(0.0, min(1.0, float(bg_opacity or 0)))))

    base = Image.new("RGBA", (base_width, img_height), background_color)

    if shadow_color and (shadow_opacity or 0) > 0:
        base = apply_shadow(
            base,
            lines,
            base_font=base_font,
            highlight_font=highlight_font,
            emoji_font_path=emoji_font_path,
            emoji_scale=emoji_scale,
            align=align,
            line_height=line_height,
            default_style=default_style,
            highlight_style=highlight_style,
            padding=padding,
        )

    # if glow_color or highlight_style.get("glow_color"):
    #     base = apply_glow(
    #         base,
    #         lines,
    #         base_font=base_font,
    #         highlight_font=highlight_font,
    #         emoji_font_path=emoji_font_path,
    #         emoji_scale=emoji_scale,
    #         align=align,
    #         line_height=line_height,
    #         default_style=default_style,
    #         highlight_style=highlight_style,
    #         padding=padding,
    #     )

    base = apply_stroke(
        base,
        lines,
        base_font=base_font,
        highlight_font=highlight_font,
        emoji_font_path=emoji_font_path,
        emoji_scale=emoji_scale,
        align=align,
        line_height=line_height,
        default_style=default_style,
        highlight_style=highlight_style,
        padding=padding,
    )

    return np.array(base)


def _normalise_alignment(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return str(value).strip().lower() or fallback


def _extract_position(position: Sequence[str | None] | None) -> Tuple[str | None, str | None]:
    if not position:
        return None, None
    try:
        horizontal = position[0] if len(position) > 0 else None
    except TypeError:
        horizontal = None
    try:
        vertical = position[1] if len(position) > 1 else None
    except TypeError:
        vertical = None
    return horizontal, vertical


def make_subtitle_clip(
    *,
    text: str,
    start: float,
    end: float,
    font_path: str,
    font_size: int,
    highlight_font_size: int | None = None,
    emoji_font_size: int | None = None,
    color: str,
    stroke_color: str,
    stroke_width: int,
    max_width: int,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    bottom_padding: float | None = None,
    fade_in: float | None = 0.25,
    fade_out: float | None = 0.25,
    position: Sequence[str | None] | None = None,
    align: str | None = None,
    vertical_align: str | None = None,
    padding_x: float | None = None,
    padding_y: float | None = None,
    emoji_font_path: str = "NotoColorEmoji.ttf",
    highlight_style: dict | None = None,
    **kwargs,
) -> ImageClip:
    """Create a positioned subtitle clip respecting custom alignment and styling."""

    img = render_text_layer(
        text=text,
        font_path=font_path,
        font_size=font_size,
        highlight_font_size=highlight_font_size,
        emoji_font_size=emoji_font_size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        max_width=max_width,
        align=_normalise_alignment(align, "center"),
        highlight_style=highlight_style,
        emoji_font_path=emoji_font_path,
        **kwargs,
    )
    clip = ImageClip(img).set_start(start).set_end(end)

    horiz_pos, vert_pos = _extract_position(position)
    horizontal = _normalise_alignment(horiz_pos, _normalise_alignment(align, "center"))
    vertical = _normalise_alignment(vert_pos, _normalise_alignment(vertical_align, "center"))

    pad_x = float(padding_x) if padding_x is not None else 0.0
    pad_y = float(padding_y) if padding_y is not None else (
        float(bottom_padding) if bottom_padding is not None else 0.0
    )

    if horizontal == "left":
        x = box_x + pad_x
    elif horizontal == "right":
        x = box_x + box_w - clip.w - pad_x
    else:
        x = box_x + (box_w - clip.w) / 2.0 + pad_x

    if vertical == "top":
        y = box_y + pad_y
    elif vertical == "bottom":
        y = box_y + box_h - clip.h - pad_y
    else:
        y = box_y + (box_h - clip.h) / 2.0 + pad_y

    if fade_in and fade_in > 0:
        clip = clip.crossfadein(fade_in)
    if fade_out and fade_out > 0:
        clip = clip.crossfadeout(fade_out)

    return clip.set_position((x, y))
