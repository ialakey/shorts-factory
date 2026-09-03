from PIL import Image, ImageDraw, ImageFont, ImageColor
from moviepy.editor import ImageClip
import numpy as np


def make_watermark_clip(
        text: str,
        font_path: str,
        font_size: int,
        color: str,
        opacity: float,
        position: tuple,
        padding_x: int,
        padding_y: int,
        video_w: int,
        video_h: int,
        shadow_color: str = None,
        shadow_opacity: float = 0.0,
        shadow_offset: int = 0,
        duration: float = 5.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
):
    """Создаёт полупрозрачный текстовый вотермарк без ImageMagick."""
    font = ImageFont.truetype(str(font_path), font_size)
    text = text.strip()

    # примерное ограничение размеров
    dummy = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img = Image.new("RGBA", (text_w + 10, text_h + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # тень (если указана)
    if shadow_color and shadow_opacity > 0:
        shadow_rgba = (*ImageColor.getrgb(shadow_color), int(255 * shadow_opacity))
        draw.text(
            (shadow_offset, shadow_offset),
            text,
            font=font,
            fill=shadow_rgba,
        )

    # основной текст
    rgba = (*ImageColor.getrgb(color), int(255 * opacity))
    draw.text((0, 0), text, font=font, fill=rgba)

    arr = np.array(img)

    clip = ImageClip(arr).set_duration(duration)

    # позиционирование
    x, y = 0, 0
    horiz, vert = position
    if horiz == "right":
        x = video_w - img.width - padding_x
    elif horiz == "center":
        x = (video_w - img.width) / 2
    else:
        x = padding_x

    if vert == "bottom":
        y = video_h - img.height - padding_y
    elif vert == "center":
        y = (video_h - img.height) / 2
    else:
        y = padding_y

    return clip.set_position((x + float(offset_x), y + float(offset_y)))