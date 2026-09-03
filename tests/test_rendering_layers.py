"""Слои рендера: субтитры, титр, водяной знак, компрессия звука."""

from __future__ import annotations

import numpy as np
import pytest
from moviepy.audio.AudioClip import AudioClip

from rendering.audio import enhance_dialogue_audio
from rendering.layers import subtitle_renderer, title_renderer, watermark_renderer


@pytest.fixture
def font(project_font):
    return str(project_font)


class TestRenderTextLayer:
    def test_returns_rgba_image(self, font):
        layer = subtitle_renderer.render_text_layer(
            text="HELLO", font_path=font, font_size=24,
            color="#FFFFFF", stroke_color="#000000", stroke_width=2, max_width=300,
        )

        assert layer.ndim == 3 and layer.shape[2] == 4
        assert layer.shape[0] > 0 and layer.shape[1] > 0
        assert layer[..., 3].max() > 0, "текст не отрисовался (полностью прозрачный слой)"

    def test_empty_text_gives_empty_layer(self, font):
        layer = subtitle_renderer.render_text_layer(
            text="   ", font_path=font, font_size=24,
            color="#FFFFFF", stroke_color="#000000", stroke_width=2, max_width=300,
        )
        assert layer.shape == (1, 1, 4)

    def test_highlight_uses_its_own_color(self, font):
        common = dict(
            font_path=font, font_size=24, color="#FFFFFF",
            stroke_color="#000000", stroke_width=1, max_width=400,
        )
        plain = subtitle_renderer.render_text_layer(text="WORD", **common)
        highlighted = subtitle_renderer.render_text_layer(
            text="<hl>WORD</hl>", highlight_style={"color": "#FF0000"}, **common
        )

        assert highlighted[..., 0].max() > 0
        # красный акцент даёт заметно меньше зелёного канала, чем белый текст
        assert highlighted[..., 1].sum() < plain[..., 1].sum()

    def test_long_text_wraps_within_max_width(self, font):
        layer = subtitle_renderer.render_text_layer(
            text="ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT", font_path=font, font_size=24,
            color="#FFFFFF", stroke_color="#000000", stroke_width=2, max_width=150,
        )
        assert layer.shape[1] <= 150 + 40  # запас на обводку/тень

    def test_background_box_is_drawn(self, font):
        layer = subtitle_renderer.render_text_layer(
            text="BG", font_path=font, font_size=24, color="#FFFFFF",
            stroke_color="#000000", stroke_width=1, max_width=200,
            bg_color="#00FF00", bg_opacity=1.0,
        )
        assert layer[..., 1].max() > 100


class TestMakeSubtitleClip:
    def test_positioned_inside_video_box(self, font):
        clip = subtitle_renderer.make_subtitle_clip(
            text="HELLO", start=0.5, end=1.5, font_path=font, font_size=20,
            color="#FFFFFF", stroke_color="#000000", stroke_width=2, max_width=180,
            box_x=0, box_y=0, box_w=216, box_h=384, bottom_padding=20,
        )

        assert clip.start == pytest.approx(0.5)
        assert clip.duration == pytest.approx(1.0)
        assert clip.w <= 216


class TestMakeTitleClip:
    def test_title_fits_canvas(self, font):
        clip = title_renderer.make_title_clip(
            text="ЗАГОЛОВОК КЛИПА", start=0.0, duration=1.0, font_path=font,
            font_size=28, color="#FFFFFF", stroke_color="#000000", stroke_width=2,
            max_width=200, video_width=216, top_padding=10, max_height=90,
            max_words_per_line=2,
        )

        assert clip.duration == pytest.approx(1.0)
        assert clip.w <= 216

    def test_very_long_title_shrinks_but_not_below_min_font_size(self, font):
        """Титр ужимается под бокс, но не мельче min_font_size — иначе нечитаемо."""

        clip = title_renderer.make_title_clip(
            text="ОЧЕНЬ ДЛИННЫЙ ЗАГОЛОВОК КОТОРЫЙ ТОЧНО НЕ ВЛЕЗЕТ В ОДНУ СТРОКУ",
            start=0.0, duration=1.0, font_path=font, font_size=48, color="#FFFFFF",
            stroke_color="#000000", stroke_width=2, max_width=200, video_width=216,
            top_padding=10, max_height=80, max_words_per_line=3, min_font_size=20,
        )

        assert 20 <= clip.title_font_size < 48


class TestWatermark:
    def _clip(self, font, position):
        return watermark_renderer.make_watermark_clip(
            text="I_Alakey", font_path=font, font_size=16, color="#FFFFFF",
            opacity=0.5, position=position, padding_x=4, padding_y=6,
            video_w=216, video_h=384, duration=1.0,
        )

    def test_bottom_center_is_inside_frame(self, font):
        clip = self._clip(font, ("center", "bottom"))
        x, y = clip.pos(0)

        assert 0 <= x <= 216 - clip.w + 1
        assert 0 <= y <= 384 - clip.h + 1

    def test_top_left_respects_padding(self, font):
        clip = self._clip(font, ("left", "top"))
        x, y = clip.pos(0)

        assert x == pytest.approx(4)
        assert y == pytest.approx(6)

    def test_offset_shifts_watermark(self, font):
        base = self._clip(font, ("left", "top")).pos(0)
        shifted = watermark_renderer.make_watermark_clip(
            text="I_Alakey", font_path=font, font_size=16, color="#FFFFFF",
            opacity=0.5, position=("left", "top"), padding_x=4, padding_y=6,
            video_w=216, video_h=384, duration=1.0, offset_x=30, offset_y=40,
        ).pos(0)

        assert shifted[0] == pytest.approx(base[0] + 30)
        assert shifted[1] == pytest.approx(base[1] + 40)


class TestAudioEnhancer:
    def _clip(self, amplitude=0.9, freq=220.0, duration=0.5, fps=8000):
        def make_frame(t):
            wave = amplitude * np.sin(2 * np.pi * freq * np.asarray(t))
            return np.array([wave, wave]).T if np.ndim(t) else np.array([wave, wave])

        return AudioClip(make_frame, duration=duration, fps=fps)

    def test_output_never_clips(self):
        enhanced = enhance_dialogue_audio(self._clip(amplitude=1.0), {"output_ceiling": 0.9})

        samples = enhanced.to_soundarray(fps=8000)
        assert np.max(np.abs(samples)) <= 0.9 + 1e-3

    def test_quiet_audio_gets_louder(self):
        cfg = {"threshold": 0.3, "ratio": 4.0, "makeup_gain": 2.0, "output_ceiling": 0.99}
        quiet = self._clip(amplitude=0.05)

        before = np.max(np.abs(quiet.to_soundarray(fps=8000)))
        after = np.max(np.abs(enhance_dialogue_audio(quiet, cfg).to_soundarray(fps=8000)))

        assert after > before

    def test_dynamic_range_is_compressed(self):
        cfg = {"threshold": 0.2, "ratio": 8.0, "makeup_gain": 1.0, "output_ceiling": 0.99}
        loud = self._clip(amplitude=1.0)

        peak_before = np.max(np.abs(loud.to_soundarray(fps=8000)))
        peak_after = np.max(np.abs(enhance_dialogue_audio(loud, cfg).to_soundarray(fps=8000)))

        assert peak_after < peak_before

    def test_none_clip_raises(self):
        with pytest.raises(ValueError):
            enhance_dialogue_audio(None, {})

    def test_default_settings_work(self):
        enhanced = enhance_dialogue_audio(self._clip(), None)
        assert np.isfinite(enhanced.to_soundarray(fps=8000)).all()
