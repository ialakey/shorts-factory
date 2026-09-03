"""Вспомогательные функции рендера: имена файлов, чанки субтитров, геометрия."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ingestion.transcriber import rebuild_words_from_clean_text, sanitize_subtitle_text
from rendering.support import helper
from rendering import video_editor


class TestSafeFilename:
    @pytest.mark.parametrize("raw", ['a<b>c:d"e/f\\g|h?i*j', "«кавычки»"])
    def test_removes_forbidden_characters(self, raw):
        result = helper.safe_filename(raw)
        assert not set(result) & set('<>:"/\\|?*«»')

    def test_collapses_whitespace_and_underscores(self):
        assert helper.safe_filename("a   b__c") == "a_b_c"

    def test_truncates_long_names(self):
        assert len(helper.safe_filename("x" * 500)) == 150

    def test_strips_edges(self):
        assert helper.safe_filename("__name..") == "name"


class TestSplitSegmentByWords:
    def _segment(self, words):
        return {
            "start": 0.0,
            "end": float(len(words)),
            "words": [
                {"word": w, "start": float(i), "end": float(i + 1)}
                for i, w in enumerate(words)
            ],
        }

    def test_one_word_per_chunk(self):
        chunks = helper.split_segment_by_words(self._segment(["a", "b", "c"]), max_words=1)
        assert [c["text"] for c in chunks] == ["a", "b", "c"]

    def test_groups_words(self):
        chunks = helper.split_segment_by_words(self._segment(["a", "b", "c", "d"]), max_words=2)
        assert [c["text"] for c in chunks] == ["a b", "c d"]

    def test_chunk_timings_follow_words(self):
        chunks = helper.split_segment_by_words(self._segment(["a", "b"]), max_words=2)
        assert chunks[0]["start"] == 0.0
        assert chunks[0]["end"] == 2.0

    def test_highlight_tags_wrap_words(self):
        segment = self._segment(["до", "<hl>", "ключ", "</hl>", "после"])
        chunks = helper.split_segment_by_words(segment, max_words=1)
        texts = [c["text"] for c in chunks]
        assert texts == ["до", "<hl>ключ</hl>", "после"]

    def test_segment_without_words(self):
        assert helper.split_segment_by_words({"start": 0, "end": 1}, max_words=1) == []


class TestHighlightRoundTrip:
    """Регрессия: закрывающий тег не должен «прилипать» к слову.

    Раньше токенизатор склеивал "ВАЖНО</hl>", подсветка не выключалась до конца
    сегмента, а в кадр попадал литерал "</HL>".
    """

    def test_highlight_closes_on_its_word(self):
        segment = {"start": 0.0, "end": 3.0}
        rebuild_words_from_clean_text(segment, "ЭТО <hl>ВАЖНО</hl> ОЧЕНЬ")

        rendered = [
            sanitize_subtitle_text(chunk["text"])
            for chunk in helper.split_segment_by_words(segment, max_words=1)
        ]

        assert rendered == ["ЭТО", "<hl>ВАЖНО</hl>", "ОЧЕНЬ"]
        assert not any("</HL>" in text for text in rendered)


class TestTransparentBox:
    def test_detects_hole(self, tmp_path):
        img = Image.new("RGBA", (100, 200), (255, 0, 0, 255))
        for x in range(20, 60):
            for y in range(30, 90):
                img.putpixel((x, y), (0, 0, 0, 0))
        path = tmp_path / "bg.png"
        img.save(path)

        x, y, w, h, bg_w, bg_h = helper.get_transparent_box(str(path))

        assert (x, y, w, h) == (20, 30, 40, 60)
        assert (bg_w, bg_h) == (100, 200)

    def test_missing_alpha_raises(self, tmp_path):
        path = tmp_path / "opaque.png"
        Image.new("RGB", (10, 10), (1, 2, 3)).save(path)

        with pytest.raises(ValueError):
            helper.get_transparent_box(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            helper.get_transparent_box(str(tmp_path / "nope.png"))


class TestCanvasGeometry:
    def test_region_fully_inside(self):
        assert video_editor._visible_region_center(10, 100, 1000) == 60.0
        assert video_editor._visible_region_bounds(10, 100, 1000) == (10.0, 100.0)

    def test_region_clipped_by_canvas(self):
        left, width = video_editor._visible_region_bounds(-50, 200, 100)
        assert left == 0.0
        assert width == 100.0

    def test_region_outside_canvas_falls_back(self):
        centre = video_editor._visible_region_center(500, 100, 100)
        assert centre == 50.0
        left, width = video_editor._visible_region_bounds(500, 100, 100)
        assert 0.0 <= left <= 100.0
        assert width > 0

    def test_aligned_position_centers_clip(self):
        assert video_editor._aligned_position(100, 40) == 80.0


class TestBlurredBackground:
    def test_matches_canvas_size(self):
        from moviepy.editor import ColorClip

        clip = ColorClip(size=(64, 36), color=(10, 200, 30), duration=0.5)
        background = video_editor._build_blurred_background(
            clip, 108, 192, blur_strength=11, scale=1.0
        )

        frame = background.get_frame(0.1)
        assert frame.shape == (192, 108, 3)
        assert np.isfinite(frame).all()
