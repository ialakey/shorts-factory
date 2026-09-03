import json
import os
import random
import tempfile
from pathlib import Path

import cv2
import whisper
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
)
import moviepy.audio.fx.all as afx

from analysis.subtitles_cleaner import clean_and_correct_text
from config import OPENAI_API_KEY
from ingestion.transcriber import (
    rebuild_words_from_clean_text,
    sanitize_subtitle_text,
    select_whisper_model,
)
from rendering.support.helper import (
    safe_filename,
    split_segment_by_words,
    get_transparent_box,
)
from rendering.layers.subtitle_renderer import make_subtitle_clip
from rendering.layers.title_renderer import make_title_clip
from rendering.layers.watermark_renderer import make_watermark_clip
from rendering.face_detector import build_dynamic_short_clip
from rendering.audio import enhance_dialogue_audio


def _visible_region_center(start, length, canvas_size):
    """Return the centre of the visible portion of a region.

    Parameters
    ----------
    start:
        The left/top coordinate of the region relative to the canvas.
    length:
        The width/height of the region.
    canvas_size:
        The total width/height of the canvas.
    """

    start = float(start)
    length = float(length)
    canvas_size = float(canvas_size)

    left = start
    right = start + length

    visible_left = max(0.0, min(canvas_size, left))
    visible_right = max(visible_left, min(canvas_size, max(left, right)))

    # If the region lies completely outside of the canvas fall back to the
    # canvas centre to avoid NaNs.
    visible_width = visible_right - visible_left
    if visible_width == 0:
        return canvas_size / 2.0

    return visible_left + visible_width / 2.0


def _visible_region_bounds(start, length, canvas_size):
    """Return the visible left coordinate and width of a region."""

    start = float(start)
    length = float(length)
    canvas_size = float(canvas_size)

    visible_left = max(0.0, min(canvas_size, start))
    visible_right = max(visible_left, min(canvas_size, start + length))
    visible_width = visible_right - visible_left

    if visible_width <= 0.0:
        centre = _visible_region_center(start, length, canvas_size)
        visible_width = max(1.0, min(canvas_size, abs(length)))
        half = visible_width / 2.0
        visible_left = max(0.0, min(canvas_size - visible_width, centre - half))

    return visible_left, visible_width


def _aligned_position(centre, clip_size):
    """Convert a centre coordinate into the top-left position."""

    return float(centre) - float(clip_size) / 2.0


def _build_blurred_background(
    clip,
    width,
    height,
    *,
    blur_strength=55,
    scale=1.3,
):
    """Create a blurred background from ``clip`` covering ``width`` x ``height``.

    The clip is first scaled to cover the full canvas and optionally enlarged by
    ``scale`` to give some breathing room. A strong Gaussian blur is then
    applied so that the colours match the foreground without distracting
    details.
    """

    width = int(round(width))
    height = int(round(height))
    if width <= 0 or height <= 0:
        raise ValueError("Background dimensions must be positive")

    base_scale = max(width / clip.w, height / clip.h)
    target_scale = base_scale * max(1.0, float(scale))

    scaled = clip.resize(target_scale)
    x_center = scaled.w / 2.0
    y_center = scaled.h / 2.0

    try:
        cropped = scaled.fx(
            vfx.crop,
            width=width,
            height=height,
            x_center=x_center,
            y_center=y_center,
        )
    except TypeError:
        cropped = scaled.crop(
            width=width,
            height=height,
            x_center=x_center,
            y_center=y_center,
        )

    kernel_size = int(round(blur_strength))
    if kernel_size <= 0:
        blurred = cropped
    else:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 301))
        sigma = kernel_size / 3.0

        def _blur_frame(frame):
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)

        blurred = cropped.fl_image(_blur_frame)

    return blurred.set_duration(clip.duration).set_position((0, 0))


def make_clips(
    video_path,
    moments,
    output_folder,
    cfg,
    *,
    pipeline=None,
    channel_dir=None,
):
    vcfg = cfg["video"]
    vecfg = cfg["video_effects"]
    mcfg = cfg["music"]
    scfg = cfg["subtitles"]
    tcfg = cfg.get("title", {})
    wcfg = cfg.get("watermark", {})
    bgcfg = cfg.get("background", {})
    dynamic_cfg = cfg.get("dynamic_shorts", {})
    audio_enhancer_cfg = cfg.get("audio_enhancer", {})
    base_canvas_width = int(vcfg.get("width", 1080))
    base_canvas_height = int(vcfg.get("height", 1920))

    pipeline_set = set(pipeline or [])
    feature_steps = {
        "dynamic_shorts",
        "video_effects",
        "speed",
        "watermark",
        "enhance_audio",
        "music",
        "subtitles",
        "title",
    }
    feature_gate_active = bool(pipeline_set & feature_steps)

    effects_pipeline_enabled = not feature_gate_active or "video_effects" in pipeline_set
    speed_pipeline_enabled = (
        not feature_gate_active
        or "speed" in pipeline_set
        or "video_effects" in pipeline_set
    )
    watermark_pipeline_enabled = not feature_gate_active or "watermark" in pipeline_set
    subtitles_pipeline_enabled = not feature_gate_active or "subtitles" in pipeline_set
    music_pipeline_enabled = not feature_gate_active or "music" in pipeline_set
    title_pipeline_enabled = not feature_gate_active or "title" in pipeline_set
    def _section_enabled(section, default, stage_name=None):
        flag = section.get("enabled") if isinstance(section, dict) else None
        if flag is None and stage_name and feature_gate_active and stage_name in pipeline_set:
            return True
        return default if flag is None else bool(flag)

    def _feature_allowed(step_name, default):
        if feature_gate_active:
            return step_name in pipeline_set
        return default

    WORDS_PER_CHUNK = scfg.get("words_per_chunk", 1)
    SPEED_MULTIPLIER = vecfg.get("speed", 1.0)
    FRAME_RATE = vecfg.get("frame_rate", None)
    SUBTITLES_LANGUAGE = str(scfg.get("language", "")).strip() or None

    effects_cfg_enabled = _section_enabled(vecfg, True)
    speed_cfg_enabled = _section_enabled(vecfg, True, stage_name="speed")
    watermark_cfg_enabled = _section_enabled(
        wcfg, False, stage_name="watermark"
    )
    subtitles_cfg_enabled = _section_enabled(scfg, True)
    music_cfg_enabled = _section_enabled(mcfg, True)
    title_cfg_enabled = _section_enabled(tcfg, True)
    dynamic_cfg_enabled = _section_enabled(
        dynamic_cfg, False, stage_name="dynamic_shorts"
    )
    audio_enhancer_cfg_enabled = _section_enabled(
        audio_enhancer_cfg, True, stage_name="enhance_audio"
    )
    APPLY_EFFECTS = effects_cfg_enabled and _feature_allowed(
        "video_effects", effects_cfg_enabled
    )
    speed_allowed = _feature_allowed("speed", speed_cfg_enabled) or _feature_allowed(
        "video_effects", speed_cfg_enabled
    )
    APPLY_SPEED = SPEED_MULTIPLIER not in (0.0, 1.0) and speed_cfg_enabled and speed_allowed
    if not APPLY_EFFECTS:
        FRAME_RATE = None

    ADD_SUBTITLES = subtitles_cfg_enabled and _feature_allowed(
        "subtitles", subtitles_cfg_enabled
    )
    ADD_MUSIC = music_cfg_enabled and _feature_allowed("music", music_cfg_enabled)
    APPLY_AUDIO_ENHANCER = audio_enhancer_cfg_enabled and _feature_allowed(
        "enhance_audio", audio_enhancer_cfg_enabled
    )
    ADD_WATERMARK = watermark_cfg_enabled and _feature_allowed(
        "watermark", watermark_cfg_enabled
    )
    ADD_TITLE = title_cfg_enabled and _feature_allowed("title", title_cfg_enabled)
    USE_BLURRED_BACKGROUND = bool(bgcfg.get("use_blurred_background", False))
    USE_TRANSPARENT_WINDOW = (
        bool(bgcfg.get("use_transparent_window", True)) and not USE_BLURRED_BACKGROUND
    )
    if USE_BLURRED_BACKGROUND and bgcfg.get("use_transparent_window", True):
        print("ℹ️ Прозрачное окно отключено: выбран режим размытого фона.")
    FIT_MODE = bgcfg.get("fit_mode", "contain")
    DYNAMIC_SHORTS = (
        dynamic_cfg_enabled
        and _feature_allowed("dynamic_shorts", dynamic_cfg_enabled)
        and not USE_TRANSPARENT_WINDOW
    )

    RESIZE_VALUE = vecfg.get("resize", None)

    base_dir = Path(channel_dir) if channel_dir is not None else Path(video_path).resolve().parents[1]
    assets_folder = base_dir / "assets"
    background_folder = assets_folder / "backgrounds"
    music_folder = assets_folder / "musics"
    fonts_folder = assets_folder / "fonts"

    required_folders = [
        (music_folder, "musics"),
        (fonts_folder, "fonts"),
    ]
    if not USE_BLURRED_BACKGROUND:
        required_folders.insert(0, (background_folder, "backgrounds"))

    for f, name in required_folders:
        if not f.exists():
            raise FileNotFoundError(f"❌ Не найдена папка {name}: {f}")

    fonts_list = list(fonts_folder.glob("*.*"))
    if not fonts_list:
        raise FileNotFoundError(f"❌ В папке шрифтов ничего нет: {fonts_folder}")
    font_file = random.choice(fonts_list)
    print(f"🎲 Случайный шрифт: {font_file.name}")

    background_candidates = list(background_folder.glob("*.*"))
    if not background_candidates and not USE_BLURRED_BACKGROUND:
        raise FileNotFoundError(
            f"❌ В папке фонов ничего нет, а use_blurred_background=false: {background_folder}"
        )

    music_candidates = list(music_folder.glob("*.*"))
    if not music_candidates and ADD_MUSIC:
        raise FileNotFoundError(f"❌ В папке музыки ничего нет: {music_folder}")

    try:
        video = VideoFileClip(str(video_path))
    except OSError as exc:
        print(
            "❌ Не удалось открыть исходное видео:",
            video_path,
            "\n   Причина:",
            exc,
        )
        print("   Пропускаем обработку этого файла.")
        return

    improve_transcript_quality = bool(scfg.get("improve_transcript_quality")) and ADD_SUBTITLES
    subtitle_mode = str(scfg.get("subtitle_mode", "normal") or "normal").strip().lower()
    if subtitle_mode not in {"normal", "meta_ad"}:
        print(f"⚠️ Неизвестный subtitles.subtitle_mode='{subtitle_mode}', используем normal.")
        subtitle_mode = "normal"

    model = None
    subtitles_model_name = None
    if ADD_SUBTITLES:
        if improve_transcript_quality:
            subtitles_model_name = (
                scfg.get("enhanced_whisper_model")
                or scfg.get("whisper_model")
                or "large-v3"
            )
            print(
                "🧠 Режим улучшенных финальных субтитров активирован — используется модель "
                f"Whisper '{subtitles_model_name}'"
            )
        else:
            subtitles_model_name = select_whisper_model(scfg)
            print(
                "🧠 Распознавание речи для субтитров клипов через Whisper "
                f"('{subtitles_model_name}')..."
            )

        model = whisper.load_model(subtitles_model_name)
        if SUBTITLES_LANGUAGE:
            print(f"🌐 Целевой язык субтитров: {SUBTITLES_LANGUAGE}")

    def _sanitize_transcript(result_dict):
        if not isinstance(result_dict, dict):
            return result_dict

        segments = result_dict.get("segments")
        if not isinstance(segments, list):
            return result_dict

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_text = segment.get("text", "")
            segment["text"] = sanitize_subtitle_text(segment_text)

            words = segment.get("words")
            if isinstance(words, list):
                cleaned_words = []
                for word in words:
                    if not isinstance(word, dict):
                        continue
                    original_word = str(word.get("word", ""))
                    cleaned_word = sanitize_subtitle_text(original_word).replace(" ", "")
                    if not cleaned_word:
                        continue
                    cleaned_entry = dict(word)
                    cleaned_entry["word"] = cleaned_word
                    cleaned_words.append(cleaned_entry)
                if cleaned_words:
                    segment["words"] = cleaned_words
                else:
                    segment.pop("words", None)

        cleaned_segments = [s.get("text", "").strip() for s in segments if isinstance(s, dict)]
        result_dict["text"] = " ".join(filter(None, cleaned_segments)).strip()
        return result_dict

    if isinstance(moments, str):
        try:
            moments = json.loads(moments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "❌ Не удалось распарсить моменты: ожидается JSON-объект."
            ) from exc

    if isinstance(moments, list):
        iterable_moments = ((f"moment_{idx}", data) for idx, data in enumerate(moments, start=1))
    elif isinstance(moments, dict):
        iterable_moments = moments.items()
    else:
        raise TypeError(
            "❌ Поле moments должно быть dict, list или JSON-строкой с этими структурами."
        )

    def _normalize_segments(segments_value):
        if segments_value is None:
            return []
        if isinstance(segments_value, dict):
            segment_items = list(segments_value.items())
        elif isinstance(segments_value, list):
            segment_items = list(enumerate(segments_value, start=1))
        else:
            return []

        indexed_items = []
        for idx, (segment_key, segment_val) in enumerate(segment_items, start=1):
            order = None
            if isinstance(segment_key, str) and segment_key.startswith("segment_"):
                suffix = segment_key.split("_", 1)[-1]
                if suffix.isdigit():
                    order = int(suffix)
            if order is None:
                order = idx
            indexed_items.append((order, segment_val))

        indexed_items.sort(key=lambda item: item[0])
        normalized = []
        for _, segment_val in indexed_items:
            if not isinstance(segment_val, dict):
                continue
            seg_start = segment_val.get("start")
            seg_end = segment_val.get("end")
            if seg_start is None or seg_end is None:
                continue
            normalized.append((seg_start, seg_end))
        return normalized

    for i, (key, val) in enumerate(iterable_moments, start=1):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                print(
                    f"⚠️ Момент {key} имеет строковое значение, которое не похоже на JSON. Пропускаем."
                )
                continue

        if not isinstance(val, dict):
            print(f"⚠️ Момент {key} имеет неподдерживаемый тип {type(val).__name__}. Пропускаем.")
            continue

        title_text = val.get("title", "")
        segments = _normalize_segments(val.get("segments"))
        if segments:
            clips = [video.subclip(seg_start, seg_end) for seg_start, seg_end in segments]
            clip = concatenate_videoclips(clips, method="compose")
        else:
            start, end = val.get("start"), val.get("end")
            if start is None or end is None:
                continue
            clip = video.subclip(start, end)

        if clip.audio is not None and not getattr(clip.audio, "fps", None):
            clip = clip.set_audio(clip.audio.set_fps(44100))

        background_path = (
            None if USE_BLURRED_BACKGROUND else random.choice(background_candidates)
        )
        music_path = random.choice(music_candidates) if music_candidates else None

        print(f"\n🎞️ Клип {i}")
        if USE_BLURRED_BACKGROUND:
            print("🌀 Фон: используется размытая версия исходного видео")
        elif background_path is not None:
            print(f"🖼 Фон: {background_path.name}")
        if ADD_MUSIC:
            print(f"🎵 Музыка: {music_path.name}")
        else:
            if not music_cfg_enabled:
                print("🔇 Музыка отключена (music.enabled=False)")
            elif not music_pipeline_enabled:
                print("⏭️ Музыка пропущена (music не в pipeline)")
            else:
                print("🔇 Музыка отключена.")

        # --- Ускорение до эффектов и субтитров ---
        if APPLY_SPEED:
            clip = clip.fx(vfx.speedx, SPEED_MULTIPLIER)
            print(f"⏩ Ускоряем клип: x{SPEED_MULTIPLIER:.3f}")
        else:
            if SPEED_MULTIPLIER in (0.0, 1.0):
                print("⏩ Ускорение не требуется (speed=1.0).")
            elif not speed_cfg_enabled:
                print("⏩ Ускорение отключено (video_effects.enabled=False).")
            elif not speed_pipeline_enabled:
                print("⏭️ Ускорение пропущено (speed не в pipeline).")
            else:
                print("⏩ Ускорение отключено.")

        # --- Определение области под видео ---
        canvas_width = float(base_canvas_width)
        canvas_height = float(base_canvas_height)
        box_x, box_y = 0.0, 0.0
        box_w, box_h = canvas_width, canvas_height

        try:
            if USE_TRANSPARENT_WINDOW:
                x, y, w, h, bg_w, bg_h = get_transparent_box(str(background_path))
                box_x, box_y, box_w, box_h = float(x), float(y), float(w), float(h)
                canvas_width, canvas_height = float(bg_w), float(bg_h)
                print("✅ Используется авто-детект прозрачного окна.")
            else:
                bg_w = int(bgcfg.get("width", base_canvas_width))
                bg_h = int(bgcfg.get("height", base_canvas_height))
                canvas_width, canvas_height = float(bg_w), float(bg_h)
                video_box_w = int(bgcfg.get("video_width", bg_w))
                video_box_h = int(bgcfg.get("video_height", bg_h))
                if video_box_w <= 0 or video_box_h <= 0:
                    video_box_w, video_box_h = bg_w, bg_h

                default_box_x = (bg_w - video_box_w) / 2
                default_box_y = (bg_h - video_box_h) / 2
                box_x = float(bgcfg.get("video_x", default_box_x))
                box_y = float(bgcfg.get("video_y", default_box_y))
                box_w, box_h = float(video_box_w), float(video_box_h)
                print(
                    "🖼 Используется фиксированный фон "
                    f"{bg_w}x{bg_h}. Видеообласть {box_w:.0f}x{box_h:.0f}"
                    f" (x={box_x:.1f}, y={box_y:.1f})."
                )
        except Exception as e:
            print(f"⚠️ Ошибка при определении области ({e}), используем настройки cfg.")
            canvas_width, canvas_height = float(base_canvas_width), float(base_canvas_height)
            box_x, box_y, box_w, box_h = 0.0, 0.0, canvas_width, canvas_height

        viewport_x, viewport_y = box_x, box_y
        viewport_w, viewport_h = box_w, box_h
        content_x, content_y = viewport_x, viewport_y
        content_w, content_h = viewport_w, viewport_h
        canvas_w_int = max(1, int(round(canvas_width)))
        canvas_h_int = max(1, int(round(canvas_height)))

        tmp_audio = None
        result = {"segments": []}
        if ADD_SUBTITLES:
            # --- Подготовка аудио для субтитров ---
            tmp_audio = os.path.join(tempfile.gettempdir(), f"segment_{i}.wav")
            clip.audio.write_audiofile(tmp_audio, fps=44100, verbose=False, logger=None)
            transcribe_kwargs = {"word_timestamps": True}
            if SUBTITLES_LANGUAGE:
                transcribe_kwargs["language"] = SUBTITLES_LANGUAGE

            result = model.transcribe(tmp_audio, **transcribe_kwargs)

            if improve_transcript_quality:
                if not OPENAI_API_KEY:
                    print(
                        "⚠️ Улучшение качества финальных субтитров включено, но OPENAI_API_KEY недоступен. Используется сырой текст."
                    )
                else:
                    segments = result.get("segments") if isinstance(result, dict) else None
                    if isinstance(segments, list):
                        for segment in segments:
                            if not isinstance(segment, dict):
                                continue
                            raw_text = str(segment.get("text", ""))
                            if not raw_text:
                                continue
                            try:
                                cleaned_text = clean_and_correct_text(
                                    raw_text,
                                    api_key=OPENAI_API_KEY,
                                    model=scfg.get("enhancer_model", "gpt-5-nano"),
                                    prompt_template=scfg.get("prompt"),
                                    meta_ad_prompt_template=scfg.get("meta_ad_prompt"),
                                    subtitle_mode=subtitle_mode,
                                    language=SUBTITLES_LANGUAGE,
                                    log_preview=False,
                                )
                            except Exception as exc:
                                print(
                                    "⚠️ Не удалось очистить текст сегмента через GPT-5-nano:",
                                    f" {exc}"
                                )
                                continue
                            segment["text"] = cleaned_text
                            rebuild_words_from_clean_text(segment, cleaned_text)

                    result = _sanitize_transcript(result)

        # --- Масштабируем клип ---
        if USE_TRANSPARENT_WINDOW:
            # Старое поведение — вставляем видео в прозрачное окно
            clip_resized = clip.resize((viewport_w, viewport_h)).set_position(
                (viewport_x, viewport_y)
            )
            content_x, content_y = viewport_x, viewport_y
            content_w, content_h = viewport_w, viewport_h
        else:
            if DYNAMIC_SHORTS:
                video_box_w = viewport_w
                video_box_h = viewport_h

                video_aspect = clip.w / clip.h
                target_aspect = (
                    video_box_w / video_box_h if video_box_h not in (0, None) else video_aspect
                )

                if FIT_MODE == "cover":
                    if video_aspect > target_aspect:
                        scale_height = video_box_h
                        scale_width = clip.w * (scale_height / clip.h)
                    else:
                        scale_width = video_box_w
                        scale_height = clip.h * (scale_width / clip.w)
                else:  # contain (default)
                    if video_aspect > target_aspect:
                        scale_width = video_box_w
                        scale_height = clip.h * (scale_width / clip.w)
                    else:
                        scale_height = video_box_h
                        scale_width = clip.w * (scale_height / clip.h)

                scale_width = float(scale_width)
                scale_height = max(1.0, float(scale_height))

                viewport_width = min(scale_width, video_box_w, canvas_width)
                if viewport_width <= 0:
                    viewport_width = min(scale_width, canvas_width)
                viewport_width = max(1.0, viewport_width)

                dynamic_config = dict(dynamic_cfg)
                dynamic_config.update({
                    "target_width": int(round(viewport_w)),
                    "target_height": int(round(viewport_h)),
                    "video_width": int(round(viewport_w)),
                    "video_height": int(round(viewport_h)),
                })

                # dynamic_config = dict(dynamic_cfg)
                # dynamic_config["target_width"] = int(round(viewport_width))
                # dynamic_config["target_height"] = int(round(scale_height))

                try:
                    dynamic_result = build_dynamic_short_clip(
                        clip,
                        dynamic_config,
                    )
                except ValueError as exc:
                    print(
                        "⚠️ Не удалось запустить динамический кроп "
                        f"({exc}). Переходим в статичный режим."
                    )
                    dynamic_result = None

                if dynamic_result:
                    viewport_cx = viewport_x + viewport_w / 2.0
                    viewport_cy = viewport_y + viewport_h / 2.0
                    pos_x = _aligned_position(viewport_cx, dynamic_result.clip.w)
                    pos_y = _aligned_position(viewport_cy, dynamic_result.clip.h)
                    clip_positioned = dynamic_result.clip.set_position((pos_x, pos_y))
                    clip_resized = clip_positioned
                    content_x, content_y = pos_x, pos_y
                    content_w, content_h = dynamic_result.clip.w, dynamic_result.clip.h
                    print(
                        "🎯 Динамический кроп активирован: "
                        f"точек анализа={dynamic_result.analysis_points}, "
                        f"масштабированная ширина={dynamic_result.scaled_width:.1f}px, "
                        f"видимая область={content_w:.0f}x{content_h:.0f}"
                    )
                    if dynamic_result.used_face_track:
                        print("   • Используется слежение за лицами.")
                    else:
                        print("   • Запущен режим Ken Burns (fallback).")
                    print(
                        f"   • Склейки: {dynamic_result.scene_cuts}, "
                        f"смены героя: {dynamic_result.face_switches}, "
                        f"зум: {dynamic_result.zoom_range[0]}–{dynamic_result.zoom_range[1]}"
                    )
                else:
                    # fallback to static behaviour below
                    DYNAMIC_SHORTS = False

            if not DYNAMIC_SHORTS:
                # Новый режим — без прозрачного окна, центрирование
                video_aspect = clip.w / clip.h
                target_aspect = viewport_w / viewport_h

                if FIT_MODE == "cover":
                    if video_aspect > target_aspect:
                        clip_scaled = clip.resize(height=viewport_h)
                    else:
                        clip_scaled = clip.resize(width=viewport_w)
                else:
                    if video_aspect > target_aspect:
                        clip_scaled = clip.resize(width=viewport_w)
                    else:
                        clip_scaled = clip.resize(height=viewport_h)

                pos_x = viewport_x + (viewport_w - clip_scaled.w) / 2.0
                pos_y = viewport_y + (viewport_h - clip_scaled.h) / 2.0
                clip_resized = clip_scaled.set_position((pos_x, pos_y))
                print(
                    "📐 Видео отцентрировано: "
                    f"x={pos_x:.1f}, y={pos_y:.1f}, w={clip_scaled.w:.1f}, "
                    f"h={clip_scaled.h:.1f}, mode={FIT_MODE}"
                )

                content_x, content_y = pos_x, pos_y
                content_w, content_h = clip_scaled.w, clip_scaled.h

        visible_box_x, visible_box_w = _visible_region_bounds(
            content_x, content_w, canvas_width
        )
        visible_box_y, visible_box_h = _visible_region_bounds(
            content_y, content_h, canvas_height
        )

        # --- Эффекты ---
        def _apply_video_effects(base_clip):
            processed = base_clip
            if vecfg.get("mirror", False):
                processed = processed.fx(vfx.mirror_x)
            if vecfg.get("color_saturation"):
                processed = processed.fx(vfx.colorx, vecfg["color_saturation"])
            if vecfg.get("gamma"):
                processed = processed.fx(vfx.gamma_corr, vecfg["gamma"])
            if vecfg.get("contrast"):
                processed = processed.fx(
                    vfx.lum_contrast, contrast=vecfg["contrast"]
                )
            return processed

        clip_for_background = clip

        if APPLY_EFFECTS:
            clip_resized = _apply_video_effects(clip_resized)
            clip_for_background = _apply_video_effects(clip_for_background)
            print("🎨 Эффекты применены.")
        else:
            if not effects_cfg_enabled:
                print("⚙️ Монтаж видео отключён (video_effects.enabled=False).")
            elif not effects_pipeline_enabled:
                print("⏭️ Монтаж видео пропущен (video_effects не в pipeline).")
            else:
                print("⚙️ Монтаж видео отключён.")

        # --- Фон ---
        if USE_BLURRED_BACKGROUND:
            blur_strength = bgcfg.get("blur_strength", 85)
            background_scale = bgcfg.get("background_scale", 1.35)
            background = _build_blurred_background(
                clip_for_background,
                canvas_w_int,
                canvas_h_int,
                blur_strength=blur_strength,
                scale=background_scale,
            )
            print(
                "🌫 Размытый фон подготовлен: "
                f"blur_strength={blur_strength}, scale={background_scale}"
            )
        else:
            background = (
                ImageClip(str(background_path))
                .set_duration(clip.duration)
                .resize((canvas_w_int, canvas_h_int))
            )

        # === ВОДЯНОЙ ЗНАК ===
        watermark_clip = None
        if ADD_WATERMARK:
            try:
                wm_text = wcfg.get("text", "@MyChannel")
                wm_font = str(font_file)
                wm_color = wcfg.get("color", "#FFFFFF")
                wm_fontsize = wcfg.get("font_size", 38)
                wm_opacity = wcfg.get("opacity", 0.25)
                wm_position_cfg = wcfg.get("position", ("left", "top"))
                if isinstance(wm_position_cfg, (list, tuple)):
                    wm_position = tuple(wm_position_cfg[:2])
                else:
                    wm_position = ("left", "top")
                wm_padding_x = wcfg.get("padding_x", 30)
                wm_padding_y = wcfg.get("padding_y", 30)
                wm_shadow_color = wcfg.get("shadow_color", "#000000")
                wm_shadow_opacity = wcfg.get("shadow_opacity", 0.3)
                wm_shadow_offset = wcfg.get("shadow_offset", 2)

                # Привязываем вотермарк к видимой области клипа, а не к фону
                video_x, video_y = visible_box_x, visible_box_y
                video_w = max(1, int(round(visible_box_w)))
                video_h = max(1, int(round(visible_box_h)))

                watermark_clip = make_watermark_clip(
                    text=wm_text,
                    font_path=wm_font,
                    font_size=wm_fontsize,
                    color=wm_color,
                    opacity=wm_opacity,
                    position=wm_position,
                    padding_x=wm_padding_x,
                    padding_y=wm_padding_y,
                    video_w=video_w,
                    video_h=video_h,
                    shadow_color=wm_shadow_color,
                    shadow_opacity=wm_shadow_opacity,
                    shadow_offset=wm_shadow_offset,
                    duration=clip.duration,
                    offset_x=video_x,
                    offset_y=video_y,
                )

                print(f"💧 Водяной знак добавлен и привязан к видео ({wm_position})")
            except Exception as e:
                print(f"⚠️ Ошибка при добавлении вотермарка: {e}")
                watermark_clip = None  # ← не даём переменной исчезнуть

        else:
            if not watermark_cfg_enabled:
                print("🈚 Водяной знак отключён (watermark.enabled=False)")
            elif not watermark_pipeline_enabled:
                print("⏭️ Водяной знак пропущен (watermark не в pipeline)")
            else:
                print("🈚 Водяной знак отключён.")

        # --- Заголовок ---
        title_clips = []
        if title_text and ADD_TITLE:
            try:
                configured_width = tcfg.get("max_width", canvas_w_int - 80)
                try:
                    max_width = float(configured_width)
                except (TypeError, ValueError):
                    max_width = canvas_w_int - 80
                max_width = max(10.0, min(float(canvas_w_int), float(max_width)))

                max_height = tcfg.get("max_height")
                if max_height is None:
                    max_height = int(canvas_h_int * 0.22)

                fade_in = tcfg.get("fade_in", 0.25)
                fade_out = tcfg.get("fade_out", 0.25)
                switch_enabled = tcfg.get("switch_enabled", True)
                switch_after = tcfg.get("switch_after")
                switch_text = tcfg.get("switch_text")

                base_title_kwargs = dict(
                    font_path=font_file,
                    font_size=tcfg.get("font_size", 52),
                    color=tcfg.get("color", "#FFFFFF"),
                    stroke_color=tcfg.get("stroke_color", "#000000"),
                    stroke_width=tcfg.get("stroke_width", 4),
                    max_width=max_width,
                    video_width=canvas_w_int,
                    top_padding=tcfg.get("top_padding", 60),
                    max_height=max_height,
                    max_words_per_line=tcfg.get("max_words_per_line", 3),
                    min_font_size=tcfg.get("min_font_size"),
                    emoji_font_size=tcfg.get("emoji_font_size"),
                    bg_color=tcfg.get("bg_color"),
                    bg_opacity=tcfg.get("bg_opacity"),
                    shadow_color=tcfg.get("shadow_color"),
                    shadow_opacity=tcfg.get("shadow_opacity"),
                    shadow_offset_x=tcfg.get("shadow_offset_x"),
                    shadow_offset_y=tcfg.get("shadow_offset_y"),
                    shadow_blur=tcfg.get("shadow_blur"),
                    fade_in=fade_in,
                    fade_out=fade_out,
                    align=tcfg.get("align", "center"),
                    emoji_font_path=tcfg.get("emoji_font_path", "NotoColorEmoji.ttf"),
                )

                if switch_enabled and switch_after is not None and switch_text:
                    try:
                        switch_point = max(0.0, float(switch_after))
                    except (TypeError, ValueError):
                        switch_point = 0.0
                    switch_point = min(switch_point, clip.duration)
                    first_duration = max(0.0, switch_point)
                    second_duration = max(0.0, clip.duration - switch_point)

                    if first_duration > 0:
                        title_clips.append(
                            make_title_clip(
                                text=title_text,
                                start=0.0,
                                duration=first_duration,
                                **base_title_kwargs,
                            )
                        )
                    if second_duration > 0:
                        title_clips.append(
                            make_title_clip(
                                text=str(switch_text),
                                start=switch_point,
                                duration=second_duration,
                                **base_title_kwargs,
                            )
                        )
                else:
                    title_clips.append(
                        make_title_clip(
                            text=title_text,
                            start=0.0,
                            duration=clip.duration,
                            **base_title_kwargs,
                        )
                    )

                print("🧷 Добавлен заголовок поверх видео.")
            except Exception as e:
                print(f"⚠️ Ошибка при создании заголовка: {e}")
        elif title_text:
            if not title_cfg_enabled:
                print("🈚 Заголовок отключён (title.enabled=False)")
            elif not title_pipeline_enabled:
                print("⏭️ Заголовок пропущен (title не в pipeline)")
            else:
                print("🈚 Заголовок отключён.")

        # --- Субтитры ---
        subtitle_clips = []
        if ADD_SUBTITLES:
            print(f"💬 Добавляю субтитры (WORDS_PER_CHUNK={WORDS_PER_CHUNK})...")
            for segment in result.get("segments", []):
                chunks = split_segment_by_words(segment, max_words=WORDS_PER_CHUNK)
                for chunk in chunks:
                    text = sanitize_subtitle_text(chunk["text"]).strip()
                    if not text:
                        continue
                    configured_width = scfg.get("max_width", visible_box_w - 40)
                    try:
                        max_width = float(configured_width)
                    except (TypeError, ValueError):
                        max_width = visible_box_w - 40
                    max_width = max(10.0, min(float(visible_box_w), float(max_width)))
                    highlight_cfg = scfg.get("highlight_style") or {}
                    try:
                        txt_clip = make_subtitle_clip(
                            text=text,
                            start=chunk["start"],
                            end=chunk["end"],
                            font_path=font_file,
                            font_size=scfg["font_size"],
                            highlight_font_size=highlight_cfg.get("font_size"),
                            emoji_font_size=scfg.get("emoji_font_size"),
                            color=scfg["color"],
                            stroke_color=scfg["stroke_color"],
                            stroke_width=scfg["stroke_width"],
                            max_width=max_width,
                            box_x=visible_box_x,
                            box_y=visible_box_y,
                            box_w=visible_box_w,
                            box_h=visible_box_h,
                            bottom_padding=scfg.get("bottom_padding"),
                            fade_in=scfg.get("fade_in", 0.25),
                            fade_out=scfg.get("fade_out", 0.25),
                            position=scfg.get("position"),
                            align=scfg.get("align"),
                            vertical_align=scfg.get("vertical_align"),
                            padding_x=scfg.get("padding_x"),
                            padding_y=scfg.get("padding_y"),
                            shadow_color=scfg.get("shadow_color"),
                            shadow_opacity=scfg.get("shadow_opacity"),
                            shadow_offset_x=scfg.get("shadow_offset_x"),
                            shadow_offset_y=scfg.get("shadow_offset_y"),
                            shadow_blur=scfg.get("shadow_blur"),
                            glow_color=scfg.get("glow_color"),
                            glow_radius=scfg.get("glow_radius"),
                            bg_color=scfg.get("bg_color"),
                            bg_opacity=scfg.get("bg_opacity"),
                            emoji_font_path=scfg.get("emoji_font_path", "NotoColorEmoji.ttf"),
                            highlight_style=highlight_cfg,
                        )
                        subtitle_clips.append(txt_clip)
                    except Exception as e:
                        print(f"⚠️ Ошибка при создании субтитров: {e}")
        else:
            if not subtitles_cfg_enabled:
                print("🈚 Субтитры отключены (subtitles.enabled=False).")
            elif not subtitles_pipeline_enabled:
                print("⏭️ Субтитры пропущены (subtitles не в pipeline).")
            else:
                print("🈚 Субтитры отключены.")

        # --- Слои ---
        layers = [background, clip_resized]
        if title_clips:
            layers.extend(title_clips)
        if subtitle_clips:
            layers.extend(subtitle_clips)
        if watermark_clip:
            layers.append(watermark_clip)

        final = CompositeVideoClip(layers, size=(canvas_w_int, canvas_h_int))

        # --- Аудио ---
        dialog_audio = clip.audio
        if dialog_audio is not None and APPLY_AUDIO_ENHANCER:
            try:
                print("🎚 Улучшаем оригинальную аудиодорожку (dynamic range compression)...")
                dialog_audio = enhance_dialogue_audio(dialog_audio, audio_enhancer_cfg)
            except Exception as exc:
                print(f"⚠️ Не удалось применить компрессию к аудио: {exc}")
            else:
                print("✅ Громкость диалогов выровнена.")

        if dialog_audio is not None:
            dialog_audio = dialog_audio.volumex(1.0)

        if ADD_MUSIC:
            music_volume = mcfg.get("volume", 0.06)
            music = AudioFileClip(str(music_path)).volumex(music_volume)

            target_duration = float(clip.duration or music.duration or 0.0)
            padding = 1e-3
            adjusted_duration = max(0.0, target_duration - padding)

            if music.duration and music.duration < adjusted_duration:
                music = music.fx(afx.audio_loop, duration=adjusted_duration)
            elif music.duration:
                music = music.subclip(0, min(music.duration, adjusted_duration))

            music = music.set_duration(target_duration)

            sources = [music]
            if dialog_audio is not None:
                sources.insert(0, dialog_audio)
            final_audio = CompositeAudioClip(sources)
        else:
            final_audio = dialog_audio

        if final_audio is None:
            final_audio = clip.audio
        final = final.set_audio(final_audio)

        # --- Частота кадров ---
        if FRAME_RATE:
            fps = random.uniform(29.7, 30.3) if FRAME_RATE == "random" else float(FRAME_RATE)
        else:
            fps = 30.0
        print(f"🎞 FPS: {fps:.2f}")

        out_file = output_folder / f"{safe_filename(val['title'].replace(' ', '_'))}.mp4"
        print(f"💾 Рендерю: {out_file}")
        final.write_videofile(str(out_file), codec="libx264", audio_codec="aac", fps=fps)

        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    print("\n✅ Все клипы успешно созданы!")
