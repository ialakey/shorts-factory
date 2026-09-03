import json
import os

import yaml
from moviepy.editor import VideoFileClip

from analysis.gpt_analyzer import analyze_moment
from config import (
    CHANNELS_DIR,
    OPENAI_API_KEY,
    OUTPUT_TEST_DATA_DIR,
    TEST_DATA_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from ingestion.autodownload import auto_download_titles
from ingestion.parser import find_input_videos
from ingestion.transcriber import transcribe_video
from publishing.spoof_metadata import spoof_metadata
from publishing.telegram_notifier import (
    TelegramNotifier,
    format_clip_title,
    normalize_anime_name,
)
from rendering.video_editor import make_clips


def _ensure_debug_folders():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _dump_debug_result(video_path, suffix, data):
    output_name = f"{video_path.stem}_{suffix}"
    target = OUTPUT_TEST_DATA_DIR / output_name
    if isinstance(data, (dict, list)):
        target = target.with_suffix(".json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        target = target.with_suffix(".txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write(str(data))
    print(f"🧪 Debug: сохранён результат {suffix} → {target.relative_to(OUTPUT_TEST_DATA_DIR.parent)}")


POST_CLIP_STAGES = {
    "make_clips",
    "dynamic_shorts",
    "watermark",
    "enhance_audio",
    "video_effects",
    "music",
    "subtitles",
    "spoof_metadata",
    "telegram_notify",
}


def _build_full_length_segment(video_path):
    try:
        with VideoFileClip(str(video_path)) as clip:
            duration = float(clip.duration or 0)
    except Exception as exc:
        print(
            "⚠️ Не удалось определить длительность видео для debug-момента:",
            exc,
        )
        duration = 0

    if duration <= 0:
        end_time = 1.0
    else:
        end_time = max(duration, 1.0)

    return [
        {
            "start": 0.0,
            "end": end_time,
            "title": video_path.stem,
        }
    ]


def process_channel(channel_name: str):
    """Основная логика обработки одного канала:
    управляется через config.yaml → секцию pipeline.
    """

    base_path = CHANNELS_DIR / channel_name
    print(f"🔍 Ищу конфиг по пути: {base_path}")

    input_dir = base_path / "input_videos"
    output_dir = base_path / "output_clips"
    config_file = base_path / "config.yaml"

    if not config_file.exists():
        print(f"❌ Нет config.yaml для канала {channel_name}, пропускаем...")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        channel_cfg = yaml.safe_load(f)

    pipeline = channel_cfg.get(
        "pipeline",
        [
            "transcribe_video",
            "analyze_moment",
            "make_clips",
            "dynamic_shorts",
            "watermark",
            "enhance_audio",
            "video_effects",
            "music",
            "subtitles",
            "spoof_metadata",
            "telegram_notify",
        ],
    )
    pipeline_set = set(pipeline)

    debug_mode = bool(channel_cfg.get("debug", False))

    if debug_mode:
        print("🧪 Debug режим активен — используется test_data/output_test_data")
        _ensure_debug_folders()
        input_dir = TEST_DATA_DIR
        output_dir = OUTPUT_TEST_DATA_DIR
    else:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)

    if debug_mode:
        print("⏩ Debug режим: загрузка с Kodik отключена")
    elif {"kodik_download", "autodownload"} & pipeline_set:
        if "autodownload" in pipeline_set and "kodik_download" not in pipeline_set:
            print("ℹ️ Шаг autodownload переименован в kodik_download — обновите pipeline")

        print("▶️ Загрузка исходных видео через Kodik ...")
        try:
            auto_download_titles(
                channel_cfg.get("kodik_download")
                or channel_cfg.get("autodownload"),
                input_dir,
            )
        except Exception as exc:  # noqa: BLE001 - скачивание не должно ронять пайплайн
            print(f"⚠️ Загрузка с Kodik не выполнена: {exc}")
            print("   Продолжаем с уже имеющимися видео в input_videos.")
    else:
        print("⏩ Пропускаем загрузку с Kodik (kodik_download не в pipeline)")

    videos = find_input_videos(input_dir)

    if not videos:
        print(f"❌ Нет видео в input_videos для канала {channel_name}")
        return

    print(f"🎬 Найдено {len(videos)} видео для {channel_name}")

    # --- Основной цикл по видео ---
    requires_clips = bool(pipeline_set & POST_CLIP_STAGES)
    dependent_post_clip_steps = (pipeline_set & POST_CLIP_STAGES) - {"make_clips"}

    telegram_notifier = None
    if "telegram_notify" in pipeline_set:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        else:
            print(
                "⚠️ Telegram notify включён в pipeline, но TELEGRAM_BOT_TOKEN "
                "или TELEGRAM_CHAT_ID не заданы. Отправка пропущена."
            )

    for video_path in videos:
        print(f"\n▶️ Обработка {video_path.name} ...")

        transcript = None
        moments = None
        existing_clips = set(output_dir.glob("*.mp4"))

        # === Транскрибация ===
        if "transcribe_video" in pipeline:
            print("▶️ Транскрибация ...")
            transcript = transcribe_video(
                video_path,
                cfg=channel_cfg,
            )
            if debug_mode and transcript:
                transcript_text = transcript.get("text") or ""
                if not transcript_text and transcript.get("segments"):
                    transcript_text = "\n".join(
                        s.get("text", "") for s in transcript.get("segments", [])
                    )
                _dump_debug_result(video_path, "transcript", transcript_text.strip())
        else:
            print("⏩ Пропускаем транскрибацию (transcribe_video не в pipeline)")

        # === Анализ ===
        if "analyze_moment" in pipeline:
            if transcript is None:
                print("⚠️ Нет транскрипта, невозможно проанализировать.")
            else:
                print("▶️ Анализируем моменты ...")
                moments = analyze_moment(
                    transcript,
                    video_path,
                    channel_cfg,
                    OPENAI_API_KEY,
                    output_dir=output_dir,
                )
                if debug_mode and moments:
                    _dump_debug_result(video_path, "moments", moments)
        else:
            print("⏩ Пропускаем анализ (analyze_moment не в pipeline)")

        # === Создание клипов ===
        should_run_make_clips = "make_clips" in pipeline
        fallback_make_clips = False
        fallback_reasons = set()

        if dependent_post_clip_steps and "make_clips" not in pipeline_set:
            should_run_make_clips = True
            fallback_make_clips = True
            fallback_reasons.update(dependent_post_clip_steps)

        if not should_run_make_clips and debug_mode and requires_clips:
            should_run_make_clips = True
            fallback_make_clips = True

        if fallback_make_clips:
            if fallback_reasons:
                steps_list = ", ".join(sorted(fallback_reasons))
                print(
                    "ℹ️ Автоматически запускаем make_clips, "
                    f"чтобы отработали этапы: {steps_list}."
                )
            elif debug_mode:
                print(
                    "ℹ️ Debug: make_clips запущен автоматически для теста этапов"
                )

            if moments is None:
                print(
                    "ℹ️ Сегменты не заданы — используем весь ролик для наложения субтитров."
                )
                moments = _build_full_length_segment(video_path)

        if should_run_make_clips:
            if moments is None:
                print("⚠️ Нет данных о моментах, невозможно сделать клипы.")
            else:
                if fallback_make_clips and not fallback_reasons:
                    print(
                        "🧪 Debug: make_clips запущен автоматически для теста этапов",
                    )
                elif not fallback_make_clips:
                    print("▶️ Создание клипов ...")
                make_clips(
                    video_path,
                    moments,
                    output_dir,
                    channel_cfg,
                    pipeline=pipeline,
                    channel_dir=base_path,
                )
        else:
            print("⏩ Пропускаем создание клипов (make_clips не в pipeline)")

        # === Подмена метаданных ===
        if "spoof_metadata" in pipeline:
            print("🧹 Подмена метаданных у сгенерированных клипов ...")
            # Фиксируем список заранее: во время итерации мы создаём временные
            # *_spoofed.mp4, которые иначе могут попасть в тот же обход.
            for clip in sorted(output_dir.glob("*.mp4")):
                spoofed_path = clip.with_name(f"{clip.stem}_spoofed.mp4")
                # Оригинал удаляем только после успешной пересборки контейнера,
                # иначе при падении ffmpeg клип был бы потерян безвозвратно.
                if spoof_metadata(clip, spoofed_path):
                    clip.unlink(missing_ok=True)
                    spoofed_path.replace(clip)
                else:
                    print(f"⚠️ Оставляем оригинальные метаданные: {clip.name}")
            print("✅ Метаданные обновлены для всех клипов!\n")
        else:
            print("⏩ Пропускаем spoof_metadata")

        current_clips = set(output_dir.glob("*.mp4"))
        new_clips = sorted(current_clips - existing_clips, key=lambda p: p.name)

        if "telegram_notify" in pipeline_set:
            if not telegram_notifier:
                print("⚠️ Telegram отключён или не настроен, отправка пропущена.")
            elif not new_clips:
                print("⚠️ Нет новых клипов для отправки в Telegram.")
            else:
                clip_titles = []
                if isinstance(moments, dict):
                    iterable_moments = moments.values()
                else:
                    iterable_moments = moments or []
                for item in iterable_moments:
                    if not item:
                        continue
                    if isinstance(item, dict):
                        title = str(item.get("title", "")).strip()
                    else:
                        title = str(item).strip()

                    display_title = format_clip_title(title, fallback=video_path.stem)
                    if display_title:
                        clip_titles.append(display_title)
                anime_name = normalize_anime_name(video_path.stem)
                telegram_notifier.send_media_group(
                    clip_titles,
                    anime_name,
                    new_clips,
                )
        else:
            print("⏩ Пропускаем отправку в Telegram (telegram_notify не в pipeline)")

    print(f"✅ Обработка канала {channel_name} завершена!\n")
