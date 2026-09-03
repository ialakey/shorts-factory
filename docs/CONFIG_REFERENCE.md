# Справочник по конфигурации `channels/<Channel>/config.yaml`

Ниже — рабочий минимум и пояснения по наиболее важным секциям.

## Минимальный пример

```yaml
channel_name: DemoChannel
debug: false

pipeline:
  - transcribe_video
  - analyze_moment
  - make_clips
  - dynamic_shorts
  - watermark
  - enhance_audio
  - video_effects
  - music
  - subtitles
  - spoof_metadata
  - telegram_notify

video:
  width: 1080
  height: 1920
```

## Ключевые секции

### `pipeline`
Порядок и состав этапов. Поддерживаются:
- `kodik_download` (`autodownload` как legacy alias)
- `transcribe_video`
- `analyze_moment`
- `make_clips`
- `dynamic_shorts`
- `watermark`
- `enhance_audio`
- `video_effects`
- `music`
- `subtitles`
- `title`
- `spoof_metadata`
- `telegram_notify`

### `debug`
- `true`: использует `test_data/` и `output_test_data/`, Kodik-загрузка отключается.
- `false`: стандартный режим по папкам канала.

### `video`
Размер итогового полотна рендера (`width` x `height`).

### `video_effects`
Основные визуальные параметры:
- `enabled`
- `mirror`
- `color_saturation`
- `gamma`
- `contrast`
- `resize`
- `frame_rate`
- `speed`

### `dynamic_shorts`
Виртуальная камера: авто-кадрирование по лицу (работает, если отключено прозрачное окно фона).

- `profile` — базовый темперамент камеры: `static` | `operator` | `action`. Любой ключ ниже переопределяет профиль точечно.
- Детект и трекинг: `analysis_fps`, `min_face_ratio`, `min_face_confidence`, `match_tolerance`, `max_miss_time`.
- Выбор героя: `face_switch_margin`, `face_switch_hold_s` — насколько и как долго новый персонаж должен выигрывать, чтобы забрать фокус.
- Фильтрация сигнала: `face_filter` (low-pass), `max_face_step` (анти-джерк), `stabilization_strength`.
- Движение: `smoothing` (лаг по цели), `follow_stiffness`, `follow_damping`, `max_center_accel`, `max_center_speed`, `center_dead_zone` (мёртвая зона по ошибке слежения), `predictive_lead`, `human_lag`.
- Композиция: `side_bias`, `side_bias_strength`, `eye_level_lift`, `face_margin`.
- Зум: `face_coverage` (целевая высота лица, `0` — зум выключен), `max_zoom` (> 1.0 включает зум), `zoom_smoothing`, `max_zoom_speed`.
- Склейки: `scene_cut_detection`, `scene_cut_threshold` (абсолютный минимум различия кадров), `scene_cut_relative` (во сколько раз выше локального фона), `scene_cut_window`, `scene_cut_min_interval_s`.
- Fallback без лица: `fallback_recenter_s`, `ken_burns_period`, `ken_burns_pan_amplitude`, `ken_burns_tilt_amplitude`, `ken_burns_zoom_amplitude`.

> Коэффициенты сглаживания (`smoothing`, `face_filter`, `zoom_smoothing`) заданы для `analysis_fps: 8` и автоматически пересчитываются под другой шаг анализа.

### `moment_scoring`
Детерминированный отбор кандидатов до обращения к LLM.

- `enabled` — выключение вернёт поведение «решает только модель» (кандидаты не строятся и в промпт не уходят).
- `cell_s`, `step_s`, `length_steps` — разрешение сетки сигналов и шаг/варианты длительности скользящего окна.
- `hook_window_s` — окно анализа входа в клип.
- `max_candidates`, `prompt_limit`, `nms_overlap` — сколько кандидатов оставить и сколько показать модели.
- `min_speech_density`, `min_face_coverage`, `max_silence_gap_s` — пороги штрафов за пустые и провисающие окна.
- `weights` — веса сигналов `transcript` / `audio` / `face` / `scene` / `pacing` / `hook` (нормализуются к сумме 1).

### `moment_validation`
Починка и проверка того, что вернула модель.

- `min_segment_s` — минимальная длина одного segment монтажной сборки.
- `snap_tolerance_s`, `snap_to_speech`, `snap_to_cuts` — притяжка резов к границам фраз и склейкам.
- `min_speech_density`, `min_audio_energy` — жёсткие пороги «пустого» фрагмента.
- `fill_from_candidates` — добивать недостачу моментов эвристическими кандидатами вместо падения.

### `background`
Режимы размещения исходного видео на холсте:
- фон-изображение,
- размытие оригинала,
- прозрачное окно,
- режимы fit/position.

### `subtitles`
- `enabled`
- внешний вид (font size/color/stroke/shadow/bg)
- `language`
- `whisper_model`
- `improve_transcript_quality`
- `enhanced_whisper_model`
- `subtitle_mode`: `normal` или `meta_ad`

### `music`
- `enabled`
- `volume`

### `audio_enhancer`
Пост-обработка речи (компрессия/нормализация).

### `watermark`
Настройки водяного знака.

### `title`
Настройки заголовка в верхней части видео.

### `gpt`
Настройки выбора моментов:
- модель,
- количество клипов (`min_count`/`max_count`),
- длительность (`min_time`/`max_time`) — контролируется валидатором, а не только промптом,
- промпт,
- `face_analysis_interval_s` — шаг анализа лиц для GPT в секундах (по умолчанию `5.0`, чтобы не анализировать каждый кадр).

Плейсхолдеры промпта: `{transcript_text}`, `{audio}`, `{face}`, `{visual}`, `{tempo}`,
`{emotion}`, `{hooks}`, `{candidates}`, `{tone}`, `{platforms}`, `{audience_age}`,
`{min_count}`, `{max_count}`, `{min_time}`, `{max_time}`. Неиспользуемые плейсхолдеры
можно опускать; `{candidates}` — отранжированные окна из `moment_scoring`.

### `kodik_download`
Параметры автозагрузки тайтлов через Kodik.

## Переменные окружения

Обязательная:
- `OPENAI_API_KEY`

Опциональные:
- `KODIK_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Практические рекомендации

1. Сначала запускайте на коротком тестовом видео.
2. Не включайте одновременно слишком агрессивные `speed`/`resize`/`dynamic_shorts` без проверки качества.
3. Для стабильности держите pipeline явным и фиксированным для каждого канала.
4. Все изменения в `config.yaml` версионируйте в git.
