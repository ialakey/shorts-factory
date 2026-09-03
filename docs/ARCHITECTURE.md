# Архитектура проекта

## 1. Оркестрация запуска

### `app.py`
- запускает проверку каналов (`check_all_channels`),
- получает список валидных каналов,
- последовательно вызывает `process_channel(channel_name)`.

### `infrastructure/check_structure.py`
- сканирует `channels/*`,
- создает недостающие папки,
- при отсутствии `config.yaml` генерирует его из `DEFAULT_CONFIG`.

## 2. Обработка канала

### `core/channel_processor.py`
Основной orchestrator по шагам pipeline:

1. Загружает `channels/<name>/config.yaml`.
2. Вычисляет `pipeline` и режим `debug`.
3. Опционально запускает Kodik-загрузку (`kodik_download`/`autodownload`).
4. Собирает входные видео из `input_videos`.
5. Для каждого видео выполняет этапы:
   - `transcribe_video`
   - `analyze_moment`
   - `make_clips` (включая fallback-автозапуск)
   - `spoof_metadata`
   - `telegram_notify`

### Debug mode
Если в config `debug: true`, то:
- input = `test_data/`
- output = `output_test_data/`
- создаются debug-дампы транскрипта/моментов.

## 3. Модули пайплайна

### Ingestion
- `ingestion/autodownload.py`: поиск/скачивание эпизодов через Kodik.
- `ingestion/transcriber.py`: извлечение аудио (`ffmpeg`) + Whisper транскрибация.
- `ingestion/parser.py`: поиск локальных видео в `input_videos/`.

### Analysis
- `analysis/gpt_analyzer.py`: сбор сигналов, обращение к OpenAI API, финальный отбор моментов.
- `analysis/audio_analyzer.py`: вспомогательные аудио-сигналы (пики/эмоции).
- `analysis/moment_scorer.py`: детерминированный скоринг окон-кандидатов.
- `analysis/moment_validator.py`: починка и валидация моментов, вернувшихся от LLM.
- `analysis/subtitles_cleaner.py`: пост-обработка/очистка текста субтитров.

#### Слои отбора моментов

```
transcript → audio events → face/scene activity → tempo windows → hooks
      → scored candidates → LLM → repair + validation → moments
```

Каждый сигнал по отдельности шумный, поэтому решение принимается композицией:

| Сигнал | Что оценивает | Зачем |
|--------|---------------|-------|
| `transcript` | плотность и объём реплик | смысловой крючок |
| `audio` | энергия и «острота» пиков | не потерять эмоциональные всплески |
| `face` | наличие и крупность лиц | есть ли на чём держать вертикальный кадр |
| `scene` | склейки и движение | отсечь статичные пустые окна |
| `pacing` | паузы и провисания | отсечь «вязкие» куски |
| `hook` | первые 1–2 секунды | без входа момент почти всегда проигрывает |

`score = Σ weight_i * signal_i` (веса в `moment_scoring.weights`, нормализуются
к 1), затем NMS по пересечению окон. Кандидаты уходят в промпт LLM: модель
работает как слой агрегации уже подготовленных сигналов, а не гадает с нуля.

Ответ модели не принимается на веру: `moment_validator` клампит границы,
разводит пересечения segments, притягивает резы к паузам речи и склейкам,
подгоняет сумму длительностей под `gpt.min_time`/`max_time` и отбрасывает
пустые фрагменты. Если валидных моментов не хватает — список добивается
эвристическими кандидатами, поэтому эпизод не падает из-за плохого ответа
модели.

Промежуточные артефакты (`*_candidates.json`, `*_signals.json`,
`*_chatgpt_payload.txt`) сохраняются рядом с клипами: их можно разбирать
отдельно, не перезапуская весь пайплайн.

### Rendering
- `rendering/video_editor.py`: сборка клипов, эффекты, фон, музыка, субтитры, тайтл, watermark.
- `rendering/face_detector.py`: виртуальная камера — динамический авто-кроп 16:9 → 9:16.

#### Виртуальная камера

Камера — не «поставить центр на лицо», а физическая модель:

1. **Детект**: каскад MediaPipe → YuNet → Haar (fail-soft).
2. **Выбор героя**: saliency = уверенность + крупность + центральность,
   переключение фокуса только с запасом (`face_switch_margin`) и выдержкой
   (`face_switch_hold_s`) — иначе камера пинг-понгует между персонажами.
3. **Фильтрация**: анти-джерк (`max_face_step`) + low-pass (`face_filter`).
4. **Композиция**: eye-level lift, rule-of-thirds side bias, защитные поля.
5. **Физика**: `a = k·error − c·v` с ограничением ускорения и скорости,
   dead zone по ошибке слежения, predictive lead и human lag.
6. **Зум**: целевая крупность лица (`face_coverage`) с плавным изменением и
   лимитом скорости.
7. **Склейки**: на монтажном резе камера не «переезжает» через кадр, а
   мгновенно пересобирается (порог адаптивный: абсолютный минимум + множитель
   к локальному фону).
8. **Fallback**: без лица — Ken Burns вокруг последнего известного положения
   героя с медленным возвратом к центру, а не прыжок в середину кадра.

Профили `static` / `operator` / `action` задают базовый темперамент камеры;
любой ключ в `dynamic_shorts` переопределяет профиль точечно.
- `rendering/layers/*`: рендер отдельных визуальных слоев.
- `rendering/audio/*`: обработка голосовой дорожки.

### Publishing
- `publishing/spoof_metadata.py`: подмена метаданных mp4.
- `publishing/telegram_notifier.py`: отправка медиа-группы в Telegram.

## 4. Конфигурация

- Глобальные пути и env-хуки: `config.py`.
- Дефолтная структура канала и шаблон `config.yaml`: `DEFAULT_CONFIG` в `config.py`.
- Главный runtime-конфиг: `channels/<Channel>/config.yaml`.

## 5. Поток данных

`input_videos/*.mp4 -> transcribe -> transcript -> GPT moment selection -> clip rendering -> output_clips/*.mp4 -> optional metadata spoof -> optional telegram upload`

## 6. Внешние зависимости

- FFmpeg/ffprobe
- OpenAI API
- Whisper (локальная модель)
- MoviePy + OpenCV + Pillow
- Опционально: Kodik API, Telegram Bot API
