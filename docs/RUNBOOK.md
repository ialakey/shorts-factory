# Runbook: запуск и диагностика

## Базовая проверка окружения

```bash
python -V
ffmpeg -version
ffprobe -version
```

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Типовые проблемы

### 1) `OPENAI_API_KEY` отсутствует

Симптом: падение на старте с ошибкой из `config.py`.

Что делать:
- создать `.env` в корне;
- добавить `OPENAI_API_KEY=...`;
- перезапустить.

### 2) Нет клипов на выходе

Проверьте:
1. есть ли видео в `channels/<Channel>/input_videos`;
2. присутствует ли `make_clips` в pipeline или активирован ли fallback-путь;
3. вернул ли `analyze_moment` хотя бы один валидный сегмент.

### 3) Ошибка субтитров/Whisper

Проверьте:
- корректность `subtitles.whisper_model`;
- доступность ресурсов для модели;
- что ffmpeg корректно извлекает аудио дорожку.

### 4) Ошибка Kodik download

Проверьте:
- `pipeline` содержит `kodik_download`;
- задан `KODIK_TOKEN` в `.env` (в коде токена нет);
- параметры тайтла/серий/озвучки в секции `kodik_download`.

#### Домены Kodik переехали

Kodik снял с делегирования старые домены: `kodikapi.com`, `kodik.info` и
`kodik.biz` возвращают **NXDOMAIN от авторитативных серверов зоны** (проверено
через DoH Cloudflare, а не только через локальный DNS) — VPN и смена DNS тут не
помогут. Актуальные адреса:

| Назначение | Было | Стало |
| --- | --- | --- |
| API | `kodikapi.com` | `kodik-api.com` |
| Плеер | `kodik.info` | `kodikplayer.com` |

Пайплайн использует `anime_parsers_ru>=1.17.0` — там новые домены уже прошиты.
Если увидели сообщение вида

```
⚠️ Пропускаем скачивание: хост kodik-api.com недоступен.
```

значит, домен не резолвится и на этот раз. Порядок действий:

1. проверить, что установлена актуальная версия парсера:
   `pip install -U anime_parsers_ru` (и обновить пин в `requirements.txt`);
2. посмотреть, какой домен API сейчас использует Kodik — он зашит в
   `https://kodik-add.com/add-players.min.js?v=2` (там же лежит и публичный
   токен);
3. поправить `KODIK_API_HOST` в `ingestion/autodownload.py`.

Пайплайн при недоступности Kodik **не падает**: шаг скачивания пропускается, и
обработка продолжается с теми файлами, что уже лежат в `input_videos`. Как
временное решение можно складывать исходники в
`channels/<канал>/input_videos` вручную либо убрать `kodik_download` из
`pipeline`, чтобы не видеть предупреждение.

### 5) Telegram notify не отправляет

Проверьте:
- `pipeline` содержит `telegram_notify`;
- выставлены `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`;
- после обработки реально появились новые `.mp4`.

## Режим безопасной отладки

Рекомендуется для инженера:
1. поставить `debug: true`;
2. положить тестовое видео в `test_data/`;
3. запустить `python app.py`;
4. анализировать результаты в `output_test_data/`.

## Полезные команды

```bash
# список входных видео
find channels -maxdepth 3 -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.mkv' \)

# быстрый просмотр размера output
du -sh channels/*/output_clips 2>/dev/null

# проверка что python импортирует app
python -m py_compile app.py core/channel_processor.py rendering/video_editor.py
```
