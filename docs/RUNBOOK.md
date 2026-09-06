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

### 4) Ошибка автозагрузки (`kodik_download`)

Проверьте:
- `pipeline` содержит `kodik_download`;
- параметры тайтла/серий/озвучки в секции `kodik_download`;
- ffmpeg есть в `PATH` — он нужен, если плеер отдаёт только `hls`/`dash`.

Токен здесь больше не нужен: этап работает на
[anime-dl-core](https://github.com/ialakey/anime-dl-core), а она читает ровно то
же, что читает обычный плеер в браузере.

#### Как устроен этап

1. каталог AnimeGO ищет тайтл и отдаёт ссылки на плееры конкретной серии;
2. `anime_dl_core.extract` превращает ссылку на плеер в прямые ссылки на видео;
3. поток скачивается: `mp4` — через `requests`, `hls`/`dash` — через `ffmpeg`.

Плееров у серии обычно несколько (Kodik, CVH, Aniboom, Sibnet), и они
перебираются по порядку: сначала запрошенная озвучка, затем те плееры, что
отдают прямой `mp4`. Отказ одного плеера больше не означает отказ этапа —
в логе будет строка `⚠️ Плеер ... не отдал видео`, а скачивание продолжится
со следующего.

#### Каталог недоступен

```
⚠️ Пропускаем скачивание: хост animego.org недоступен.
```

Значит, домен не резолвится — у многих провайдеров аниме-сайты заблокированы.
Порядок действий:

1. задать зеркало: `ANIMEGO_MIRROR=animego.me` в `.env`;
2. либо пустить трафик через прокси: `ANIME_DL_PROXY=socks5://127.0.0.1:1080`
   (для socks нужен `pip install "anime-dl-core[socks]"`);
3. если сломался сам разбор страницы (`ExtractionError`) — обновить библиотеку:
   `pip install -U anime-dl-core` и поправить пин в `requirements.txt`.

Пайплайн при недоступности каталога **не падает**: шаг скачивания пропускается, и
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
