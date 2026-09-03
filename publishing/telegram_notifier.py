import html
import json
import re
from pathlib import Path
from typing import Iterable, List

import requests

MEDIA_GROUP_LIMIT = 10


def _humanize_text(raw_text: str) -> str:
    """Make text human-friendly: drop underscores and collapse whitespace."""

    text = str(raw_text or "")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def normalize_anime_name(raw_name: str) -> str:
    """Prepare a readable anime title from a filename stem.

    Removes common quality suffixes like "720p"/"1080p" (with optional dashes or
    spaces), strips extra punctuation, and replaces underscores with spaces for
    nicer display in Telegram.
    """

    name = str(raw_name)
    # `\b` здесь не годится: "_" — словообразующий символ, поэтому в
    # "Ад_1080p_серия" граница слова после "1080p" не срабатывала.
    name = re.sub(r"(?i)(?:\s*[-_]?\s*)?(\d{3,4}p)(?![^\W_])", "", name)
    name = re.sub(r"[^\w]+", " ", name, flags=re.UNICODE)
    name = _humanize_text(name)
    return name or "anime"


def format_clip_title(raw_title: str, fallback: str | None = None) -> str:
    """Return a cleaned-up title for Telegram captions.

    Underscores are turned into spaces, placeholder titles like "moment_1" are
    replaced with the provided fallback, and redundant whitespace is removed.
    """

    title = _humanize_text(raw_title)
    # _humanize_text уже превратил "moment_1" в "moment 1", поэтому проверяем
    # оба варианта разделителя, иначе плейсхолдер утекал в подпись.
    if re.fullmatch(r"moment[\s_]*\d+", title, flags=re.IGNORECASE):
        title = _humanize_text(fallback)

    return title or _humanize_text(fallback)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"

    def _build_message(self, titles: List[str], anime_name: str) -> str:
        title_lines = [t for t in (titles or []) if t]
        if not title_lines:
            title_lines = ["Без названия"]

        escaped_titles = [html.escape(t.strip()) for t in title_lines]
        formatted_titles = "\n\n".join(escaped_titles)
        anime_block = f"<b>Название аниме:</b> {html.escape(anime_name)}"

        return f"{formatted_titles}\n\n{anime_block}"

    def send_media_group(
        self,
        titles: List[str],
        anime_name: str,
        video_paths: Iterable[Path],
    ) -> bool:
        paths = [Path(p) for p in video_paths]
        if not paths:
            print("⚠️ Нет клипов для отправки в Telegram.")
            return False

        message = self._build_message(titles, anime_name)

        # Telegram принимает не больше MEDIA_GROUP_LIMIT файлов в одной группе.
        chunks = [
            paths[i : i + MEDIA_GROUP_LIMIT]
            for i in range(0, len(paths), MEDIA_GROUP_LIMIT)
        ]

        all_sent = True
        for chunk_idx, chunk in enumerate(chunks):
            if not self._send_chunk(chunk, message if chunk_idx == 0 else None):
                all_sent = False

        if all_sent:
            print("✅ Сообщение и клипы отправлены в Telegram!")
        return all_sent

    def _send_chunk(self, paths: List[Path], caption: str | None) -> bool:
        files = {}
        media = []
        try:
            for idx, path in enumerate(paths):
                file_key = f"video{idx}"
                try:
                    files[file_key] = open(path, "rb")
                except OSError as exc:
                    print(f"❌ Не удалось открыть {path}: {exc}")
                    return False

                media_item = {
                    "type": "video",
                    "media": f"attach://{file_key}",
                }

                if idx == 0 and caption:
                    media_item["caption"] = str(caption)
                    media_item["parse_mode"] = "HTML"

                media.append(media_item)

            try:
                response = requests.post(
                    f"{self.api_base}/sendMediaGroup",
                    data={"chat_id": self.chat_id, "media": json.dumps(media)},
                    files=files,
                    timeout=120,
                )
            except requests.RequestException as exc:
                print(f"❌ Не удалось связаться с Telegram: {exc}")
                return False

            if response.status_code != 200:
                print(
                    "❌ Ошибка Telegram:",
                    response.status_code,
                    response.text,
                )
                return False
            return True
        finally:
            for fh in files.values():
                try:
                    fh.close()
                except Exception:
                    pass


__all__ = ["TelegramNotifier", "normalize_anime_name", "format_clip_title"]
