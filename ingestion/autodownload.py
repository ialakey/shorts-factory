"""Автоматическая загрузка исходных эпизодов через Kodik.

Модуль адаптирован из скрипта ``kodik-download/download_episode.py`` и
предназначен для использования внутри пайплайна канала. Основная задача —
скачать указанные тайтлы в папку ``input_videos`` перед запуском остальных
этапов обработки.
"""

from __future__ import annotations

import os
import socket
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
from anime_parsers_ru import KodikParser
from requests import exceptions as req_exc

import config


class AutodownloadError(RuntimeError):
    """Базовая ошибка автозагрузки."""


# Kodik переехал: старый ``kodikapi.com`` снят с делегирования (NXDOMAIN),
# актуальный домен API — ``kodik-api.com``.
KODIK_API_HOST = "kodik-api.com"

NETWORK_HINT = (
    f"хост {KODIK_API_HOST} недоступен. Проверьте интернет, DNS или VPN "
    "(у многих провайдеров Kodik заблокирован)."
)


def is_kodik_reachable(host: str = KODIK_API_HOST) -> bool:
    """Быстро проверяет, что домен Kodik вообще резолвится.

    Библиотека ``anime_parsers_ru`` ходит в сеть без таймаутов и превращает
    сетевые сбои в ``UnexpectedBehavior``, поэтому дешевле отсечь заведомо
    недоступный хост заранее и вернуть понятное сообщение.
    """

    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    return True


@dataclass
class SearchResult:
    raw: dict

    @property
    def title(self) -> str:
        return self.raw.get("title") or ""

    @property
    def shikimori_id(self) -> Optional[str]:
        value = self.raw.get("shikimori_id")
        return str(value) if value is not None else None

    @property
    def kinopoisk_id(self) -> Optional[str]:
        value = self.raw.get("kinopoisk_id")
        return str(value) if value is not None else None


@dataclass
class Translation:
    id: str
    name: str
    type: str

    @classmethod
    def from_dict(cls, data: dict) -> "Translation":
        return cls(
            id=str(data.get("id", "0")),
            name=str(data.get("name", "Неизвестно")),
            type=str(data.get("type", "voice")),
        )


def normalise_filename(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value)
    allowed = []
    for char in normalised:
        if char.isalnum() or char in {" ", "-", "_"}:
            allowed.append(char)
        elif char in {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}:
            allowed.append("-")
    cleaned = "".join(allowed).strip()
    return "_".join(cleaned.split()) or "episode"


def resolve_token() -> Optional[str]:
    token = os.environ.get("KODIK_TOKEN")
    if token:
        return token
    if isinstance(config.KODIK_TOKEN, str) and config.KODIK_TOKEN:
        return config.KODIK_TOKEN
    return None


def _build_parser(token: Optional[str], validate: bool) -> KodikParser:
    if token:
        return KodikParser(token=token, use_lxml=config.USE_LXML, validate_token=validate)
    return KodikParser(use_lxml=config.USE_LXML, validate_token=validate)


def create_parser(token: Optional[str]) -> KodikParser:
    if not is_kodik_reachable():
        raise AutodownloadError(f"Kodik недоступен: {NETWORK_HINT}")

    try:
        return _build_parser(token, validate=True)
    except Exception as exc:  # noqa: BLE001 - библиотека бросает разные исключения
        print(f"   ⚠️ Не удалось проверить токен Kodik ({exc}). Пробуем без валидации ...")

    try:
        return _build_parser(token, validate=False)
    except Exception as exc:  # noqa: BLE001 - библиотека бросает разные исключения
        raise AutodownloadError(
            f"Не удалось инициализировать парсер Kodik: {exc}"
        ) from exc


def choose_best_result(results: Iterable[SearchResult], query: str) -> SearchResult:
    try:
        return max(
            results,
            key=lambda item: SequenceMatcher(None, item.title.lower(), query.lower()).ratio(),
        )
    except ValueError as exc:
        raise AutodownloadError("Ничего не найдено по заданному названию.") from exc


def choose_translation(translations: Iterable[dict], voice: Optional[str]) -> Translation:
    prepared = [Translation.from_dict(item) for item in translations]
    if not prepared:
        raise AutodownloadError("Нет доступных переводов для выбранного тайтла.")

    if not voice:
        return prepared[0]

    voice_lower = voice.lower()

    def weight(item: Translation) -> float:
        return SequenceMatcher(None, item.name.lower(), voice_lower).ratio()

    return max(prepared, key=weight)


def download_episode(url: str, destination: Path) -> None:
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except req_exc.RequestException as exc:
        raise AutodownloadError(f"Ошибка при скачивании: {exc}") from exc

    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0

    # Пишем во временный файл, чтобы недокачанный ролик не попал в input_videos.
    partial = destination.with_name(destination.name + ".part")
    try:
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress = downloaded / total * 100
                    print(f"\r   ⬇️ Загрузка: {progress:5.1f}%", end="", flush=True)
    except (OSError, req_exc.RequestException) as exc:
        partial.unlink(missing_ok=True)
        raise AutodownloadError(f"Ошибка при скачивании: {exc}") from exc
    finally:
        response.close()

    if total and downloaded < total:
        partial.unlink(missing_ok=True)
        raise AutodownloadError(
            f"Файл скачан не полностью ({downloaded} из {total} байт)."
        )

    partial.replace(destination)
    print("\r   ⬇️ Загрузка завершена." + " " * 20)


def parse_episode_range(spec: str) -> List[int]:
    result: List[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_i = int(start)
            end_i = int(end)
            if end_i < start_i:
                raise ValueError("Диапазон серий указан некорректно.")
            result.extend(range(start_i, end_i + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def download_by_title(
    title: str,
    episodes: str = "1",
    voice: Optional[str] = "AniLibria",
    destination: Optional[Path] = None,
) -> List[Path]:
    episode_list = parse_episode_range(str(episodes))

    if not episode_list:
        raise AutodownloadError("Не указаны серии для скачивания.")

    destination = Path(destination or Path.cwd())
    destination.mkdir(parents=True, exist_ok=True)

    token = resolve_token()
    kodik = create_parser(token)

    try:
        results = [SearchResult(raw=item) for item in kodik.search(title, limit=15)]
    except Exception as exc:  # noqa: BLE001 - библиотека генерирует разные исключения
        raise AutodownloadError(f"Ошибка поиска тайтла '{title}': {exc}") from exc

    chosen = choose_best_result(results, title)

    if chosen.shikimori_id:
        id_type = "shikimori"
        serial_id = chosen.shikimori_id
    elif chosen.kinopoisk_id:
        id_type = "kinopoisk"
        serial_id = chosen.kinopoisk_id
    else:
        raise AutodownloadError("Не удалось определить идентификатор тайтла.")

    try:
        serial_info = kodik.get_info(serial_id, id_type)
    except Exception as exc:  # noqa: BLE001 - библиотека генерирует разные исключения
        raise AutodownloadError(
            f"Не удалось получить информацию о тайтле '{title}': {exc}"
        ) from exc

    translation = choose_translation(serial_info.get("translations", []), voice)

    saved_files: List[Path] = []

    for ep in episode_list:
        print(f"  🎬 Скачиваем {title} — серия {ep}")
        try:
            # anime_parsers_ru >= 1.13 возвращает третьим элементом список качеств,
            # более старые версии — только ссылку и максимальное качество.
            link, max_quality, *_ = kodik.get_link(serial_id, id_type, ep, translation.id)
        except Exception as exc:  # noqa: BLE001 - библиотека генерирует разные исключения
            print(f"   ❌ Не удалось получить ссылку на серию {ep}: {exc}")
            continue

        filename = (
            normalise_filename(
                f"{chosen.title or title} - Серия {ep} - {translation.name} - {max_quality}p"
            )
            + ".mp4"
        )
        output_path = destination / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"   ⚠️ Файл уже существует, пропускаем: {output_path.name}")
            saved_files.append(output_path)
            continue

        print(f"   🎧 Озвучка: {translation.name}")
        print(f"   📺 Качество: {max_quality}p")
        print(f"   📂 Путь сохранения: {output_path}")

        try:
            download_episode(f"https:{link}{max_quality}.mp4", output_path)
        except AutodownloadError as exc:
            print(f"   ❌ Серия {ep} не скачана: {exc}")
            continue

        print(f"   💾 Сохранено: {output_path}")
        saved_files.append(output_path)

    return saved_files


def _normalise_entry(entry: object) -> Tuple[str, str, Optional[str]]:
    if isinstance(entry, dict):
        title = entry.get("title") or entry.get("name")
        episodes = entry.get("episodes") or entry.get("count")
        voice = entry.get("voice") or entry.get("translation")
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        try:
            title, episodes, *rest = entry
        except ValueError as exc:
            raise AutodownloadError(
                "Каждый элемент kodik_download/autodownload должен содержать минимум название и количество серий."
            ) from exc
        voice = rest[0] if rest else None
    else:
        raise AutodownloadError(
            "Элементы kodik_download/autodownload должны быть словарями или последовательностями (title, episodes, voice)."
        )

    if not title:
        raise AutodownloadError(
            "Не указано название тайтла в настройках kodik_download/autodownload."
        )

    if episodes is None:
        raise AutodownloadError(
            f"Не указаны серии для тайтла '{title}' в настройках kodik_download/autodownload."
        )

    return str(title), str(episodes), (str(voice) if voice not in (None, "") else None)


def auto_download_titles(entries: Optional[Iterable[object]], destination: Path) -> List[Path]:
    if not entries:
        print("⚠️ Раздел kodik_download/autodownload пуст — пропускаем скачивание.")
        return []

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if not is_kodik_reachable():
        print(f"⚠️ Пропускаем скачивание: {NETWORK_HINT}")
        print("   Продолжаем с уже скачанными видео в input_videos.")
        return []

    all_downloaded: List[Path] = []

    for raw_entry in entries:
        try:
            title, episodes, voice = _normalise_entry(raw_entry)
        except AutodownloadError as exc:
            print(f"❌ Пропускаем запись kodik_download/autodownload: {exc}")
            continue

        print(f"⬇️ Автозагрузка: {title} (серии: {episodes}, озвучка: {voice or 'по умолчанию'})")
        try:
            downloaded = download_by_title(title, episodes, voice, destination)
        except AutodownloadError as exc:
            print(f"❌ Не удалось скачать '{title}': {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - сеть/библиотека не должны ронять пайплайн
            print(f"❌ Непредвиденная ошибка при скачивании '{title}': {exc}")
            continue

        all_downloaded.extend(downloaded)

    if all_downloaded:
        print(f"✅ Скачано файлов: {len(all_downloaded)}")
    else:
        print("⚠️ Не удалось скачать ни одного файла.")

    return all_downloaded
