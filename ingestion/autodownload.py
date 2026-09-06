"""Автоматическая загрузка исходных эпизодов через ``anime-dl-core``.

Модуль скачивает указанные тайтлы в папку ``input_videos`` перед запуском
остальных этапов обработки.

Как это устроено:

1. каталог (AnimeGO) ищет тайтл и отдаёт ссылки на плееры для нужной серии —
   ``anime_dl_core.sources.AnimeGo``;
2. ядро библиотеки превращает ссылку на плеер в прямые ссылки на видео —
   ``anime_dl_core.extract``;
3. прямой ``mp4`` качается обычным ``requests``, ``hls``/``dash`` — через
   ``ffmpeg`` (он и так нужен пайплайну).

Токен не нужен: библиотека читает ровно то же, что читает обычный плеер в
браузере. Плееров несколько (Kodik, CVH, Aniboom, Sibnet, ...), и если один
не отдал видео, берётся следующий — раньше отказ Kodik означал отказ всего
этапа.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import anime_dl_core as adc
import requests
from anime_dl_core.sources import AnimeGo, AnimeItem, PlayerLink
from requests import exceptions as req_exc

import config


class AutodownloadError(RuntimeError):
    """Базовая ошибка автозагрузки."""


#: Каталог, из которого берутся ссылки на плееры (можно заменить зеркалом).
SOURCE_HOST = "animego.org"

#: Порядок предпочтения плееров: сначала те, что отдают прямой mp4 —
#: их можно скачать без ffmpeg и без склейки сегментов.
PLAYER_PRIORITY = (
    "kodik",
    "cvh",
    "sibnet",
    "vk",
    "animedia",
    "aniboom",
    "anilibria",
    "sovetromantica",
)


def source_host() -> str:
    """Домен каталога с учётом зеркала из ``ANIMEGO_MIRROR``."""

    mirror = getattr(config, "ANIMEGO_MIRROR", None)
    return mirror if isinstance(mirror, str) and mirror else SOURCE_HOST


def network_hint() -> str:
    return (
        f"хост {source_host()} недоступен. Проверьте интернет, DNS или VPN "
        "(у многих провайдеров аниме-сайты заблокированы). Зеркало задаётся "
        "переменной ANIMEGO_MIRROR, прокси — ANIME_DL_PROXY."
    )


def is_source_reachable(host: Optional[str] = None) -> bool:
    """Быстро проверяет, что домен каталога вообще резолвится.

    Дешевле отсечь заведомо недоступный хост заранее и вернуть понятное
    сообщение, чем ждать таймаутов на каждой серии.
    """

    try:
        socket.getaddrinfo(host or source_host(), 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    return True


def client_kwargs() -> Dict[str, Any]:
    """Общие параметры http-клиента библиотеки (таймаут, прокси)."""

    kwargs: Dict[str, Any] = {"timeout": 30.0}
    proxy = getattr(config, "ANIME_DL_PROXY", None)
    if isinstance(proxy, str) and proxy:
        kwargs["proxy"] = proxy
    return kwargs


def create_site() -> AnimeGo:
    """Клиент каталога. Бросает :class:`AutodownloadError`, если хост недоступен."""

    if not is_source_reachable():
        raise AutodownloadError(f"Каталог недоступен: {network_hint()}")

    mirror = getattr(config, "ANIMEGO_MIRROR", None)
    return AnimeGo(
        mirror=mirror if isinstance(mirror, str) and mirror else None,
        **client_kwargs(),
    )


@dataclass(frozen=True)
class DownloadRequest:
    """Одна запись секции ``kodik_download``/``autodownload``."""

    title: str
    episodes: str
    voice: Optional[str] = None
    max_quality: Optional[int] = None
    player: Optional[str] = None


@dataclass(frozen=True)
class Candidate:
    """Ссылка на плеер, пригодная для скачивания."""

    link: PlayerLink
    player: str
    """Имя плеера в терминах библиотеки (``kodik``, ``cvh``, ...)."""
    voice_score: float

    @property
    def voice(self) -> str:
        return self.link.label or self.player


def normalise_filename(value: str) -> str:
    # Именно NFKC, а не NFKD: разложение отрывает диакритику в отдельный
    # combining-символ, который не проходит проверку isalnum() и выпадает —
    # «Тайтл» превращался в «Таитл», «ё» в «е».
    normalised = unicodedata.normalize("NFKC", value)
    allowed = []
    for char in normalised:
        if char.isalnum() or char in {" ", "-", "_"}:
            allowed.append(char)
        elif char in {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}:
            allowed.append("-")
    cleaned = "".join(allowed).strip()
    return "_".join(cleaned.split()) or "episode"


def choose_best_result(results: Iterable[AnimeItem], query: str) -> AnimeItem:
    """Самый похожий на запрос тайтл (сравниваются оба названия — ру и ориг.)."""

    query_lower = query.lower()

    def ratio(item: AnimeItem) -> float:
        titles = [item.title or "", item.original_title or ""]
        return max(SequenceMatcher(None, name.lower(), query_lower).ratio() for name in titles)

    try:
        return max(results, key=ratio)
    except ValueError as exc:
        raise AutodownloadError("Ничего не найдено по заданному названию.") from exc


def library_player(link: PlayerLink) -> Optional[str]:
    """Имя плеера библиотеки для ссылки или ``None``, если плеер не поддержан."""

    try:
        return adc.get_player_class(link.embed).name
    except adc.UnsupportedUrl:
        return None


def order_candidates(
    links: Iterable[PlayerLink],
    voice: Optional[str] = None,
    player: Optional[str] = None,
) -> List[Candidate]:
    """Кандидаты по убыванию пригодности: сначала нужная озвучка, затем плеер.

    Возвращается список, а не один вариант: плееры регулярно ломаются, и
    перебор — единственный способ всё-таки скачать серию.
    """

    voice_lower = voice.lower() if voice else None
    player_lower = player.lower() if player else None

    candidates: List[Candidate] = []
    for link in links:
        name = library_player(link)
        if name is None:
            continue
        if player_lower and name != player_lower:
            continue
        score = (
            SequenceMatcher(None, (link.label or "").lower(), voice_lower).ratio()
            if voice_lower
            else 0.0
        )
        candidates.append(Candidate(link=link, player=name, voice_score=score))

    def priority(candidate: Candidate) -> int:
        try:
            return PLAYER_PRIORITY.index(candidate.player)
        except ValueError:
            return len(PLAYER_PRIORITY)

    candidates.sort(key=lambda item: (-item.voice_score, priority(item)))
    return candidates


def pick_stream(result: adc.PlayerResult, max_quality: Optional[int] = None) -> Optional[adc.Stream]:
    """Лучший поток: максимальное качество, при равенстве — mp4 (его проще скачать).

    Мастер-плейлист берётся, только если ничего другого нет, а ограничение
    ``max_quality`` снимается, если под него не подошёл ни один поток.
    """

    limits = (max_quality, None) if max_quality else (None,)
    for limit in limits:
        for allow_master in (False, True):
            try:
                return result.best(max_quality=limit, allow_master=allow_master)
            except adc.NoStreamsFound:
                continue
    return None


def download_episode(
    url: str, destination: Path, headers: Optional[Dict[str, str]] = None
) -> None:
    """Скачивает прямой файл. ``headers`` обязательны: без Referer CDN отдаёт 403."""

    try:
        response = requests.get(url, stream=True, timeout=30, headers=headers or {})
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


def download_via_ffmpeg(stream: adc.Stream, destination: Path) -> None:
    """Склеивает HLS/DASH в mp4 без перекодирования."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AutodownloadError(
            "Поток отдаётся только как HLS/DASH, а ffmpeg не найден в PATH."
        )

    # ffmpeg выбирает контейнер по расширению, поэтому временный файл — тоже .mp4.
    partial = destination.with_name(destination.stem + ".part.mp4")
    args = stream.ffmpeg_args(str(partial))
    args[0] = ffmpeg
    args[1:1] = ["-y", "-hide_banner", "-loglevel", "error", "-nostdin"]

    result = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        details = (result.stderr or result.stdout or "").strip().splitlines()
        tail = details[-1] if details else "без подробностей"
        raise AutodownloadError(
            f"ffmpeg не смог скачать поток (код {result.returncode}): {tail}"
        )

    partial.replace(destination)
    print("   ⬇️ Загрузка завершена (ffmpeg).")


def download_stream(stream: adc.Stream, destination: Path) -> None:
    if stream.kind is adc.StreamKind.MP4:
        download_episode(stream.url, destination, headers=dict(stream.headers))
    else:
        download_via_ffmpeg(stream, destination)


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


def _episode_filename(title: str, episode: int, voice: str, quality: Optional[int]) -> str:
    quality_part = f"{quality}p" if quality else "auto"
    return normalise_filename(f"{title} - Серия {episode} - {voice} - {quality_part}") + ".mp4"


def _download_one_episode(
    site: AnimeGo,
    anime: AnimeItem,
    episode: int,
    request: DownloadRequest,
    destination: Path,
) -> Optional[Path]:
    """Скачивает одну серию, перебирая плееры. ``None`` — не получилось."""

    try:
        links = site.players(anime.id, episode=episode)
    except adc.AnimeDlCoreError as exc:
        print(f"   ❌ Нет плееров для серии {episode}: {exc}")
        return None

    candidates = order_candidates(links, request.voice, request.player)
    if not candidates:
        print(f"   ❌ Для серии {episode} нет плееров, поддержанных anime-dl-core.")
        return None

    for candidate in candidates:
        try:
            result = adc.extract(candidate.link.embed, **client_kwargs())
        except adc.AnimeDlCoreError as exc:
            print(f"   ⚠️ Плеер {candidate.player} ({candidate.voice}) не отдал видео: {exc}")
            continue

        stream = pick_stream(result, request.max_quality)
        if stream is None:
            print(f"   ⚠️ Плеер {candidate.player} ({candidate.voice}) не вернул ни одного потока.")
            continue

        output_path = destination / _episode_filename(
            anime.title or request.title, episode, candidate.voice, stream.quality
        )
        if output_path.exists():
            print(f"   ⚠️ Файл уже существует, пропускаем: {output_path.name}")
            return output_path

        print(f"   🎧 Озвучка: {candidate.voice}")
        print(
            f"   📺 Плеер: {candidate.player}, качество: {stream.quality or 'auto'}, "
            f"формат: {stream.kind}"
        )
        print(f"   📂 Путь сохранения: {output_path}")

        try:
            download_stream(stream, output_path)
        except AutodownloadError as exc:
            print(f"   ❌ Не удалось скачать через {candidate.player}: {exc}")
            continue

        print(f"   💾 Сохранено: {output_path}")
        return output_path

    print(f"   ❌ Серия {episode} не скачана: все плееры отказали.")
    return None


def download_request(request: DownloadRequest, destination: Path) -> List[Path]:
    episode_list = parse_episode_range(request.episodes)
    if not episode_list:
        raise AutodownloadError("Не указаны серии для скачивания.")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []
    with create_site() as site:
        try:
            results = site.search(request.title, limit=15)
        except adc.AnimeDlCoreError as exc:
            raise AutodownloadError(
                f"Ошибка поиска тайтла '{request.title}': {exc}"
            ) from exc

        anime = choose_best_result(results, request.title)
        print(f"   🔎 Найдено: {anime.title} ({anime.url})")

        for episode in episode_list:
            print(f"  🎬 Скачиваем {anime.title or request.title} — серия {episode}")
            saved = _download_one_episode(site, anime, episode, request, destination)
            if saved is not None:
                saved_files.append(saved)

    return saved_files


def download_by_title(
    title: str,
    episodes: str = "1",
    voice: Optional[str] = None,
    destination: Optional[Path] = None,
    *,
    max_quality: Optional[int] = None,
    player: Optional[str] = None,
) -> List[Path]:
    request = DownloadRequest(
        title=title,
        episodes=str(episodes),
        voice=voice,
        max_quality=max_quality,
        player=player,
    )
    return download_request(request, Path(destination or Path.cwd()))


def _as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(str(value).rstrip("pP"))
    except ValueError:
        return None


def _normalise_entry(entry: object) -> DownloadRequest:
    quality: Any = None
    player: Any = None

    if isinstance(entry, dict):
        title = entry.get("title") or entry.get("name")
        episodes = entry.get("episodes") or entry.get("count")
        voice = entry.get("voice") or entry.get("translation")
        quality = entry.get("quality") or entry.get("max_quality")
        player = entry.get("player")
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        try:
            title, episodes, *rest = entry
        except ValueError as exc:
            raise AutodownloadError(
                "Каждый элемент kodik_download/autodownload должен содержать минимум название и количество серий."
            ) from exc
        voice = rest[0] if rest else None
        quality = rest[1] if len(rest) > 1 else None
        player = rest[2] if len(rest) > 2 else None
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

    return DownloadRequest(
        title=str(title),
        episodes=str(episodes),
        voice=str(voice) if voice not in (None, "") else None,
        max_quality=_as_int(quality),
        player=str(player) if player not in (None, "") else None,
    )


def auto_download_titles(entries: Optional[Iterable[object]], destination: Path) -> List[Path]:
    if not entries:
        print("⚠️ Раздел kodik_download/autodownload пуст — пропускаем скачивание.")
        return []

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if not is_source_reachable():
        print(f"⚠️ Пропускаем скачивание: {network_hint()}")
        print("   Продолжаем с уже скачанными видео в input_videos.")
        return []

    all_downloaded: List[Path] = []

    for raw_entry in entries:
        try:
            request = _normalise_entry(raw_entry)
        except AutodownloadError as exc:
            print(f"❌ Пропускаем запись kodik_download/autodownload: {exc}")
            continue

        print(
            f"⬇️ Автозагрузка: {request.title} (серии: {request.episodes}, "
            f"озвучка: {request.voice or 'любая'})"
        )
        try:
            downloaded = download_request(request, destination)
        except AutodownloadError as exc:
            print(f"❌ Не удалось скачать '{request.title}': {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - сеть/библиотека не должны ронять пайплайн
            print(f"❌ Непредвиденная ошибка при скачивании '{request.title}': {exc}")
            continue

        all_downloaded.extend(downloaded)

    if all_downloaded:
        print(f"✅ Скачано файлов: {len(all_downloaded)}")
    else:
        print("⚠️ Не удалось скачать ни одного файла.")

    return all_downloaded
