"""Тесты автозагрузки исходников (``ingestion/autodownload.py``).

Сеть не трогается: каталог AnimeGO подменяется фейком, ``anime_dl_core.extract``
и само скачивание — монкипатчами. Модели и реестр плееров библиотеки при этом
настоящие — именно они и проверяются.
"""

from __future__ import annotations

from pathlib import Path

import anime_dl_core as adc
import pytest
from anime_dl_core.sources import AnimeItem, PlayerLink

from ingestion import autodownload as ad


# ============================================================
# Фейковый каталог
# ============================================================

KODIK_EMBED = "https://kodikplayer.com/seria/1304528/932d5da818729ec5ccc9be7968ee3717/720p"
ANIBOOM_EMBED = "https://aniboom.one/embed/9G1MJ6NMV8z?episode=1&translation=30"
CVH_EMBED = "https://animego.org/cdn-iframe/40748/JamClub/1/1"


def link(player: str, label: str, embed: str) -> PlayerLink:
    return PlayerLink(player=player, label=label, embed=embed)


class FakeSite:
    """Минимальный двойник :class:`anime_dl_core.sources.AnimeGo`."""

    def __init__(self, results, players_by_episode):
        self.results = results
        self.players_by_episode = players_by_episode
        self.searched = []
        self.closed = False

    def search(self, query, limit=15):
        self.searched.append((query, limit))
        return self.results

    def players(self, anime_id, episode=1):
        if episode not in self.players_by_episode:
            raise adc.NotFound(f"нет серии {episode}")
        return self.players_by_episode[episode]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True


def anime(title="Тайтл", original=None, anime_id="42") -> AnimeItem:
    return AnimeItem(
        id=anime_id,
        slug="title",
        title=title,
        url=f"https://animego.org/anime/title-{anime_id}",
        original_title=original,
    )


def stream(url, kind, quality=None, label=None) -> adc.Stream:
    return adc.Stream(
        url=url,
        kind=adc.StreamKind(kind),
        quality=quality,
        headers={"Referer": "https://example.org/"},
        label=label,
    )


def result(*streams, player="kodik") -> adc.PlayerResult:
    return adc.PlayerResult(player=player, source_url="https://example.org/", streams=list(streams))


@pytest.fixture
def site_factory(monkeypatch):
    """Подменяет ``create_site`` и возвращает созданный фейк."""

    holder = {}

    def install(results, players_by_episode):
        fake = FakeSite(results, players_by_episode)
        holder["site"] = fake
        monkeypatch.setattr(ad, "create_site", lambda: fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Страховка: незамоканный поход в сеть падает, а не уходит наружу.

    Перехватываются именно границы наружу — ``extract``, ``requests.get`` и
    запуск ffmpeg, — поэтому сама логика модуля в тестах остаётся настоящей.
    """

    def forbidden(*args, **kwargs):  # pragma: no cover - срабатывает при ошибке теста
        raise AssertionError("Тест попытался сходить в сеть")

    monkeypatch.setattr(ad, "is_source_reachable", lambda host=None: True)
    monkeypatch.setattr(ad.adc, "extract", forbidden)
    monkeypatch.setattr(ad.requests, "get", forbidden)
    monkeypatch.setattr(ad.subprocess, "run", forbidden)


# ============================================================
# Разбор параметров
# ============================================================


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("1", [1]),
        ("1-3", [1, 2, 3]),
        ("1, 3, 2", [1, 2, 3]),
        ("2-3,7", [2, 3, 7]),
        ("1,1,1", [1]),
    ],
)
def test_parse_episode_range(spec, expected):
    assert ad.parse_episode_range(spec) == expected


def test_parse_episode_range_rejects_reversed():
    with pytest.raises(ValueError):
        ad.parse_episode_range("5-2")


def test_normalise_filename_strips_path_separators():
    name = ad.normalise_filename('Тайтл: "часть" 2/3')
    assert "/" not in name and ":" not in name and '"' not in name
    assert " " not in name


def test_normalise_filename_keeps_cyrillic_intact():
    # NFKD разложил бы «й» и «ё» на букву + combining-символ, а тот выпал бы
    # из-за isalnum() — в имени файла оставалось бы «Таитл про ежика».
    assert ad.normalise_filename("Тайтл про ёжика") == "Тайтл_про_ёжика"


def test_normalise_entry_sequence_with_quality_and_player():
    request = ad._normalise_entry(["Тайтл", "1-2", "AniLibria", "720p", "kodik"])
    assert request == ad.DownloadRequest(
        title="Тайтл", episodes="1-2", voice="AniLibria", max_quality=720, player="kodik"
    )


def test_normalise_entry_dict_form():
    request = ad._normalise_entry(
        {"title": "Тайтл", "episodes": 3, "translation": "Студия", "max_quality": 480}
    )
    assert request.episodes == "3"
    assert request.voice == "Студия"
    assert request.max_quality == 480
    assert request.player is None


def test_normalise_entry_keeps_minimal_form():
    request = ad._normalise_entry(["Тайтл", "1"])
    assert request.voice is None and request.max_quality is None and request.player is None


@pytest.mark.parametrize(
    "entry",
    [
        "просто строка",
        ["ТолькоНазвание"],
        [None, "1"],
        {"episodes": "1"},
        {"title": "Тайтл"},
    ],
)
def test_normalise_entry_rejects_broken_entries(entry):
    with pytest.raises(ad.AutodownloadError):
        ad._normalise_entry(entry)


# ============================================================
# Выбор тайтла, плеера и потока
# ============================================================


def test_choose_best_result_matches_original_title():
    items = [anime("Совсем другое"), anime("Магическая битва", original="Jujutsu Kaisen")]
    assert ad.choose_best_result(items, "Jujutsu Kaisen").title == "Магическая битва"


def test_choose_best_result_without_results():
    with pytest.raises(ad.AutodownloadError):
        ad.choose_best_result([], "Тайтл")


def test_library_player_recognises_embeds_and_skips_unknown():
    assert ad.library_player(link("Kodik", "AniLibria", KODIK_EMBED)) == "kodik"
    assert ad.library_player(link("AniBoom", "AniLibria", ANIBOOM_EMBED)) == "aniboom"
    assert ad.library_player(link("CVH", "Jam Club", CVH_EMBED)) == "cvh"
    assert ad.library_player(link("Что-то", "Озвучка", "https://example.org/embed/1")) is None


def test_order_candidates_prefers_requested_voice():
    links = [
        link("AniBoom", "Студийная банда", ANIBOOM_EMBED),
        link("Kodik", "AniLibria", KODIK_EMBED),
    ]
    ordered = ad.order_candidates(links, voice="AniLibria")
    assert [c.voice for c in ordered] == ["AniLibria", "Студийная банда"]


def test_order_candidates_falls_back_to_player_priority():
    links = [
        link("AniBoom", "Одна озвучка", ANIBOOM_EMBED),
        link("Kodik", "Одна озвучка", KODIK_EMBED),
    ]
    # Без запрошенной озвучки решает приоритет плееров: kodik отдаёт прямой mp4.
    assert [c.player for c in ad.order_candidates(links)] == ["kodik", "aniboom"]


def test_order_candidates_filters_by_player_and_drops_unsupported():
    links = [
        link("Kodik", "AniLibria", KODIK_EMBED),
        link("AniBoom", "AniLibria", ANIBOOM_EMBED),
        link("Странный", "AniLibria", "https://example.org/embed/1"),
    ]
    assert [c.player for c in ad.order_candidates(links)] == ["kodik", "aniboom"]
    assert [c.player for c in ad.order_candidates(links, player="aniboom")] == ["aniboom"]


def test_pick_stream_prefers_mp4_on_equal_quality():
    picked = ad.pick_stream(
        result(stream("hls-720", "hls", 720), stream("mp4-720", "mp4", 720))
    )
    assert picked.url == "mp4-720"


def test_pick_stream_respects_max_quality():
    picked = ad.pick_stream(
        result(stream("mp4-1080", "mp4", 1080), stream("mp4-480", "mp4", 480)),
        max_quality=720,
    )
    assert picked.url == "mp4-480"


def test_pick_stream_relaxes_max_quality_when_nothing_matches():
    picked = ad.pick_stream(result(stream("mp4-1080", "mp4", 1080)), max_quality=480)
    assert picked.url == "mp4-1080"


def test_pick_stream_takes_master_only_as_last_resort():
    master = stream("master", "hls")
    assert ad.pick_stream(result(master)).url == "master"
    assert ad.pick_stream(result(master, stream("mp4-360", "mp4", 360))).url == "mp4-360"


def test_pick_stream_without_streams():
    assert ad.pick_stream(result()) is None


# ============================================================
# Сквозной сценарий
# ============================================================


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Подменяет скачивание записью пустого файла; возвращает список вызовов."""

    calls = []

    def fake_download(stream_obj, destination: Path):
        calls.append((stream_obj, Path(destination)))
        Path(destination).write_bytes(b"video")

    monkeypatch.setattr(ad, "download_stream", fake_download)
    return calls


def test_auto_download_titles_downloads_requested_episodes(
    monkeypatch, tmp_path, site_factory, downloads
):
    site_factory(
        [anime("Тайтл")],
        {
            1: [link("Kodik", "AniLibria", KODIK_EMBED)],
            2: [link("Kodik", "AniLibria", KODIK_EMBED)],
        },
    )
    monkeypatch.setattr(
        ad.adc, "extract", lambda url, **kwargs: result(stream("mp4-720", "mp4", 720))
    )

    saved = ad.auto_download_titles([["Тайтл", "1-2", "AniLibria"]], tmp_path)

    assert len(saved) == 2
    assert all(path.exists() for path in saved)
    assert [path.name for path in saved] == [
        "Тайтл_-_Серия_1_-_AniLibria_-_720p.mp4",
        "Тайтл_-_Серия_2_-_AniLibria_-_720p.mp4",
    ]


def test_falls_back_to_next_player_when_first_fails(
    monkeypatch, tmp_path, site_factory, downloads
):
    site_factory(
        [anime("Тайтл")],
        {
            1: [
                link("Kodik", "AniLibria", KODIK_EMBED),
                link("AniBoom", "AniLibria", ANIBOOM_EMBED),
            ]
        },
    )

    def flaky_extract(url, **kwargs):
        if url == KODIK_EMBED:
            raise adc.ExtractionError("плеер сменил разметку")
        return result(stream("hls-1080", "hls", 1080), player="aniboom")

    monkeypatch.setattr(ad.adc, "extract", flaky_extract)

    saved = ad.auto_download_titles([["Тайтл", "1", "AniLibria"]], tmp_path)

    assert len(saved) == 1
    assert saved[0].name.endswith("1080p.mp4")
    assert downloads[0][0].kind is adc.StreamKind.HLS


def test_episode_is_skipped_when_every_player_fails(
    monkeypatch, tmp_path, site_factory, downloads
):
    site_factory([anime("Тайтл")], {1: [link("Kodik", "AniLibria", KODIK_EMBED)]})
    monkeypatch.setattr(
        ad.adc,
        "extract",
        lambda url, **kwargs: (_ for _ in ()).throw(adc.NoStreamsFound("пусто")),
    )

    assert ad.auto_download_titles([["Тайтл", "1"]], tmp_path) == []
    assert downloads == []


def test_existing_file_is_not_redownloaded(monkeypatch, tmp_path, site_factory, downloads):
    site_factory([anime("Тайтл")], {1: [link("Kodik", "AniLibria", KODIK_EMBED)]})
    monkeypatch.setattr(
        ad.adc, "extract", lambda url, **kwargs: result(stream("mp4-720", "mp4", 720))
    )

    existing = tmp_path / "Тайтл_-_Серия_1_-_AniLibria_-_720p.mp4"
    existing.write_bytes(b"already here")

    assert ad.auto_download_titles([["Тайтл", "1", "AniLibria"]], tmp_path) == [existing]
    assert downloads == []
    assert existing.read_bytes() == b"already here"


def test_missing_episode_does_not_break_the_rest(
    monkeypatch, tmp_path, site_factory, downloads
):
    site_factory([anime("Тайтл")], {1: [link("Kodik", "AniLibria", KODIK_EMBED)]})
    monkeypatch.setattr(
        ad.adc, "extract", lambda url, **kwargs: result(stream("mp4-720", "mp4", 720))
    )

    saved = ad.auto_download_titles([["Тайтл", "1-2", "AniLibria"]], tmp_path)

    assert [path.name for path in saved] == ["Тайтл_-_Серия_1_-_AniLibria_-_720p.mp4"]


def test_broken_entry_does_not_stop_the_good_one(
    monkeypatch, tmp_path, site_factory, downloads
):
    site_factory([anime("Тайтл")], {1: [link("Kodik", "AniLibria", KODIK_EMBED)]})
    monkeypatch.setattr(
        ad.adc, "extract", lambda url, **kwargs: result(stream("mp4-720", "mp4", 720))
    )

    saved = ad.auto_download_titles(["мусор", ["Тайтл", "1", "AniLibria"]], tmp_path)

    assert len(saved) == 1


def test_search_failure_is_wrapped_and_not_fatal(monkeypatch, tmp_path, site_factory):
    site = site_factory([anime("Тайтл")], {})

    def failing_search(query, limit=15):
        raise adc.ServiceError("Cloudflare", status=403, url="https://animego.org/")

    monkeypatch.setattr(site, "search", failing_search)

    assert ad.auto_download_titles([["Тайтл", "1"]], tmp_path) == []
    assert site.closed


def test_empty_section_is_skipped(tmp_path):
    assert ad.auto_download_titles([], tmp_path) == []
    assert ad.auto_download_titles(None, tmp_path) == []


def test_unreachable_source_skips_download(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "is_source_reachable", lambda host=None: False)
    assert ad.auto_download_titles([["Тайтл", "1"]], tmp_path) == []


def test_source_host_follows_mirror_setting(monkeypatch):
    monkeypatch.setattr(ad.config, "ANIMEGO_MIRROR", "animego.me", raising=False)
    assert ad.source_host() == "animego.me"
    assert "animego.me" in ad.network_hint()

    monkeypatch.setattr(ad.config, "ANIMEGO_MIRROR", None, raising=False)
    assert ad.source_host() == ad.SOURCE_HOST


def test_client_kwargs_pass_proxy(monkeypatch):
    monkeypatch.setattr(ad.config, "ANIME_DL_PROXY", None, raising=False)
    assert "proxy" not in ad.client_kwargs()

    monkeypatch.setattr(ad.config, "ANIME_DL_PROXY", "socks5://127.0.0.1:1080", raising=False)
    assert ad.client_kwargs()["proxy"] == "socks5://127.0.0.1:1080"


def test_create_site_reports_unreachable_host(monkeypatch):
    monkeypatch.setattr(ad, "is_source_reachable", lambda host=None: False)
    with pytest.raises(ad.AutodownloadError, match="недоступен"):
        ad.create_site()


# ============================================================
# Скачивание
# ============================================================


def test_download_stream_uses_ffmpeg_for_hls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ad, "download_via_ffmpeg", lambda s, d: calls.append((s, d)))
    monkeypatch.setattr(
        ad, "download_episode", lambda *a, **kw: pytest.fail("hls нельзя качать напрямую")
    )

    hls = stream("https://cdn/master.m3u8", "hls", 720)
    ad.download_stream(hls, tmp_path / "out.mp4")
    assert calls == [(hls, tmp_path / "out.mp4")]


def test_download_stream_passes_headers_for_mp4(monkeypatch, tmp_path):
    seen = {}

    def fake_direct(url, destination, headers=None):
        seen.update({"url": url, "headers": headers})

    monkeypatch.setattr(ad, "download_episode", fake_direct)
    ad.download_stream(stream("https://cdn/720.mp4", "mp4", 720), tmp_path / "out.mp4")

    # Без Referer CDN отвечает 403 — заголовки обязаны доехать до загрузчика.
    assert seen["url"] == "https://cdn/720.mp4"
    assert seen["headers"]["Referer"] == "https://example.org/"


def test_download_via_ffmpeg_reports_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(ad.shutil, "which", lambda name: None)
    with pytest.raises(ad.AutodownloadError, match="ffmpeg"):
        ad.download_via_ffmpeg(stream("https://cdn/master.m3u8", "hls"), tmp_path / "out.mp4")


def test_download_via_ffmpeg_cleans_up_after_failure(monkeypatch, tmp_path):
    partial = tmp_path / "out.part.mp4"

    class FailedRun:
        returncode = 1
        stdout = ""
        stderr = "Server returned 403 Forbidden"

    monkeypatch.setattr(ad.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **kw: FailedRun())

    with pytest.raises(ad.AutodownloadError, match="403"):
        ad.download_via_ffmpeg(stream("https://cdn/master.m3u8", "hls"), tmp_path / "out.mp4")
    assert not partial.exists()


def test_download_via_ffmpeg_renames_partial_on_success(monkeypatch, tmp_path):
    output = tmp_path / "out.mp4"
    partial = tmp_path / "out.part.mp4"
    recorded = {}

    class OkRun:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        recorded["args"] = args
        partial.write_bytes(b"video")
        return OkRun()

    monkeypatch.setattr(ad.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ad.subprocess, "run", fake_run)

    ad.download_via_ffmpeg(stream("https://cdn/master.m3u8", "hls"), output)

    assert output.read_bytes() == b"video"
    assert not partial.exists()
    args = recorded["args"]
    assert args[0] == "/usr/bin/ffmpeg"
    assert "-y" in args and "-nostdin" in args
    # Заголовки CDN и путь вывода должны доехать до ffmpeg.
    assert "-headers" in args and str(partial) == args[-1]
