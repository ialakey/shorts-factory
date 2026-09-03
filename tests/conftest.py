"""Общие заглушки и фикстуры для тестов пайплайна.

Тесты обязаны быть герметичными:

* **никакого OpenAI** — ключ подменяется до импорта ``config.py``, все вызовы
  ``openai.ChatCompletion.create`` в тестах замоканы;
* **никакого Whisper** — модуль ``whisper`` всегда подменяется заглушкой,
  иначе тесты тянули бы torch и качали модели;
* **никакой сети** — Kodik и Telegram замоканы;
* необязательные тяжёлые зависимости (``mediapipe``, ``anime_parsers_ru``)
  могут отсутствовать: код обязан деградировать, а не падать.
"""

from __future__ import annotations

import copy
import importlib.util
import shutil
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# config.py падает на импорте без ключа, а .env разработчика содержит настоящий
# ключ — принудительно подменяем его ДО любого импорта проекта, чтобы
# случайный вызов не ушёл в реальный API. load_dotenv() уже выставленные
# переменные не перезаписывает.
os.environ["OPENAI_API_KEY"] = "test-openai-key"


# ============================================================
# Заглушки тяжёлых зависимостей
# ============================================================

DEFAULT_FAKE_TRANSCRIPT = {
    "text": "hello world this is a test",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 0.6,
            "text": " hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.3},
                {"word": "world", "start": 0.3, "end": 0.6},
            ],
        },
        {
            "id": 1,
            "start": 0.6,
            "end": 1.2,
            "text": " this is a test",
            "words": [
                {"word": "this", "start": 0.6, "end": 0.75},
                {"word": "is", "start": 0.75, "end": 0.9},
                {"word": "a", "start": 0.9, "end": 1.05},
                {"word": "test", "start": 1.05, "end": 1.2},
            ],
        },
    ],
}


def _install_whisper_stub() -> types.ModuleType:
    """Подменяет ``whisper`` детерминированной заглушкой.

    Заглушка ставится всегда — даже если настоящий whisper установлен, — чтобы
    тесты не грузили модели и давали воспроизводимый транскрипт.
    """

    stub = types.ModuleType("whisper")
    stub.transcript = copy.deepcopy(DEFAULT_FAKE_TRANSCRIPT)
    stub.loaded_models = []
    stub.calls = []

    class _FakeModel:
        def __init__(self, name: str):
            self.name = name

        def transcribe(self, audio, **kwargs):
            stub.calls.append(
                {"model": self.name, "audio": str(audio), "kwargs": dict(kwargs)}
            )
            return copy.deepcopy(stub.transcript)

    def load_model(name="base", *args, **kwargs):
        stub.loaded_models.append(name)
        return _FakeModel(name)

    stub.load_model = load_model
    stub.FakeModel = _FakeModel
    sys.modules["whisper"] = stub
    return stub


WHISPER_STUB = _install_whisper_stub()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not _module_available("anime_parsers_ru"):
    # В CI парсер Kodik не ставим: он нужен только для сетевого этапа загрузки,
    # но импортируется на уровне модуля ingestion/autodownload.py.
    _kodik_stub = types.ModuleType("anime_parsers_ru")

    class _KodikParser:  # pragma: no cover - вызывается только в сетевом коде
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "anime_parsers_ru недоступен в тестовом окружении"
            )

    _kodik_stub.KodikParser = _KodikParser
    sys.modules["anime_parsers_ru"] = _kodik_stub


@pytest.fixture(autouse=True)
def whisper_stub():
    """Сбрасывает состояние заглушки Whisper перед каждым тестом."""

    WHISPER_STUB.transcript = copy.deepcopy(DEFAULT_FAKE_TRANSCRIPT)
    WHISPER_STUB.loaded_models.clear()
    WHISPER_STUB.calls.clear()
    yield WHISPER_STUB
    WHISPER_STUB.transcript = copy.deepcopy(DEFAULT_FAKE_TRANSCRIPT)


@pytest.fixture(autouse=True)
def no_real_openai(monkeypatch):
    """Страховка: любой незамоканный вызов OpenAI падает, а не уходит в сеть."""

    import openai

    def _forbidden(*args, **kwargs):  # pragma: no cover - срабатывает при ошибке теста
        raise AssertionError(
            "Тест попытался вызвать настоящий OpenAI API — замокайте "
            "openai.ChatCompletion.create"
        )

    monkeypatch.setattr(openai.ChatCompletion, "create", _forbidden)


# ============================================================
# Окружение: ffmpeg и тестовые медиафайлы
# ============================================================


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        pytest.skip("ffmpeg не найден в PATH")
    return exe


@pytest.fixture(scope="session")
def ffprobe_bin() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        pytest.skip("ffprobe не найден в PATH")
    return exe


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory, ffmpeg_bin) -> Path:
    """Короткий синтетический ролик со звуком (2 с, 320x180, 24 fps)."""

    path = tmp_path_factory.mktemp("media") / "sample_episode.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def sample_music(tmp_path_factory, ffmpeg_bin) -> Path:
    """Трек для этапа ``music``."""

    path = tmp_path_factory.mktemp("music") / "track.wav"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=3",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def project_font(repo_root) -> Path:
    """Реальный шрифт из репозитория (рендер без шрифта невозможен)."""

    fonts = [
        p
        for p in (repo_root / "assets" / "fonts").glob("*.ttf")
        if "NotoColorEmoji" not in p.name
    ]
    if not fonts:
        pytest.skip("в assets/fonts нет обычного .ttf шрифта")
    return fonts[0]


@pytest.fixture
def channel_dir(tmp_path, project_font, sample_music) -> Path:
    """Готовая структура канала с ассетами, как её ждёт make_clips."""

    from PIL import Image

    channel = tmp_path / "TestChannel"
    for sub in ("input_videos", "output_clips", "assets/backgrounds",
                "assets/musics", "assets/fonts", "logs"):
        (channel / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy(project_font, channel / "assets" / "fonts" / project_font.name)
    shutil.copy(sample_music, channel / "assets" / "musics" / sample_music.name)
    Image.new("RGB", (216, 384), (18, 18, 32)).save(
        channel / "assets" / "backgrounds" / "bg.png"
    )
    return channel


@pytest.fixture
def render_config(repo_root) -> dict:
    """Маленький, но полный конфиг канала для быстрых рендер-тестов."""

    emoji_font = repo_root / "assets" / "fonts" / "NotoColorEmoji.ttf"
    return {
        "channel_name": "TestChannel",
        "debug": False,
        "video": {"width": 216, "height": 384},
        "background": {
            "use_transparent_window": False,
            "use_blurred_background": True,
            "blur_strength": 21,
            "background_scale": 1.0,
            "width": 216,
            "height": 384,
            "fit_mode": "contain",
            "video_width": 216,
            "video_height": 384,
            "video_x": 0,
            "video_y": 0,
        },
        "dynamic_shorts": {
            "analysis_fps": 4,
            "min_face_ratio": 0.06,
            "min_face_confidence": 0.75,
            "display_width": 216,
            "display_height": 384,
        },
        "watermark": {
            "text": "test",
            "color": "#FFFFFF",
            "font_size": 16,
            "opacity": 0.5,
            "position": ["center", "bottom"],
            "padding_x": 0,
            "padding_y": 6,
        },
        "video_effects": {
            "mirror": True,
            "color_saturation": 1.0,
            "gamma": 1.0,
            "contrast": 1.0,
            "resize": 1.0,
            "frame_rate": 24,
            "speed": 1.0,
        },
        "audio_enhancer": {
            "enabled": True,
            "threshold": 0.32,
            "ratio": 4.5,
            "soft_knee": 0.1,
            "makeup_gain": 1.4,
            "noise_floor": 0.02,
            "output_ceiling": 0.98,
        },
        "music": {"volume": 0.07},
        "title": {
            "font_size": 20,
            "max_width": 200,
            "max_words_per_line": 3,
            "emoji_font_path": str(emoji_font),
        },
        "subtitles": {
            "font_size": 18,
            "color": "#FFF8D6",
            "stroke_color": "#000000",
            "stroke_width": 2,
            "align": "center",
            "bottom_padding": 20,
            "max_width": 190,
            "words_per_chunk": 1,
            "language": "en",
            "improve_transcript_quality": False,
            "whisper_model": "base",
            "subtitle_mode": "normal",
            "emoji_font_path": str(emoji_font),
        },
        "gpt": {
            "model": "test-model",
            "min_time": 1,
            "max_time": 2,
            "min_count": 1,
            "max_count": 1,
            "prompt": "{transcript_text}",
        },
    }


@pytest.fixture
def single_moment() -> dict:
    """Один короткий момент — ровно то, что возвращает analyze_moment."""

    return {
        "moment_1": {
            "title": "TEST CLIP",
            "duration": 1.2,
            "segments": {"segment_1": {"start": 0.2, "end": 1.4}},
        }
    }
