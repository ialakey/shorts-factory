import json
import re
import subprocess
import tempfile
from pathlib import Path

import whisper


def extract_audio_track(video_path: Path) -> Path:
    """
    Извлекает первую аудиодорожку из видео.
    Возвращает путь к временному .wav файлу.
    """

    tmp_audio = Path(tempfile.gettempdir()) / f"{video_path.stem}_audio.wav"

    # --- ffprobe ---
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type:stream_tags",
        "-of", "json", str(video_path)
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True, encoding="utf-8")
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        raise RuntimeError(f"❌ ffprobe вернул некорректный JSON для {video_path.name}")

    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise RuntimeError(f"❌ В файле {video_path.name} нет аудиодорожек")

    selected_index = audio_streams[0]["index"]

    # --- извлекаем ---
    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", f"0:{selected_index}",
        "-vn",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(tmp_audio)
    ]

    result = subprocess.run(cmd_ffmpeg, capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0 or not tmp_audio.exists():
        print("⚠️ ffmpeg не смог извлечь выбранную дорожку, пробуем первую аудиодорожку...")
        # fallback — первая дорожка
        first_index = audio_streams[0]["index"]
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-map", f"0:{first_index}",
            "-vn",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(tmp_audio)
        ]
        subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not tmp_audio.exists():
        raise RuntimeError(f"❌ ffmpeg не создал аудиофайл: {tmp_audio}")

    return tmp_audio


_H1_OPEN_RE = re.compile(r"<\s*h1\s*>", re.IGNORECASE)
_H1_CLOSE_RE = re.compile(r"<\s*/\s*h1\s*>", re.IGNORECASE)
_HL_TOKEN_RE = re.compile(r"<\s*([\\/]*)\s*hl\s*>", re.IGNORECASE)
_HL_OPEN_RE = re.compile(r"<\s*hl\s*>", re.IGNORECASE)
_HL_CLOSE_RE = re.compile(r"<\s*[\\/]\s*hl\s*>", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.,!?:;\-—–()\[\]\"'«»…]+")


def _caps_and_strip_punct_preserve_highlight(text: str) -> str:
    """Convert text to CAPS, remove punctuation, keep <hl> tags intact."""

    def _clean_chunk(chunk: str) -> str:
        chunk = _PUNCT_RE.sub("", chunk)
        return chunk.upper()

    parts = re.split(r"(<\s*hl\s*>.*?<\s*[\\/]\s*hl\s*>)", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned: list[str] = []
    for part in parts:
        if not part:
            continue
        if _HL_OPEN_RE.match(part):
            inner = _HL_OPEN_RE.sub("", part)
            inner = _HL_CLOSE_RE.sub("", inner)
            cleaned_inner = _clean_chunk(inner)
            cleaned.append(f"<hl>{cleaned_inner}</hl>")
        else:
            cleaned.append(_clean_chunk(part))
    result = "".join(cleaned)
    result = re.sub(r"\s+", " ", result)
    return result.strip()


def sanitize_subtitle_text(text: str) -> str:
    """Нормализует текст субтитров, сохраняя emoji и <hl>-теги."""

    if not text:
        return ""

    cleaned = str(text)
    cleaned = _H1_OPEN_RE.sub("<hl>", cleaned)
    cleaned = _H1_CLOSE_RE.sub("</hl>", cleaned)
    cleaned = _HL_TOKEN_RE.sub(lambda m: "</hl>" if "/" in (m.group(1) or "") else "<hl>", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    return _caps_and_strip_punct_preserve_highlight(cleaned)


def rebuild_words_from_clean_text(segment: dict, clean_text: str) -> None:
    """Пересобирает список ``words`` после GPT-очистки, чтобы сохранить <hl> и emoji.

    Whisper отдаёт пометку ``words`` без наших правок, поэтому добавленные GPT теги
    и эмодзи не попадали в чанки. Мы равномерно распределяем слова по таймкоду
    сегмента, чтобы split_segment_by_words вернул тот же текст, который мы отдали
    в ``segment["text"]``.
    """

    if not isinstance(segment, dict):
        return

    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    duration = max(end - start, 0.01)

    # `\S+` в одной альтернации с тегами не работает: в "<hl>ВАЖНО</hl>" он
    # жадно съедал "ВАЖНО</hl>" целиком, закрывающий тег терялся, и подсветка
    # не выключалась до конца сегмента. Сначала вырезаем теги, потом бьём
    # остаток по пробелам.
    tokens = []
    for part in re.split(r"(<\s*[\\/]?\s*hl\s*>)", clean_text, flags=re.IGNORECASE):
        if not part:
            continue
        if re.fullmatch(r"<\s*[\\/]?\s*hl\s*>", part, flags=re.IGNORECASE):
            tokens.append(part)
        else:
            tokens.extend(part.split())
    if not tokens:
        segment["words"] = []
        return

    step = duration / len(tokens)
    words = []
    for idx, token in enumerate(tokens):
        w_start = start + idx * step
        w_end = min(end, w_start + step)
        words.append({"word": token, "start": w_start, "end": w_end})

    segment["words"] = words


def select_whisper_model(subtitles_cfg) -> str:
    """Возвращает название модели Whisper с учётом настроек улучшения."""

    subtitles_cfg = subtitles_cfg or {}
    return subtitles_cfg.get("whisper_model") or "base"


def transcribe_video(video_path, cfg=None, api_key=None):
    """
    Транскрибирует первую аудиодорожку из видео.
    """
    print("🎧 Извлечение аудио дорожки...")
    audio_path = extract_audio_track(Path(video_path))

    subtitles_cfg = (cfg or {}).get("subtitles", {})
    whisper_model = select_whisper_model(subtitles_cfg)
    print(f"🧠 Распознавание речи через Whisper ('{whisper_model}')...")

    model = whisper.load_model(whisper_model)
    transcribe_kwargs = {}
    language = str(subtitles_cfg.get("language", "")).strip() or None
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(str(audio_path), **transcribe_kwargs)

    try:
        audio_path.unlink()
    except Exception:
        pass

    return result
