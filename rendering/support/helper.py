import cv2
import numpy as np
import re
import unicodedata


def safe_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)

    filename = re.sub(r'[<>:"/\\|?*«»]', "_", filename)

    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"_+", "_", filename)

    filename = filename.strip("._")

    return filename[:150]


def split_segment_by_words(segment, max_words=1):
    """Разбивает сегмент Whisper на чанки по словам."""
    words = segment.get("words", [])
    if not words:
        return []

    highlight_on = False
    processed = []
    for token in words:
        raw = str(token.get("word", "")).strip()
        if not raw:
            continue
        if re.match(r"<\s*hl\s*>", raw, flags=re.IGNORECASE):
            highlight_on = True
            continue
        if re.match(r"<\s*[\\/]\s*hl\s*>", raw, flags=re.IGNORECASE):
            highlight_on = False
            continue
        processed.append((raw, highlight_on, token.get("start", 0), token.get("end", 0)))

    if not processed:
        return []

    max_words = max(int(max_words), 1)

    def _wrap(words_batch, highlighted):
        joined = " ".join(words_batch)
        return f"<hl>{joined}</hl>" if highlighted else joined

    def _build_text(batch):
        parts = []
        buffer = []
        buffer_highlight = None
        for word, highlighted, *_ in batch:
            if buffer_highlight is None:
                buffer_highlight = highlighted
            if highlighted != buffer_highlight:
                parts.append(_wrap(buffer, buffer_highlight))
                buffer = [word]
                buffer_highlight = highlighted
            else:
                buffer.append(word)
        if buffer:
            parts.append(_wrap(buffer, buffer_highlight))
        return " ".join(parts)

    chunks = []
    for i in range(0, len(processed), max_words):
        batch = processed[i : i + max_words]
        start_time = batch[0][2]
        end_time = batch[-1][3]
        text = _build_text(batch).strip()
        if text:
            chunks.append({"text": text, "start": start_time, "end": end_time})
    return chunks


def get_transparent_box(image_path: str):
    """Определяет координаты прозрачного прямоугольника на PNG."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"❌ Файл не найден: {image_path}")
    if img.shape[2] < 4:
        raise ValueError("❌ У изображения нет альфа-канала (прозрачности)")

    alpha = img[:, :, 3]
    transparent_mask = alpha == 0
    coords = cv2.findNonZero(transparent_mask.astype(np.uint8))
    if coords is None:
        raise ValueError("⚠️ Прозрачных областей не найдено.")

    x, y, w, h = cv2.boundingRect(coords)
    print(f"🟩 Прозрачное окно найдено: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h, img.shape[1], img.shape[0]