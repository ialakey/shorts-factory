"""Utilities for cleaning Whisper transcripts with GPT models."""

import re
import time
from typing import Optional

import openai
from openai.error import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ServiceUnavailableError,
    Timeout,
)


_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0001F1E6-\U0001F1FF]"
)
_HL_TOKEN_RE = re.compile(r"<\s*([\\/]*)\s*hl\s*>", re.IGNORECASE)
_HL_OPEN_RE = re.compile(r"<\s*hl\s*>", re.IGNORECASE)
_HL_CLOSE_RE = re.compile(r"<\s*[\\/]\s*hl\s*>", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.,!?:;\-—–()\[\]\"'«»…]+")


def _normalise_highlight_tags(text: str) -> str:
    """Unify highlight tags to <hl>...</hl> so renderer keeps red styling."""

    def _replace(match: re.Match) -> str:
        prefix = (match.group(1) or "").strip()
        return "</hl>" if "/" in prefix else "<hl>"

    text = re.sub(r"<\s*h1\s*>", "<hl>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*h1\s*>", "</hl>", text, flags=re.IGNORECASE)
    text = _HL_TOKEN_RE.sub(_replace, text)
    return text


def _caps_and_strip_punct_preserve_highlight(text: str) -> str:
    """Uppercase text, drop punctuation, keep highlight wrappers and emoji."""

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


def _log_subtitle_preview(clean_text: str) -> None:
    """Print where GPT suggests emojis and highlight spans for easier QA."""

    highlights = re.findall(r"<hl>(.*?)</hl>", clean_text, flags=re.IGNORECASE)
    emojis = _EMOJI_RE.findall(clean_text)
    print(f"💡 GPT-подсказка по субтитрам: {clean_text}")
    if highlights:
        print(f"   🔴 Выделено: {', '.join([h.strip() for h in highlights if h.strip()])}")
    if emojis:
        print(f"   🙂 Эмодзи: {' '.join(emojis)}")


def clean_and_correct_text(
    text: str,
    *,
    api_key: str,
    model: str = "gpt-5.1-2025-11-13",
    prompt_template: Optional[str] = None,
    meta_ad_prompt_template: Optional[str] = None,
    subtitle_mode: str = "normal",
    language: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
    log_preview: bool = False,
) -> str:
    """Return a cleaned version of ``text`` using an OpenAI chat model.

    The function mirrors the retry strategy used in ``gpt_analyzer`` to keep the
    behaviour consistent across the project.
    """

    if not text:
        return ""

    openai.api_key = api_key

    target_language = (language or "").strip()

    default_prompt = (
        "Ты редактор субтитров. Твой ввод — черновой текст после распознавания речи.\n"
        "Нужно почистить текст, исправить орфографию и удалить мусорные символы.\n"
        "Правила:\n"
        "1. Сохрани смысл и порядок слов.\n"
        "2. Удали повторяющиеся буквы, бессмысленные вставки и спецсимволы.\n"
        "3. Не добавляй новые предложения, только исправляй существующие.\n"
        "4. Добавь 1 уместное эмодзи, если оно действительно подходит по эмоциональному контексту.\n"
        "   - Эмодзи НЕ должно появляться в каждом предложении.\n"
        "   - Минимальная дистанция: эмодзи можно ставить только если от предыдущего прошло > 7 слов.\n"
        "   - Эмодзи ставится отдельно через пробел.\n"
        "   Используй эмоции:\n"
        "     - ярость: 😤🔥😡\n"
        "     - шок: 😳🤯😱😮\n"
        "     - грусть: 😢💔😭\n"
        "     - любовь/романтика: ❤️😍🥰😘\n"
        "     - эпик/сила: 🔥😎💀\n"
        "     - тёмные моменты: 😈💀🖤☠️\n"
        "5. Обязательно выделяй только одно самое важное слово во всём предложении.\n"
        "   - Выделение делается тегом <hl>...</hl>.\n"
        "   - Не выделяй фразы или несколько слов подряд — только ОДНО ключевое слово.\n"
        "   - Выбирай действительно смысловой центр фразы.\n"
        "6. Верни только финальный исправленный текст без пояснений.\n"
        "{language_instructions}\n"
        "Текст:\n{text}\n"
    )

    language_instructions = ""
    if target_language:
        language_instructions = (
            f"7. Верни результат на языке: {target_language}.\n"
            "8. Если исходный текст на другом языке, бережно переведи его, сохранив смысл и стиль."
        )

    normalized_mode = str(subtitle_mode or "normal").strip().lower()
    if normalized_mode == "meta_ad":
        template = (meta_ad_prompt_template or "").strip()
        if not template:
            raise ValueError(
                "❌ Для subtitles.subtitle_mode='meta_ad' нужно задать subtitles.meta_ad_prompt в config.yaml"
            )
    else:
        template = prompt_template or default_prompt
    try:
        prompt = template.format(
            text=text,
            target_language=target_language,
            language_instructions=language_instructions,
        )
    except Exception:
        prompt = f"{template.rstrip()}\n\nТекст:\n{text}\n"

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            clean_text = response["choices"][0]["message"]["content"].strip()
            clean_text = _normalise_highlight_tags(clean_text)
            clean_text = _caps_and_strip_punct_preserve_highlight(clean_text)
            if log_preview:
                _log_subtitle_preview(clean_text)
            return clean_text
        except AuthenticationError as exc:
            raise ValueError(
                "❌ Ошибка авторизации: токен OpenAI недействителен или истёк. Проверь OPENAI_API_KEY."
            ) from exc
        except (APIConnectionError, Timeout, ConnectionError) as exc:
            last_error = exc
            print(
                f"⚠️ Потеря соединения при очистке субтитров (попытка {attempt}/{max_retries}): {exc}"
            )
            time.sleep(5 * attempt)
            continue
        except (APIError, ServiceUnavailableError) as exc:
            last_error = exc
            code = getattr(exc, "http_status", None)
            if code == 502 or "502" in str(exc):
                print(
                    "⚠️ OpenAI вернул 502 при очистке субтитров, повтор через 15 секунд... "
                    f"(попытка {attempt}/{max_retries})"
                )
                time.sleep(15)
                continue
            raise ValueError(f"❌ Ошибка OpenAI при очистке субтитров ({code}): {exc}")
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Неожиданная ошибка при очистке субтитров: {exc}")
            time.sleep(5 * attempt)
            continue

    if last_error:
        raise RuntimeError(
            "🚫 Не удалось очистить текст субтитров через OpenAI API после нескольких попыток."
        ) from last_error
    raise RuntimeError(
        "🚫 Не удалось очистить текст субтитров через OpenAI API после нескольких попыток."
    )
