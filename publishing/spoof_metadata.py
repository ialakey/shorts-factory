import random
import subprocess
import datetime
from pathlib import Path


def spoof_metadata(input_path: Path, output_path: Path) -> bool:
    """
    Полностью очищает и подменяет метаданные видео.
    Создаёт новый видеофайл с "ручными" метаданными (CapCut, OBS и т.д.).

    Возвращает True, если ffmpeg отработал и файл действительно создан.
    Вызывающий код обязан проверять результат: без этого можно удалить
    исходный клип, не получив взамен новый.
    """

    # случайная длительность редактирования (в минутах)
    edit_duration_min = random.randint(32, 223)
    edit_duration_str = f"{edit_duration_min} мин"

    # случайная дата "монтажа" (в пределах 1–3 дней назад)
    fake_creation_date = (
            datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 3))
    ).strftime("%Y-%m-%d %H:%M:%S")

    # случайная версия CapCut / OBS
    fake_soft = random.choice([
        "CapCut 8.4.2 (Android)",
        "CapCut 9.0.1 (Windows)",
        "CapCut 10.2.1 (iOS)",
        "OBS Studio 30.1.2",
        "Adobe Premiere Pro 2025.1",
        "DaVinci Resolve 19.0.1"
    ])

    # описание — как будто ручной монтаж
    fake_comment = random.choice([
        f"Edited manually in {fake_soft}, layered transitions, subtitles, LUT filters.",
        f"Project rendered via {fake_soft} after {edit_duration_str} of fine-tuning.",
        f"Montage completed manually in {fake_soft} — color correction, speed ramps, effects.",
        f"Final export using {fake_soft}, manual keyframes and audio sync adjustments."
    ])

    # метаданные для подмены
    metadata_args = [
        "-metadata", f"encoder={fake_soft}",
        "-metadata", f"software={fake_soft}",
        "-metadata", f"description={fake_comment}",
        "-metadata", f"comment={fake_comment}",
        "-metadata", f"creation_time={fake_creation_date}",
        "-metadata", f"editing_duration={edit_duration_str}",
        "-metadata", f"title=Edited Clip",
        "-metadata", f"artist=Independent Creator",
        "-metadata", f"location=Local Project"
    ]

    # основная команда ffmpeg: очистить + пересобрать контейнер
    cmd = [
              "ffmpeg",
              "-y",
              "-i", str(input_path),
              "-map", "0",
              "-map_metadata", "-1",  # полностью удалить старые метаданные
              "-c", "copy",           # не перекодировать, быстро
          ] + metadata_args + [str(output_path)]

    print(f"🧹 Очистка и подмена метаданных: {input_path.name}")

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ ffmpeg не найден в PATH — метаданные не изменены.")
        return False

    if result.returncode != 0 or not output_path.exists():
        stderr_tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
        print(f"❌ ffmpeg не смог обработать {input_path.name} (код {result.returncode}).")
        for line in stderr_tail:
            print(f"   {line}")
        output_path.unlink(missing_ok=True)
        return False

    print(f"✅ Уникальные метаданные установлены ({fake_soft}, {edit_duration_str})")
    return True