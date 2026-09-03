from pathlib import Path


def find_input_videos(input_dir: Path):
    exts = {".mp4", ".mov", ".mkv"}
    videos = [p for p in input_dir.glob("*") if p.suffix.lower() in exts]
    return videos
