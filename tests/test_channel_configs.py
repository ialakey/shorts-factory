"""Конфиги каналов в репозитории должны оставаться рабочими.

Здесь ловятся опечатки, из-за которых пайплайн падает уже в проде: неизвестный
этап в ``pipeline``, сломанный шаблон промпта, отсутствующий шрифт эмодзи.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import CHANNELS_DIR, DEFAULT_CONFIG, REQUIRED_DIRS
from core.channel_processor import POST_CLIP_STAGES

#: Все этапы, которые умеет запускать пайплайн.
KNOWN_STAGES = {
    "kodik_download",
    "autodownload",  # legacy-имя kodik_download
    "transcribe_video",
    "analyze_moment",
    "make_clips",
    "dynamic_shorts",
    "watermark",
    "enhance_audio",
    "video_effects",
    "speed",
    "music",
    "subtitles",
    "title",
    "spoof_metadata",
    "telegram_notify",
}

GPT_PROMPT_PLACEHOLDERS = {
    "transcript_text": "",
    "audio": "{}",
    "face": "{}",
    "visual": "{}",
    "tempo": "{}",
    "emotion": "{}",
    "hooks": "{}",
    "candidates": "[]",
    "tone": "",
    "platforms": "",
    "audience_age": "",
    "min_count": 2,
    "max_count": 3,
    "min_time": 28,
    "max_time": 43,
}

SUBTITLE_PROMPT_PLACEHOLDERS = {
    "text": "исходный текст",
    "target_language": "ru",
    "language_instructions": "",
}


def channel_configs():
    if not CHANNELS_DIR.exists():
        return []
    return sorted(CHANNELS_DIR.glob("*/config.yaml"))


CONFIGS = channel_configs()
config_param = pytest.mark.parametrize(
    "config_path", CONFIGS, ids=[p.parent.name for p in CONFIGS]
)


@pytest.mark.skipif(not CONFIGS, reason="в репозитории нет каналов")
class TestRepositoryChannelConfigs:
    @config_param
    def test_config_is_valid_yaml(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)

    @config_param
    def test_channel_name_matches_folder(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg.get("channel_name") == config_path.parent.name

    @config_param
    def test_pipeline_contains_only_known_stages(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pipeline = cfg.get("pipeline") or []

        assert isinstance(pipeline, list) and pipeline, "pipeline пуст"
        unknown = set(pipeline) - KNOWN_STAGES
        assert not unknown, f"неизвестные этапы в pipeline: {sorted(unknown)}"
        assert len(pipeline) == len(set(pipeline)), "этапы дублируются"

    @config_param
    def test_sections_required_by_make_clips_exist(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pipeline = set(cfg.get("pipeline") or [])

        # make_clips обращается к этим секциям напрямую — их отсутствие = KeyError
        if pipeline & POST_CLIP_STAGES:
            for section in ("video", "video_effects", "music", "subtitles"):
                assert section in cfg, f"нет обязательной секции {section}"

    @config_param
    def test_video_resolution_is_vertical(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        video = cfg.get("video") or {}
        if not video:
            pytest.skip("секция video не задана")

        assert int(video["width"]) > 0 and int(video["height"]) > 0
        assert int(video["height"]) >= int(video["width"]), "ожидается вертикальный формат"

    @config_param
    def test_gpt_prompt_template_renders(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        prompt = (cfg.get("gpt") or {}).get("prompt")
        if not prompt:
            pytest.skip("gpt.prompt не задан")

        prompt.format(**GPT_PROMPT_PLACEHOLDERS)  # не должно бросать KeyError/IndexError

    @config_param
    def test_gpt_duration_and_count_ranges_are_sane(self, config_path: Path):
        gpt = (yaml.safe_load(config_path.read_text(encoding="utf-8")).get("gpt") or {})
        if not gpt:
            pytest.skip("секция gpt не задана")

        assert 0 < float(gpt["min_time"]) <= float(gpt["max_time"])
        assert 0 < int(gpt["min_count"]) <= int(gpt["max_count"])

    @config_param
    def test_subtitle_prompts_render(self, config_path: Path):
        subtitles = (yaml.safe_load(config_path.read_text(encoding="utf-8")).get("subtitles") or {})
        for key in ("prompt", "meta_ad_prompt"):
            template = subtitles.get(key)
            if template:
                template.format(**SUBTITLE_PROMPT_PLACEHOLDERS)

    @config_param
    def test_referenced_fonts_exist(self, config_path: Path, repo_root: Path):
        subtitles = (yaml.safe_load(config_path.read_text(encoding="utf-8")).get("subtitles") or {})
        emoji_font = subtitles.get("emoji_font_path")
        if not emoji_font or Path(emoji_font).is_absolute():
            pytest.skip("шрифт эмодзи не задан относительным путём")

        assert (repo_root / emoji_font).exists(), f"нет файла шрифта: {emoji_font}"

    @config_param
    def test_meta_ad_mode_has_its_prompt(self, config_path: Path):
        subtitles = (yaml.safe_load(config_path.read_text(encoding="utf-8")).get("subtitles") or {})
        if str(subtitles.get("subtitle_mode", "normal")).lower() == "meta_ad":
            assert subtitles.get("meta_ad_prompt"), "subtitle_mode=meta_ad без meta_ad_prompt"


class TestDefaultConfig:
    def test_is_yaml_serialisable(self):
        dumped = yaml.safe_dump(DEFAULT_CONFIG("Demo"), allow_unicode=True, sort_keys=False)
        assert yaml.safe_load(dumped)["channel_name"] == "Demo"

    def test_contains_sections_used_by_make_clips(self):
        cfg = DEFAULT_CONFIG("Demo")
        for section in ("video", "video_effects", "music", "subtitles"):
            assert section in cfg

    def test_debug_is_off_by_default(self):
        assert DEFAULT_CONFIG("Demo")["debug"] is False

    def test_default_subtitle_prompt_renders(self):
        prompt = DEFAULT_CONFIG("Demo")["subtitles"].get("prompt")
        if prompt:
            prompt.format(**SUBTITLE_PROMPT_PLACEHOLDERS)


class TestStageInventory:
    def test_post_clip_stages_are_known(self):
        assert POST_CLIP_STAGES <= KNOWN_STAGES

    def test_required_dirs_cover_pipeline_needs(self):
        for needed in ("input_videos", "output_clips", "assets/musics", "assets/fonts"):
            assert needed in REQUIRED_DIRS
