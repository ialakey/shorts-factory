"""Этап 0: подготовка структуры каналов."""

from __future__ import annotations

import yaml

from infrastructure import check_structure


def _use_tmp_channels(monkeypatch, tmp_path):
    channels = tmp_path / "channels"
    channels.mkdir()
    monkeypatch.setattr(check_structure, "CHANNELS_DIR", channels)
    return channels


def test_creates_missing_dirs_and_config(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    (channels / "Alpha").mkdir()

    ok = check_structure.check_and_prepare_structure("Alpha")

    base = channels / "Alpha"
    for rel in check_structure.REQUIRED_DIRS:
        assert (base / rel).is_dir(), f"не создана папка {rel}"
    assert (base / "config.yaml").exists()
    # ассеты пустые → канал помечается как «неполный»
    assert ok is False


def test_generated_config_is_valid_yaml(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    (channels / "Beta").mkdir()

    check_structure.check_and_prepare_structure("Beta")

    cfg = yaml.safe_load((channels / "Beta" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["channel_name"] == "Beta"
    for section in ("video", "video_effects", "music", "subtitles"):
        assert section in cfg, f"в дефолтном конфиге нет секции {section}"


def test_existing_config_is_not_overwritten(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    base = channels / "Gamma"
    base.mkdir()
    config_path = base / "config.yaml"
    config_path.write_text("channel_name: Gamma\ncustom: 42\n", encoding="utf-8")

    check_structure.ensure_config_yaml("Gamma", base)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["custom"] == 42


def test_channel_with_assets_is_valid(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    base = channels / "Delta"
    for sub in ("assets/backgrounds", "assets/musics", "assets/fonts"):
        (base / sub).mkdir(parents=True)
        (base / sub / "placeholder.bin").write_bytes(b"x")

    assert check_structure.check_and_prepare_structure("Delta") is True


def test_discover_channels_skips_hidden(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    (channels / "Visible").mkdir()
    (channels / ".hidden").mkdir()
    (channels / "notes.txt").write_text("x", encoding="utf-8")

    assert check_structure.discover_channels() == ["Visible"]


def test_check_all_channels_returns_only_prepared(monkeypatch, tmp_path):
    channels = _use_tmp_channels(monkeypatch, tmp_path)
    for sub in ("assets/backgrounds", "assets/musics", "assets/fonts"):
        (channels / "Full" / sub).mkdir(parents=True)
        (channels / "Full" / sub / "a.bin").write_bytes(b"x")
    (channels / "Empty").mkdir()

    assert check_structure.check_all_channels() == ["Full"]
