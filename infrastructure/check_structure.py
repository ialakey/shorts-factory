from pathlib import Path
import yaml
from config import CHANNELS_DIR, REQUIRED_DIRS, DEFAULT_CONFIG


def ensure_config_yaml(channel_name: str, base_path: Path):
    """Проверяет наличие config.yaml, создаёт если отсутствует."""
    config_path = base_path / "config.yaml"

    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG(channel_name), f, allow_unicode=True, sort_keys=False)
        print(f"🆕 Создан config.yaml для {channel_name}")
        return True

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        existing_name = cfg.get("channel_name")
        if existing_name != channel_name:
            print(
                f"⚠️ ВНИМАНИЕ: channel_name в {config_path.name} = '{existing_name}', "
                f"а имя папки — '{channel_name}'. Проверь конфиг!"
            )
    except Exception as e:
        print(f"❌ Ошибка чтения {config_path}: {e}")

    return True


def check_and_prepare_structure(channel_name: str):
    """Проверяет структуру канала, создаёт недостающие папки и config.yaml."""
    base_path = CHANNELS_DIR / channel_name
    created = []

    for rel_path in REQUIRED_DIRS:
        folder = base_path / rel_path
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            created.append(folder)

    if created:
        print(f"📁 Для {channel_name} созданы недостающие папки:")
        for f in created:
            print(f"   ├─ {f.relative_to(base_path)}")

    assets_path = base_path / "assets"
    ok = True
    for sub in ["backgrounds", "musics", "fonts"]:
        subfolder = assets_path / sub
        files = list(subfolder.glob("*"))
        if not files:
            print(f"⚠️ В '{subfolder.relative_to(base_path)}' нет файлов — рекомендуется добавить.")
            ok = False
        else:
            print(f"🎨 {channel_name}: найдено {len(files)} файлов в '{subfolder.relative_to(base_path)}'.")

    ensure_config_yaml(channel_name, base_path)
    return ok


def discover_channels():
    """Возвращает список каналов на основе подпапок в каталоге channels."""
    if not CHANNELS_DIR.exists():
        raise FileNotFoundError(f"❌ Не найдена папка каналов: {CHANNELS_DIR}")

    channels = sorted(
        p.name
        for p in CHANNELS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not channels:
        print("❌ В каталоге channels нет ни одного канала.")

    return channels


def check_all_channels():
    """Проверяет все каналы в каталоге channels и возвращает список валидных."""
    valid_channels = []

    for name in discover_channels():
        print(f"\n🔍 Проверка структуры канала: {name}")
        if check_and_prepare_structure(name):
            valid_channels.append(name)

    return valid_channels