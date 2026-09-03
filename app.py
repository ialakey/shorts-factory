import time
import traceback
from infrastructure.check_structure import check_all_channels
from core.channel_processor import process_channel

if __name__ == "__main__":
    start_time = time.time()

    print(f"🔍 Проверяем структуру всех каналов перед запуском ...")

    valid_channels = check_all_channels()
    if not valid_channels:
        print("❌ Нет корректных каналов для обработки.")
        exit(1)

    failed_channels = []

    for channel_name in valid_channels:
        try:
            process_channel(channel_name)
        except KeyboardInterrupt:
            raise
        except Exception:
            failed_channels.append(channel_name)
            print(f"\n❌ Канал {channel_name} завершился с ошибкой:")
            traceback.print_exc()

    if failed_channels:
        print(f"\n⚠️ Каналы с ошибками: {', '.join(failed_channels)}")

    print(f"\n⏱️ Время выполнения: {time.time() - start_time:.2f} секунд")