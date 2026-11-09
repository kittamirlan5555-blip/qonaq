"""Скачивает тестовые видео в data/videos/ из публичных GitHub-репозиториев.

Видео не наши: только для внутреннего тестирования, не публиковать и не коммитить.
Запуск: python scripts/fetch_videos.py
"""
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/"
VIDEOS = {
    # Реальная CCTV-запись ресторана, угол под потолком, OSD с датой, 1080p, ~60 с. Основной тест.
    "cctv_restaurant_corner.mp4": "Ghaaidda/Restaurant-Footage-Analytics/master/data/footage-short-ver.mp4",
    # Реальная CCTV уличной веранды кафе (зонты, официанты, прохожие), ~88 с.
    "cctv_terrace_cafe.mov": "joelibaceta/restaurant-vision-demo/main/data/video.mov",
    # Вид сверху на кафе с гостями, ~10 с (стоковое, короткое) — проверка автопредложения столов.
    "topdown_cafe_people.mp4": "aaryan-kamdar/CafeVision-RealTimeTableOccupancy/main/Results/RestaurantTest.mp4",
    # Пустое кафе сверху, ~8 с — эталон «все столы свободны».
    "topdown_cafe_empty.mp4": "aaryan-kamdar/CafeVision-RealTimeTableOccupancy/main/datasets/videos/RestaurantTestEmpty.mp4",
    # Фудкорт, ~70 с. ВНИМАНИЕ: на кадрах уже нарисованы чужие зоны — использовать только для проверки трекинга.
    "foodcourt_preannotated.mp4": "MohammedAlown/ai-restaurant-monitor/main/docs/demo/food-court-detection.mp4",
}


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in VIDEOS.items():
        dst = out / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"skip {name}")
            continue
        print(f"get  {name}")
        try:
            urllib.request.urlretrieve(RAW + path, dst)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
    print("done ->", out)


if __name__ == "__main__":
    main()
