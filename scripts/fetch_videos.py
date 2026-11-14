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

# Бильярд (SPRINT3 B4) — в data/videos/billiard/. Чужие записи: только для внутренней проверки видео-слоя сукна.
BILLIARD = {
    # Реальный стол сверху (вебкамера над любительским столом), 1080p, 38 с: 4 удара, прицеливание кием.
    # Репозиторий louisgeisler/Billard — лицензия в репозитории не указана.
    "louisgeisler_NOdebug.mp4": "louisgeisler/Billard/main/NOdebug.mp4",
}
# Клипы ударов из трансляций (YouTube) из набора FedericoPivotto/billiard-video-analysis (код MIT, видео — чужие
# трансляции): 1024×576, 5 с, разные ракурсы (сверху, под углом).
for _g in (1, 2, 3, 4):
    for _c in (1, 2):
        BILLIARD[f"fp_game{_g}_clip{_c}.mp4"] = (f"FedericoPivotto/billiard-video-analysis/main/dataset/"
                                                 f"game{_g}_clip{_c}/game{_g}_clip{_c}.mp4")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    (out / "billiard").mkdir(exist_ok=True)
    items = list(VIDEOS.items()) + [(f"billiard/{k}", v) for k, v in BILLIARD.items()]
    for name, path in items:
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
