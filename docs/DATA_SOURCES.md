# Источники данных и моделей (атрибуция CC BY 4.0 / Apache-2.0)

Все датасеты ниже опубликованы на Roboflow Universe под лицензией **CC BY 4.0** — использование, в том числе
коммерческое, разрешено при указании авторства. Копии архивов (формат YOLOv8) лежат в релизе `data-v1`
этого репозитория и скачиваются `python scripts/fetch_datasets.py` в `data/datasets/<slug>/`.
В git сами изображения не коммитятся.

## Датасеты

| slug | Набор (версия) | Автор (workspace Roboflow) | Ссылка | Лицензия | Используется |
|---|---|---|---|---|---|
| `smart_restaurant_a` | AI Smart Restaurant Surveillance v3 | wassimas-workspace | https://universe.roboflow.com/wassimas-workspace/ai-smart-restaurant-surveillance-veh3i-44wnt/dataset/3 | CC BY 4.0 | роли, зал A |
| `restaurant_b` | restaurant v3 | test-mlher | https://universe.roboflow.com/test-mlher/restaurant-xdxm2/dataset/3 | CC BY 4.0 | роли, зал B |
| `ramada_c` | Ramada (ramada-bytez) v3 | akash-tt3fk | https://universe.roboflow.com/akash-tt3fk/ramada-bytez/dataset/3 | CC BY 4.0 | роли, зал C (только Waiter; гости добраны детектором) |
| `worker_customer_d` | Worker_Customer_Detection v5 | fyptrolleytracking | https://universe.roboflow.com/fyptrolleytracking/worker_customer_detection/dataset/5 | CC BY 4.0 | роли, зал D (слабый, не ресторан) |
| `dirty_table` | Dirty Table v1 | louis-vd3t4 | https://universe.roboflow.com/louis-vd3t4/dirty-table/dataset/1 | CC BY 4.0 | столы |
| `clean_table` | Clean Table v4 | rohith-vr9nb | https://universe.roboflow.com/rohith-vr9nb/clean-table/dataset/4 | CC BY 4.0 | столы |
| `table_detection` | Table Detection v1 | varun-qlsfy | https://universe.roboflow.com/varun-qlsfy/table-detection-gomjy/dataset/1 | CC BY 4.0 | столы, только внешний тест |
| `table_cleanliness` | Table Cleanliness Detection v4 | — (не скачивался в W1) | — | CC BY 4.0 (по описанию владельца) | **не используется** (один домашний стол, мусорные имена классов) |

Авторы и ссылки взяты из `data.yaml` / `README.roboflow.txt` внутри каждого архива.

## Модели

| Модель | Источник | Лицензия | Где используется |
|---|---|---|---|
| YOLOX-S/M (ONNX) | https://github.com/Megvii-BaseDetection/YOLOX (релиз 0.1.1rc0) | Apache-2.0 | детектор людей/столов/посуды |
| MobileNetV2 ImageNet (`image_classification_mobilenetv2_2022apr.onnx`) | https://github.com/opencv/opencv_zoo/tree/main/models/image_classification_mobilenet | Apache-2.0 | энкодер кропов (W2, W3); обрезан до глобального пулинга |
| Youtu ReID baseline (`person_reid_youtu_2021nov.onnx`) | https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid (исходно Tencent Youtu Lab, https://github.com/ReID-Team/ReID_extra_testdata) | Apache-2.0 | кандидат-энкодер в сравнении W2 |
| Ultralytics YOLO (опционально) | https://github.com/ultralytics/ultralytics | **AGPL-3.0** | только через адаптер, см. DECISIONS D-002 |

## Видео

Тестовые ролики (`scripts/fetch_videos.py`) — чужие публичные записи с GitHub, только для внутреннего
тестирования; не публиковать и не коммитить. Источники перечислены в самом скрипте.

## Бильярд (SPRINT3 B4)

Поиск — GitHub Search API, ~30 минут (запросы и итог — EVAL §21). Скачиваются `python scripts/fetch_videos.py`
в `data/videos/billiard/`. **Только для внутренней проверки**, не публиковать и не показывать клиентам.

| Файл | Источник | Что | Лицензия |
|---|---|---|---|
| `louisgeisler_NOdebug.mp4` | github.com/louisgeisler/Billard (`NOdebug.mp4`) | любительский стол сверху, вебкамера, 1080p, 38 с, 4 удара | в репозитории не указана → считать «все права защищены», только внутренний тест |
| `fp_game{1..4}_clip{1,2}.mp4` | github.com/FedericoPivotto/billiard-video-analysis (`dataset/`) | 8 клипов ударов из трансляций турниров (YouTube), 1024×576, ~5 с, сверху и под углом | код — MIT; сами кадры — трансляции турниров (права у вещателей), только внутренний тест |

Не взяты: Danthewaann/snooker-ball-tracker (MIT, видео в репозитории не найдено поиском), marianlupascu/Video-analysis-
of-a-snooker-footage (MIT, трансляция снукера, ролики не нашлись по путям из кода), FJ-Rodriguez-Lozano/Billiard-dataset
(CC BY-SA 4.0, изображения, не видео), mattia-toffolon/BilliardEye и riccardofazzi/… (тот же набор клипов, что у
FedericoPivotto). Записей «стол пустой», «люди рядом без игры», «зал бильярдной целиком» не нашлось ни одной.

Демо-ролик бильярда (B6) — **процедурная синтетика** (`qonaq/billiard/synth.py`), чужих кадров в нём нет.
