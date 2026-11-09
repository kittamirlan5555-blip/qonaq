# Как запустить

## Вариант 1 — в облаке (ничего не ставить, комп можно выключить)
1. Открой https://claude.ai/code, выбери репозиторий `qonaq`.
2. Вставь текст из `KICKOFF_PROMPT.md` и отправь. Можно закрыть вкладку, сессия работает в облаке.
3. Утром открой ту же сессию и `PROGRESS.md` в репозитории.
Минусы: нет GPU (всё на CPU, медленнее) и нет YouTube. Тестовые видео скачиваются скриптом с GitHub.

## Вариант 2 — на своём ПК (быстрее, есть GPU)
1. `git clone` репозитория, в папке проекта: `python scripts/fetch_videos.py` (+ свои видео в `data/videos/`).
2. Отключи сон ПК.
3. `claude --permission-mode auto` и вставь `KICKOFF_PROMPT.md`.
`--dangerously-skip-permissions` на основной машине не используй.

Если упёрлись в лимит подписки: всё сохранено в git и PROGRESS.md, напиши «продолжай по TASKS.md».

## Утром на работе (ноутбук с камерой)
`git pull` → `python -m qonaq doctor` → `python -m qonaq live --site configs/sites/demo_webcam.yaml`
→ сядь за стол в кадре, подойди «официантом», посмотри, как меняются статусы.
