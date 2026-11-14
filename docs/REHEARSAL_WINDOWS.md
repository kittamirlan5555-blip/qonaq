# Репетиция установки дома на Windows (≈ 1–1,5 часа)

Цель — пройти **весь день установки** (`docs/PILOT_RUNBOOK.md`, раздел 1) у себя дома, без заведения и без камер:
вместо регистратора — эмулятор (`scripts/emulate_cameras.py`, 4 ролика по RTSP с логином и паролем, пути как у
Hikvision/Dahua). После репетиции в бильярдной/кафе вы будете делать то же самое, только с настоящим IP.

Маршрут: установка → `doctor` → эмулятор → `qonaq setup` → зоны → пробный `live` и панель → Telegram →
служба с автозапуском → `health` → обрыв камеры → восстановление → сбой службы → перезагрузка ПК → уборка.

**Как проверено.** Весь сценарий пройден 14.11.2026 в облаке на Linux (2 vCPU, без GPU, Python 3.12) — выводы
ниже взяты оттуда (пути и время поправлены под Windows). Что на Linux проверить нельзя, помечено
**⚠ Windows — не проверено**: это нужно увидеть своими глазами на репетиции. Найденное на репетиции и уже
исправленное — в конце, раздел «Что нашла репетиция».

Обозначения: `PS>` — PowerShell (обычный), `PS(адм)>` — PowerShell **от имени администратора**.
Все команды — **из папки проекта** `C:\Qonaq` (не из другой папки: журналы `runs\` пишутся в текущую).

---

## 0. Что приготовить (10 мин)

| Что | Зачем | Проверка |
|---|---|---|
| Windows 10/11, 8+ ГБ памяти, 5 ГБ на диске | — | — |
| Python **3.11 или 3.12** с python.org, при установке галочка **Add python.exe to PATH** | 3.13 не проверялся | `python --version` → `Python 3.11.x` |
| Git for Windows | клонировать проект | `git --version` |
| Папка **`C:\Qonaq`** — без пробелов и кириллицы | служба Windows запускает `.cmd`-обёртку; кириллица в пути — частая причина «служба стартует и сразу падает» | — |
| Telegram-бот и chat id (вы это уже делали, `docs/TELEGRAM.md`) | шаг 7 | — |

## 1. Установка (5–10 мин)

```powershell
PS> cd C:\
PS> git clone https://github.com/kittamirlan5555-blip/qonaq Qonaq
PS> cd C:\Qonaq
PS> git checkout claude/bold-fermat-de51i5
PS> python -m venv .venv
PS> .venv\Scripts\Activate.ps1
PS> pip install -e ".[dev]"
```

Ожидается: в конце `Successfully installed … qonaq-0.1.0 …`, слева в приглашении появляется `(.venv)`.

Для NVIDIA (у вас RTX 2060 SUPER) — как уже проверено 14.11:

```powershell
PS> pip uninstall -y onnxruntime
PS> pip install "onnxruntime-gpu[cuda,cudnn]"
```

| Ошибка | Что делать |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ответить `Y`. Или не активировать, а писать `.venv\Scripts\python.exe` вместо `python` |
| `python` открывает Microsoft Store | Параметры → Приложения → Псевдонимы выполнения → выключить `python.exe`/`python3.exe` |
| `Could not find a version that satisfies … onnxruntime` | не та версия Python (3.13/32-битный): `py -3.11 -m venv .venv` |

## 2. Проверка окружения (1–2 мин)

```powershell
PS> python -m qonaq doctor
```

Ожидается (Linux-вывод; у вас вместо `cpu` будет `cuda:0`, если стоит onnxruntime-gpu):

```
Qonaq Vision — проверка окружения

  Python      3.12.3   Linux … (x86_64)              ← у вас: Windows-10-… (AMD64)
  OK numpy        2.5.4
  OK cv2          5.0.0
  OK supervision  0.26.1
  OK pydantic     2.14.3
  OK fastapi      0.163.0
  OK onnxruntime  1.32.0
  -- torch        не установлен
  -- ultralytics  не установлен
  …
  OK детектор: yolox_s (OnnxYoloxDetector), устройство: cpu
  Видео в data/videos (0): …
Итог: всё готово к работе
```

При первом запуске качаются веса детектора (`yolox_s.onnx`, ~35 МБ) — строка `скачиваю веса …`.

## 3. Тестовые ролики и эмулятор регистратора (5 мин) — **окно 1**

```powershell
PS> python scripts/fetch_videos.py
PS> python scripts/emulate_cameras.py
```

Ожидается (первый раз ~1 мин: перекодирует 4 ролика в «субпоток» и скачивает MediaMTX с проверкой SHA-256):

```
готовлю субпоток cctv_restaurant_corner_960w_12fps.mp4 ...
…
качаю MediaMTX v1.21.2: mediamtx_v1.21.2_windows_amd64.zip
Камеры:
  cam1 [hikvision] cctv_restaurant_corner.mp4: rtsp://admin:qonaq12345@127.0.0.1:8554/Streaming/Channels/102
  cam2 [hikvision] cctv_terrace_cafe.mov: rtsp://admin:qonaq12345@127.0.0.1:8554/Streaming/Channels/202
  cam3 [hikvision] topdown_cafe_people.mp4: rtsp://admin:qonaq12345@127.0.0.1:8554/Streaming/Channels/302
  cam4 [dahua] foodcourt_preannotated.mp4: rtsp://admin:qonaq12345@127.0.0.1:8554/cam/realmonitor?channel=1&subtype=1
```

**Окно 1 не закрывать** — это и есть «регистратор». Остановить — Ctrl+C.

- ⚠ Windows — не проверено: брандмауэр Windows может спросить «Разрешить mediamtx доступ к сетям» —
  для репетиции можно «Отмена» (локально, через 127.0.0.1, работает и без разрешения).
- Проверить «камеру» глазами (необязательно): VLC → Медиа → Открыть URL → адрес cam1 из списка.

## 4. Найти камеры: `qonaq setup` (3–5 мин) — **окно 2**

Сначала **нарочно с неверным паролем** — так вы увидите, что будет в заведении при опечатке:

```powershell
PS> cd C:\Qonaq; .venv\Scripts\Activate.ps1
PS> python -m qonaq setup --site-id home_rehearsal --name "Репетиция дома" --nvr-ip 127.0.0.1 --port 8554
Логин регистратора [admin]:            ← Enter
Пароль регистратора:                   ← набрать что угодно неверное (символы не видны — так и надо)
```

```
127.0.0.1:8554 — ищу шаблон RTSP (auto)…
  hikvision  канал 1: неверный логин или пароль
  dahua      канал 1: неверный логин или пароль
  ✗ неверный логин или пароль регистратора (дальше не перебираю, чтобы его не заблокировать)
  Проверьте пароль (он же — для входа в веб-интерфейс регистратора) и запустите снова.
```

Теперь правильно (пароль `qonaq12345`):

```
127.0.0.1:8554 — ищу шаблон RTSP (auto)…
  hikvision  канал 1: поток есть
  ✓ Hikvision / HiWatch: каналы 1, 2, 3

Снимаю кадры (12 шт. с шагом 1 с) и ищу столы…
  cam01 (канал 1): 960×540, столов найдено: 0; столы не найдены автоматически — разметить вручную (zones edit)
  cam02 (канал 2): 960×508, столов найдено: 2
  cam03 (канал 3): 960×540, столов найдено: 1

Конфиг объекта: C:\Qonaq\configs\sites\home_rehearsal.yaml
Логин и пароль: C:\Qonaq\.env (QONAQ_NVR_USER / QONAQ_NVR_PASSWORD)
Кадры и черновики зон (лица размыты): C:\Qonaq\runs\setup\home_rehearsal

Дальше:
  python -m qonaq zones edit runs/setup/home_rehearsal/cam01.jpg --zones configs/zones/home_rehearsal_cam01.yaml
  …
```

На Linux заняло 49 с. Строки вида `[h264 @ …] mmco: unref short failure` — шум декодера, не ошибка.

Почему каналов 3, а не 4: на «регистраторе» Hikvision-каналы 1–3, а 4-я «камера» эмулятора — Dahua с другим
путём; `setup` берёт один шаблон на регистратор (у настоящего регистратора он один). Это ожидаемо.

Что открыть и посмотреть: `runs\setup\home_rehearsal\cam02.zones.jpg` — черновик столов на кадре (лица размыты).

| Ошибка | Что значит |
|---|---|
| `127.0.0.1:8554 не отвечает (порт RTSP закрыт…)` | окно 1 закрыто или эмулятор упал |
| `поток открылся по RTSP, но кадры не читаются (кодек H.265?…)` | на репетиции не бывает; в заведении — переключить субпоток на H.264 |
| `Камеры не найдены` | неверный `--port` (у эмулятора 8554, у регистраторов обычно 554) |

## 5. Зоны (10–15 мин)

**cam01** — столов не найдено (косая угловая камера — типичный случай). Два пути:

(а) как в заведении — нарисовать руками:

```powershell
PS> python -m qonaq zones edit runs/setup/home_rehearsal/cam01.jpg --zones configs/zones/home_rehearsal_cam01.yaml
```

Откроется окно с кадром: клик — вершина, **Enter** — закрыть многоугольник, **Backspace** — отменить, **t** — тип
зоны (`table` → `entrance` → … → `ignore`), **n** — имя (вводится в консоли), **s** — сохранить, **q** — выйти.
Стол рисовать **вместе со стульями** и сидящими (на косой камере — выше столешницы).
⚠ Windows — не проверено в этой сессии (в облаке нет экрана); окно OpenCV на Windows раньше работало.

(б) быстро — взять готовую ручную разметку этого же ролика (она в другом разрешении, пересчитается сама):

```powershell
PS> (Get-Content configs\zones\cctv_restaurant_corner.yaml) -replace '^camera_id: .*', 'camera_id: cam01' | Set-Content -Encoding utf8 configs\zones\home_rehearsal_cam01.yaml
```

Проверка — картинка с зонами:

```powershell
PS> python -m qonaq zones preview runs/setup/home_rehearsal/cam01.jpg --zones configs/zones/home_rehearsal_cam01.yaml
Превью: runs\zones\cam01.png
```

Ожидается: 7 зелёных столов `T01…T07` и розовый вход `E01` поверх кадра 960×540.

Бизнес-параметры — в `configs\sites\home_rehearsal.yaml` (`business.avg_check_kzt`, `open_hours`), для репетиции
можно не трогать.

## 6. Пробный запуск и панель (5 мин)

```powershell
PS> python -m qonaq live --site configs/sites/home_rehearsal.yaml
```

```
Telegram не настроен (нет TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID) — алерты только в лог и панель   ← если .env без бота
Объект: Репетиция дома · камер: 3 · журнал: runs/live_home_rehearsal/events.sqlite
Панель: http://127.0.0.1:8000   (Ctrl+C — остановить)
… через минуту:
ЗДОРОВЬЕ cam02: роль не определена у 100% наблюдений людей
обработано 266 кадров, 4.4 кадр/с, сцена 58 с
```

Открыть **http://127.0.0.1:8000**: три камеры с картинкой (лица размыты), столы с цветом статуса.

Что считать нормой на репетиции:
- **кадр/с.** В облаке (2 vCPU, без GPU, yolox_s) — 4,4–5,0 кадр/с на все три камеры, т.е. ~1,5 на камеру
  вместо `process_fps: 2`. На вашем ПК с GPU должно быть 2 на камеру; если меньше — `detector.onnx_model: yolox_tiny`.
- **«роль не определена у 100%»** — нормально: ролики короткие и идут по кругу, форма персонала не откалибрована.
  В заведении это предупреждение — повод для `calibrate` после суток записи.
- События на зацикленных роликах «ненастоящие» (люди телепортируются на стыке) — репетиция проверяет связь,
  панель и службу, а не точность.

Ctrl+C → в конце `Отчёт: runs/live_home_rehearsal/report.html`.

## 7. Telegram (5 мин)

Вы это уже проверяли на настоящем боте 14.11. Для репетиции достаточно:

```powershell
PS> python -m qonaq telegram test --site configs/sites/home_rehearsal.yaml
```

Без бота — то же на локальном «фейковом» Telegram (проверяет очередь при обрыве интернета, кнопку «Неверно»):

```powershell
PS> python -m qonaq telegram test --fake --site configs/sites/home_rehearsal.yaml
```

```
• алерт администратору отправлен: True
• дедупликация (одно сообщение на событие): True
• кнопка «Неверно» → feedback: True
• ответ на нажатие: True
• очередь при обрыве: доставлено после восстановления с пометкой: True
• итоги дня владельцу заведения: отправлено
• техническая сводка + кадры владельцу системы: отправлено
```

Для шагов 9–11 важно, чтобы в `.env` был **`TELEGRAM_OWNER_CHAT_ID`** (ваш чат) — туда придут «камера не отвечает»
и «камера снова работает».

## 8. Служба с автозапуском (5 мин) — **PowerShell от администратора**

```powershell
PS(адм)> cd C:\Qonaq; .venv\Scripts\Activate.ps1
PS(адм)> python -m qonaq service install --site configs/sites/home_rehearsal.yaml
```

Ожидается — текст обёртки и создание задачи Планировщика:

```
Обёртка C:\Qonaq\runs\qonaq-home-rehearsal.cmd:
@echo off
chcp 65001 >nul
cd /d "C:\Qonaq"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set QONAQ_LOG_FILE=C:\Qonaq\runs\service_home_rehearsal.log
:loop
"C:\Qonaq\.venv\Scripts\python.exe" -m qonaq live --site configs/sites/home_rehearsal.yaml
echo %date% %time% live завершился с кодом %errorlevel%, перезапуск через 10 с >> "C:\Qonaq\runs\service_home_rehearsal.log"
ping -n 11 127.0.0.1 >nul
goto loop
$ schtasks /Create /F /TN qonaq-home-rehearsal /SC ONSTART /RU SYSTEM /RL HIGHEST /TR "C:\Qonaq\runs\qonaq-home-rehearsal.cmd"
УСПЕХ. Запланированная задача "qonaq-home-rehearsal" успешно создана.        ← ⚠ Windows — не проверено
$ schtasks /Run /TN qonaq-home-rehearsal
УСПЕХ. Попытка выполнения запланированной задачи "qonaq-home-rehearsal".     ← ⚠ Windows — не проверено
```

Текст обёртки и команды проверены (сгенерированы на Linux для `C:\Qonaq`); **сама задача Планировщика, запуск от
SYSTEM и перезапуск через `ping` не проверялись ни разу** — это главное, что даёт репетиция.

Проверка (через 1–2 мин):

```powershell
PS> python -m qonaq service status --site configs/sites/home_rehearsal.yaml
PS> python -m qonaq service logs --site configs/sites/home_rehearsal.yaml --lines 20
```

Ожидается: в `status` — `Состояние: Выполняется` (англ. `Status: Running`) и последний `health.json`;
в `logs` — те же строки, что в шаге 6, с датой. Панель http://127.0.0.1:8000 открывается (служба запускает live
с панелью). В Диспетчере задач — `python.exe` от пользователя `СИСТЕМА`.

| Ошибка | Что делать |
|---|---|
| `Отказано в доступе` / `Access is denied` | PowerShell не от администратора |
| в логе раз в ~15 с `live завершился с кодом 1` | live падает сразу: открыть `runs\service_home_rehearsal.log`, первая ошибка сверху. Частые: путь с кириллицей/пробелом; `.env` не в `C:\Qonaq` |
| `[Errno 10048] … only one usage of each socket address` | панель уже занята: вы запустили `live` руками, пока работает служба. Сначала `service uninstall` (или `--port 8001`). **Не запускать live руками параллельно со службой** — оба пишут в один журнал |
| `status` пишет «Готово», а панели нет | задача создана, но не запущена: `service restart` |
| в логе `в адресе камеры не подставлены ${QONAQ_NVR_USER}…` | нет `.env` в `C:\Qonaq` (setup пишет его туда) |

## 9. `health` — сводка здоровья (1 мин)

```powershell
PS> python -m qonaq health runs/live_home_rehearsal --site configs/sites/home_rehearsal.yaml
```

```
Статус: warn
  [warn] cam02 : роль не определена у 100% наблюдений людей
  [warn] cam03 : роль не определена у 76% наблюдений людей
  cam01: посадок 1, визитов персонала 0, unknown 80%
  cam02: посадок 0, визитов персонала 0, unknown 100%
  cam03: посадок 1, визитов персонала 0, unknown 76%
```

`warn` из-за ролей — нормально для репетиции (шаг 6). `error` — повод разбираться.

## 10. Обрыв камеры и восстановление (5 мин)

Окно 1 → **Ctrl+C** (эмулятор остановлен — все «камеры» пропали). Подождать **1–2 минуты**.

В логе службы (`service logs`) и в Telegram владельцу системы (если есть `TELEGRAM_OWNER_CHAT_ID`):

```
источник rtsp://admin:***@127.0.0.1:8554/Streaming/Channels/102 недоступен, повтор через 15 с     ← каждые 15 с
ЗДОРОВЬЕ cam01: камера не отвечает 47 с                         ← в Telegram: 🛠 Репетиция дома · cam01: камера не отвечает 47 с
ЗДОРОВЬЕ cam02: камера не отвечает 47 с
ЗДОРОВЬЕ cam03: камера не отвечает 47 с
```

Снова запустить эмулятор в окне 1 (`python scripts/emulate_cameras.py`, второй раз стартует за секунды —
субпотоки и MediaMTX уже скачаны). В течение **~20–30 с** — переподключение (повтор раз в 15 с), ещё **≤ 1 мин** — сообщение:

```
источник rtsp://admin:***@127.0.0.1:8554/Streaming/Channels/102: переподключено
ЗДОРОВЬЕ cam01: камера снова работает (не было 107 с)          ← и в Telegram
```

Как было на Linux (обрыв ровно 90 с через `--outage-at 200 --outage-s 90`): «не отвечает» через 47 с после
обрыва, переподключение через 17–21 с после возврата сервера, «снова работает (не было 107–111 с)» — у камеры,
вернувшейся сразу перед плановой проверкой, через 0 с, у остальных — через 58 с (проверка раз в
`health.check_every_s` = 60 с). Сообщения приходят **по разу** на обрыв, не каждую минуту.

То же без Ctrl+C, по таймеру: `python scripts/emulate_cameras.py --outage-at 120 --outage-s 90`.

## 11. Сбой службы и перезагрузка (10 мин)

**Аварийное падение.** Диспетчер задач → Подробности → `python.exe` (пользователь `СИСТЕМА`) → Снять задачу.
Через ~10 с обёртка запускает live заново; в логе:

```
пт 14.11.2026 21:15:00,00 live завершился с кодом 1, перезапуск через 10 с      ← ⚠ Windows — формат даты/кода не проверен
… Telegram не настроен … / Объект: Репетиция дома …
```

Через 1–2 мин `health` показывает простой по журналу (раньше эта информация терялась при перезапуске — исправлено):

```
Простои по журналу (heartbeat):
  cam01: не было кадров 15.11 02:11–02:12 (1 мин)            ← шаг 10
  cam02: не было кадров 15.11 02:11–02:13 (2 мин)
  cam03: не было кадров 15.11 02:11–02:13 (2 мин)
  система не работала 15.11 02:14–02:15 (1 мин) — аварийно: сбой, выключение или питание
```

Время — местное время заведения (`business.timezone`, Asia/Almaty). Точность — ±1 мин (heartbeat раз в минуту).
Те же строки попадут в ежедневную техническую сводку в Telegram.

**Перезагрузка.** Перезагрузить ПК и **не входить в систему** 2–3 минуты, потом войти: панель уже должна работать
(задача стартует при загрузке, до входа пользователя). ⚠ Windows — не проверено. Эмулятор после перезагрузки сам
не поднимется — пока его нет, в логе «недоступен, повтор», это и есть проверка «регистратор включился позже ПК».
Запустить эмулятор → камеры подхватились без перезапуска службы.

## 12. Уборка (2 мин)

```powershell
PS(адм)> python -m qonaq service uninstall --site configs/sites/home_rehearsal.yaml
PS> Remove-Item configs\sites\home_rehearsal.yaml, configs\zones\home_rehearsal_*.yaml
PS> Remove-Item -Recurse runs\live_home_rehearsal, runs\setup\home_rehearsal
```

`.env` (пароль эмулятора) можно оставить — `setup` в заведении перезапишет `QONAQ_NVR_USER/PASSWORD`.

---

## Чек-лист репетиции (отметить, что увидели сами)

- [ ] `doctor` — «всё готово», устройство `cuda:0`
- [ ] `setup` с неверным паролем остановился после 2 попыток
- [ ] `setup` нашёл 3 канала, кадры с размытыми лицами
- [ ] зона нарисована в `zones edit` и видна на `zones preview`
- [ ] панель: 3 камеры, ≥ 2 кадр/с на камеру (на GPU)
- [ ] `service install` → `УСПЕХ`, `python.exe` от `СИСТЕМА`
- [ ] обрыв: «камера не отвечает» в Telegram ≤ 2 мин; «снова работает» ≤ 1 мин после возврата
- [ ] снятие python.exe → live снова запущен ≤ 15 с; `health` показывает «аварийно»
- [ ] после перезагрузки панель работает **без входа в систему**
- [ ] `service uninstall` — задачи нет в Планировщике

Если что-то не совпало с ожидаемым — сохранить `runs\service_home_rehearsal.log` и вывод команды.

## Что нашла репетиция на Linux (исправлено, тесты `tests/test_rehearsal.py`)

1. **`.env` искался только в текущей папке.** Команда, запущенная не из `C:\Qonaq`, стучалась в камеры с
   `${QONAQ_NVR_USER}` в адресе и получала 401 — молча, повтор за повтором. Теперь `.env` берётся из текущей папки,
   затем из корня проекта, а нераскрытая `${…}` в адресе камеры — явная ошибка в логе.
2. **Остановка службы на Linux (SIGTERM от systemd) убивала live до сохранения.** uvicorn после остановки заново
   поднимал сигнал — терялись хвост очереди событий, последний heartbeat, `health.json`, `run.json`, отчёт.
   Исправлено (`graceful_server`), проверено: после SIGTERM есть `Отчёт: …` и последний heartbeat `final`.
   На Windows `schtasks /End` и «Снять задачу» завершают процесс жёстко — там события последних секунд теряются
   всегда (известное ограничение, `DECISIONS.md` D-040).
3. **После перезапуска службы `health` и ежедневная сводка «забывали» обрывы** — они жили только в памяти процесса.
   Теперь простои системы и камер считаются по heartbeat из журнала (переживают перезапуск и отключение питания),
   штатная остановка отличается от аварийной.
4. **«Сводка за день» считала посадки и алерты за всё время** — журнал live один на объект и копится неделями.
   Теперь — за последние 24 ч.

Не исправлено, просто знать: журналы `runs\` пишутся в текущую папку — запускайте команды из `C:\Qonaq`;
`setup` на «регистраторе» со смешанными шаблонами (Hikvision + Dahua) находит каналы только одного шаблона.
