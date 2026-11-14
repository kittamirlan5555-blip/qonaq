"""Конфиг объекта (site) — единственный источник правды (SPEC §9a п.7).

Один YAML описывает объект: камеры, профиль логики, пороги, пути к зонам, средний чек, часы работы.
CLI-флаги только переопределяют значения отсюда.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Model):
    # auto: ultralytics, если установлен; иначе YOLOX через onnxruntime (Apache-2.0, без PyTorch).
    backend: Literal["auto", "ultralytics", "onnx", "fake"] = "auto"
    onnx_model: str = "yolox_s"  # yolox_tiny | yolox_s | yolox_m | yolox_l — скачивается из релиза YOLOX
    onnx_tiles: int = Field(1, ge=1, le=2)  # 2 = полный кадр + две половины (мелкие люди, x3 по времени)
    # Список кандидатов: берётся первая модель, которую удалось загрузить.
    model: str = "yolo11n.pt"
    fallback_models: list[str] = Field(default_factory=lambda: ["yolov8n.pt"])
    conf: float = Field(0.30, ge=0.01, le=1.0)
    # Низкий порог только для блюра: размываем и неуверенные детекции людей.
    blur_conf: float = Field(0.12, ge=0.01, le=1.0)
    imgsz: int = Field(960, ge=160, le=1920)
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    # Кэш детекций на диске (только рамки, без изображений) — для повторных прогонов одного видео при отладке.
    cache_dir: str | None = None
    # W3: детектировать ещё и посуду (COCO bottle, wine glass, cup, fork, knife, spoon, bowl).
    dishes: bool = False


class TrackerConfig(_Model):
    track_activation_threshold: float = 0.30
    lost_track_buffer_s: float = 3.0
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 2
    history_s: float = 30.0


class RestaurantThresholds(_Model):
    """Пороги машины состояний стола (секунды времени сцены). SPEC §5."""

    seat_min_s: float = 15.0
    visit_min_s: float = 4.0
    visit_max_s: float = 90.0
    leave_min_s: float = 90.0
    clean_min_s: float = 10.0
    dropout_tolerance_s: float = 5.0
    # Скользящее окно присутствия (SPEC §9a п.1): «включено» при доле >= on, «выключено» при доле <= off.
    presence_on_ratio: float = 0.7
    presence_off_ratio: float = 0.2
    # Пауза внутри одного визита персонала (мигание детекции) — не рвёт визит.
    visit_gap_s: float = 2.0
    # Если роли так и не подтвердили «гостя» (смена id и т.п.), стол считается занятым после
    # seat_min_s * seat_unknown_factor непрерывного присутствия не-персонала.
    seat_unknown_factor: float = 3.0
    # Стол, занятый с первых секунд потока, считается уже обслуженным (мы не знаем, сколько они ждут).
    # 30 с: детектор не всегда видит сидящего с первого кадра (перекрытия), а для потока это мелочь.
    preexisting_grace_s: float = 30.0
    # Сигнал «стол чистый» по эталону пустого стола (SPEC §9a п.4).
    use_table_appearance: bool = False
    appearance_clean_confirm_s: float = 8.0
    # W3: ещё два независимых сигнала состояния стола + правило согласия (qonaq/logic/tablesignals.py).
    use_table_classifier: bool = False  # голова clean/dirty/occupied поверх энкодера (models/tables_head.npz)
    table_classifier: str = "models/tables_head.npz"
    table_clf_dirty_p: float = 0.6  # p(dirty)/(p(dirty)+p(clean)) выше — «грязный», ниже 1−этого — «чистый»
    table_clf_every_s: float = 2.0  # как часто классифицировать кроп стола
    use_dish_detection: bool = False  # посуда детектора (cup/bowl/bottle/...) в зоне стола без людей
    dish_conf: float = 0.3
    dish_min_count: int = 1
    # Посуда не найдена ≠ стол чистый: на датасетах YOLOX находит посуду на 0–10% грязных столов (EVAL.md §7).
    # False — сигнал голосует только «грязный» (иначе молчит).
    dish_absent_is_clean: bool = False
    table_clean_min_agree: int = 2  # сколько сигналов должны сказать «чистый» (не больше числа включённых)
    table_dirty_min_agree: int = 2


class AlertThresholds(_Model):
    # W8: алерт уходит в Telegram, только если гости подтверждены ролью и средняя уверенность ролей не ниже порога.
    # Неуверенные — в панель и отчёт, но не в Telegram. «Лучше пропустить алерт, чем отправить ложный».
    min_role_confidence: float = 0.6
    confident_only: bool = True
    show_site_name: bool = False  # S4: название заведения второй строкой (если у администратора их несколько)
    shift_summary_at: str = "23:30"  # S4: сводка смены владельцу заведения (TELEGRAM_VENUE_CHAT_ID); "" — нет
    venue_chat_env: str = "TELEGRAM_VENUE_CHAT_ID"
    no_waiter_s: float = 180.0
    dirty_table_s: float = 300.0
    long_service_without_visit_s: float = 900.0
    occupied_without_check_s: float = 600.0
    # clinic
    desk_unattended_s: float = 60.0
    long_wait_s: float = 900.0


class ClinicThresholds(_Model):
    enter_min_s: float = 5.0  # сколько трек должен пробыть в зоне, чтобы считаться ожидающим
    exit_min_s: float = 8.0  # сколько трека не должно быть, чтобы считать, что он ушёл
    handoff_s: float = 20.0  # окно «ушёл из ожидания → появился у стойки»
    service_min_s: float = 10.0  # минимальное время у стойки, чтобы считать обслуживанием
    presence_on_ratio: float = 0.7
    presence_off_ratio: float = 0.2
    dropout_tolerance_s: float = 5.0


class RolesConfig(_Model):
    move_threshold: float = 0.6  # смещение якоря за seat_min_s в долях высоты bbox: меньше — «сидит»
    multi_table_count: int = 3
    multi_table_window_s: float = 300.0
    vote_window_s: float = 6.0
    reid_gap_s: float = 4.0  # новый трек рядом с недавно потерянным наследует его историю
    reid_dist: float = 0.6  # в долях диагонали bbox
    standing_aspect: float = 1.9  # h/w bbox выше — «стоит» (слабый признак)
    # W2: стоящий неподвижно у стола становится «гостем» только после стольких секунд (официант, убирающий
    # или принимающий заказ, стоит 10–60 с). Сидящий — после seat_min_s.
    standing_guest_min_s: float = 60.0
    # Если рамка трека между кадрами меняет пропорции в столько раз (сидящий 1:1 → стоящий 1:2.2), это другой
    # человек на том же id трекера (ByteTrack при перекрытии). 0 — выключить.
    identity_jump_aspect: float = 1.7
    identity_jump_similarity: float = 0.5  # ...и гистограмма торса похожа меньше этого (другая одежда)
    # Визит неизвестного к занятому столу считается визитом персонала, только если он стоит (SPEC §9a п.3).
    visit_requires_standing: bool = True
    staff_profile: str | None = None  # путь к configs/staff_profile.yaml
    uniform_threshold: float = 0.55
    # --- Внешний вид (W2): голова «персонал/гость» поверх ONNX-энкодера. None — роли только по поведению.
    classifier: str | None = None  # например models/roles_head.npz
    appearance_every_s: float = 1.0  # как часто пересчитывать внешний вид одного трека
    appearance_window_s: float = 10.0  # усреднение p(staff) по треку
    appearance_staff_thr: float = 0.7  # среднее p(staff) выше — голос «персонал»
    appearance_guest_thr: float = 0.3  # ниже — голос «гость»
    appearance_min_samples: int = 2  # сколько оценок нужно, прежде чем голосовать
    conflict_margin: float = 0.25  # если «гость» и «персонал» оба сильны и ближе этого — unknown
    # Зоны-столы, в которых человек считается персоналом всегда (стойка бара и т.п.) — задаются типом ignore.


class BusinessConfig(_Model):
    avg_check_kzt: float = 15000.0  # средний чек на стол, ₸
    default_visit_min: float = 60.0  # если по данным мало сессий
    peak_occupancy: float = 0.8  # «пик»: загрузка зала >= этого порога или очередь у входа > 0
    target_cleanup_s: float = 120.0  # нормативное время уборки; всё сверх него — «лишний простой»
    open_hours: str = "10:00-23:00"
    timezone: str = "Asia/Almaty"
    currency: str = "₸"


class PrivacyConfig(_Model):
    blur_faces: bool = True  # обязательное свойство для сохраняемых видео (SPEC §9)
    blur_top_fraction: float = 0.38
    save_raw_video: Literal[False] = False  # сырое видео не сохраняется никогда


class PosConfig(_Model):
    adapter: Literal["none", "mock", "iiko", "rkeeper", "poster"] = "none"
    scenario: str | None = None  # для mock: путь к YAML со сценарием чеков
    table_map: dict[str, str] = Field(default_factory=dict)  # zone_id -> номер стола в POS


class CameraConfig(_Model):
    id: str
    name: str = ""
    source: str  # путь к файлу | webcam:N | rtsp://...  (поддерживает ${ENV_VAR})
    zones: str | None = None  # путь к YAML зон; по умолчанию configs/zones/<id>.yaml
    process_fps: float | None = None
    loop: bool = False  # зациклить файл (для демо/эмуляции потока)
    realtime: bool | None = None  # для файла: читать в темпе реального времени (эмуляция потока)

    @property
    def title(self) -> str:
        return self.name or self.id


class HealthSettings(_Model):
    """Монитор здоровья и ежедневная сводка владельцу системы (W7). Чат — отдельный от администратора."""

    enabled: bool = True
    check_every_s: float = 60.0
    daily_at: str = "23:40"  # местное время объекта (business.timezone); "" — не отправлять сводку
    owner_chat_env: str = "TELEGRAM_OWNER_CHAT_ID"
    daily_frames: int = 5
    sample_every_s: float = 600.0  # как часто откладывать кадр (с блюром) для сводки
    repeat_issue_s: float = 3600.0  # одну и ту же проблему не присылать чаще
    camera_stale_s: float = 30.0  # нет кадров дольше — «камера не отвечает»
    stats_every_s: float = 60.0  # строка в live_stats.jsonl: fps и задержка по камерам, CPU, память (S1)


class SiteConfig(_Model):
    site_id: str = "demo"
    name: str = "Демо-объект"
    profile: Literal["restaurant", "clinic"] = "restaurant"
    process_fps: float = Field(5.0, gt=0, le=30)
    cameras: list[CameraConfig] = Field(default_factory=list)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    thresholds: RestaurantThresholds = Field(default_factory=RestaurantThresholds)
    clinic: ClinicThresholds = Field(default_factory=ClinicThresholds)
    alerts: AlertThresholds = Field(default_factory=AlertThresholds)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    pos: PosConfig = Field(default_factory=PosConfig)
    health: HealthSettings = Field(default_factory=HealthSettings)
    runs_dir: str = "runs"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @model_validator(mode="after")
    def _unique_cameras(self) -> SiteConfig:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("id камер должны быть уникальны")
        return self

    def camera(self, camera_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        raise KeyError(camera_id)


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Подставляет ${VAR} и ${VAR:-по_умолчанию} из окружения (.env). Неизвестные без умолчания остаются как есть."""
    def _sub(m: re.Match) -> str:
        if m.group(1) in os.environ:
            return os.environ[m.group(1)]
        return m.group(2) if m.group(2) is not None else m.group(0)

    return _ENV_RE.sub(_sub, value)


def load_dotenv_if_any(root: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    env = (root or Path.cwd()) / ".env"
    if env.exists():
        load_dotenv(env)


def load_site(path: str | Path) -> SiteConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SiteConfig.model_validate(data)


def site_for_file(video: str | Path, base: SiteConfig | None = None, camera_id: str | None = None) -> SiteConfig:
    """Конфиг «один файл — одна камера» для команд run-file / demo."""
    video = Path(video)
    site = (base or SiteConfig()).model_copy(deep=True)
    cam_id = camera_id or video.stem
    site.cameras = [CameraConfig(id=cam_id, name=cam_id, source=str(video))]
    if base is None:
        site.site_id = cam_id
        site.name = cam_id
    return site


def resolve_path(p: str | Path) -> Path:
    """Относительные пути считаются от текущей директории, затем от корня проекта."""
    p = Path(expand_env(str(p)))
    if p.is_absolute() or p.exists():
        return p
    alt = PROJECT_ROOT / p
    return alt if alt.exists() else p


def mask_secrets(url: str) -> str:
    """rtsp://user:pass@host -> rtsp://user:***@host (для логов)."""
    return re.sub(r"(://[^:/@]+):[^@]+@", r"\1:***@", url)
