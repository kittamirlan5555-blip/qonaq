"""LiveService: пайплайн в фоновом потоке + общее состояние для веб-панели.

Цикл обработки ничего не ждёт: события пишет AsyncEventWriter, алерты шлёт AlertDispatcher,
панель читает последний снимок состояния и последний JPEG под замком.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from qonaq.alerts.telegram import AlertDispatcher, sender_from_env
from qonaq.config import SiteConfig
from qonaq.events.store import AsyncEventWriter, EventStore
from qonaq.pipeline import FrameResult, Pipeline
from qonaq.render.annotate import draw_people, draw_zones
from qonaq.render.blur import blur_people
from qonaq.sources.video import parse_source

log = logging.getLogger("qonaq.live")


class LiveService:
    def __init__(self, site: SiteConfig, detector=None, run_dir: str | Path | None = None, sender=None,
                 use_env_sender: bool = True, max_seconds: float | None = None, pos=None,
                 jpeg_width: int = 960):
        self.site = site
        self.run_dir = Path(run_dir) if run_dir else Path(site.runs_dir) / f"live_{site.site_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(self.run_dir / "events.sqlite")
        kinds = {parse_source(c.source)[0] for c in site.cameras}
        self.time_base = "video" if kinds == {"file"} else "epoch"
        if self.time_base == "epoch":  # B8: сессии, оборванные прошлым процессом (перезапуск, сбой, питание)
            closed = self.store.close_interrupted()
            if closed:
                log.warning("закрыто оборванных прошлым запуском: %d (столы/игры: %s)", len(closed),
                            ", ".join(sorted({f"{e.camera_id}/{e.zone_id}" for e in closed})))
        self.alerts = AlertDispatcher(sender if sender is not None else (sender_from_env() if use_env_sender else None),
                                      site_name=site.name if site.alerts.show_site_name else "",
                                      tz=site.business.timezone, min_confidence=site.alerts.min_role_confidence,
                                      confident_only=site.alerts.confident_only, store=self.store)
        # W7: монитор здоровья + сводка владельцу системы (отдельный чат)
        from qonaq.health.daily import FrameSampler
        from qonaq.health.monitor import HealthConfig, HealthMonitor
        from qonaq.health.proc import CpuMeter

        self.health = HealthMonitor(HealthConfig(stale_s=site.health.camera_stale_s, target_fps=site.process_fps))
        for c in site.cameras:
            self.health.register(c.id)
        self._outages_seen: dict[str, int] = {}
        self._cpu = CpuMeter()
        self._last_stats_wall = time.time()
        self.perf: dict[str, Any] = {}
        self.sampler = FrameSampler(site.health.sample_every_s)
        self.owner = sender_from_env(site.health.owner_chat_env) if use_env_sender else None
        self._issue_sent: dict[tuple, float] = {}
        self._daily_day: str | None = None
        self._shift_day: str | None = None
        self.venue = sender_from_env(site.alerts.venue_chat_env) if use_env_sender else None
        self._health_thread: threading.Thread | None = None
        self._feedback = None
        self.writer = AsyncEventWriter(self.store)
        if pos is None and site.pos.adapter != "none":
            from qonaq.pos import create_pos

            pos = create_pos(site.pos)
        self.pipeline = Pipeline(site, detector=detector, store=self.writer, max_seconds=max_seconds, pos=pos)
        self.jpeg_width = jpeg_width
        self._lock = threading.Lock()
        self._jpeg: dict[str, bytes] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_wall = time.time()
        self.finished = threading.Event()
        self.error: str | None = None
        self._tables = sum(len(c.zones.by_type("table")) for c in self.pipeline.cameras.values())
        self.store.set_meta(time_base=self.time_base, site_id=site.site_id, profile=site.profile,
                            tables=self._tables, started_wall=self.started_wall)

    # ------------------------------------------------------------------ pipeline thread
    def _on_result(self, r: FrameResult) -> None:
        cam = self.pipeline.cameras[r.camera_id]
        titles = {z.id: z.title for z in cam.zones.zones}
        self.alerts.handle(r.events, titles)
        self.health.observe_result(r)
        img = r.frame.copy()
        if self.site.privacy.blur_faces:
            blur_people(img, r.blur_boxes, self.site.privacy.blur_top_fraction)
        draw_zones(img, cam.zones, r.zone_states, self.site.alerts.no_waiter_s, show_ignore=False,
                   font=max(14, img.shape[1] // 70))
        draw_people(img, r, show_ids=False, thickness=1)
        if self.site.privacy.blur_faces:
            self.sampler.maybe_add(r, img, self._local_time_label())
        if img.shape[1] > self.jpeg_width:
            s = self.jpeg_width / img.shape[1]
            img = cv2.resize(img, (self.jpeg_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        zones = []
        for z in cam.zones.zones:
            if z.type == "ignore":
                continue
            snap = dict(r.zone_states.get(z.id, {}))
            snap.update({"zone_id": z.id, "name": z.name, "type": z.type})
            zones.append(snap)
        state = {"camera_id": r.camera_id, "name": cam.cam.title, "ts": r.ts, "summary": r.summary, "zones": zones,
                 "frame": r.frame_index, "updated_wall": time.time()}
        with self._lock:
            if ok:
                self._jpeg[r.camera_id] = buf.tobytes()
            self._state[r.camera_id] = state

    def _run(self) -> None:
        try:
            self.pipeline.run(on_result=self._on_result, stop=self._stop, progress_every_s=60)
            self.pipeline.finish()  # штатная остановка: открытые сессии бильярда закрываются (reason=shutdown)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.exception("пайплайн остановился с ошибкой")
        finally:
            self.finished.set()

    # ------------------------------------------------------------------ здоровье (W7)
    def _local_now(self):
        from datetime import datetime

        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(self.site.business.timezone)).timetuple()
        except Exception:  # noqa: BLE001 — на Windows без пакета tzdata
            return time.localtime()

    def _tz(self):
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(self.site.business.timezone)
        except Exception:  # noqa: BLE001 — на Windows без пакета tzdata
            return None

    def _local_time_label(self) -> str:
        return time.strftime("%H:%M", self._local_now())

    def health_tick(self) -> dict[str, Any]:
        """Одна проверка: новые проблемы → владельцу; пора — ежедневная сводка. Возвращает статус."""
        from qonaq.health.daily import build_daily, due, send_daily

        st = self.health.status(live=True)
        (self.run_dir / "health.json").write_text(__import__("json").dumps(st, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
        now = time.time()
        for i in st["issues"]:
            key = (i["kind"], i["camera_id"], i["zone_id"])
            if now - self._issue_sent.get(key, -1e18) < self.site.health.repeat_issue_s:
                continue
            self._issue_sent[key] = now
            log.warning("ЗДОРОВЬЕ %s: %s", i["camera_id"], i["message"])
            if self.owner is not None:
                try:
                    self.owner.send(f"🛠 {self.site.name} · {i['camera_id']}: {i['message']}")
                except Exception as e:  # noqa: BLE001
                    log.warning("сводка/проблема владельцу не отправлена: %s", e)
        for cid, cst in st["cameras"].items():  # камера вернулась после обрыва — сообщить один раз
            n = cst.get("outages", 0)
            if n > self._outages_seen.get(cid, 0):
                self._outages_seen[cid] = n
                msg = f"камера снова работает (не было {cst['last_outage']['s']:.0f} с)"
                log.warning("ЗДОРОВЬЕ %s: %s", cid, msg)
                if self.owner is not None:
                    try:
                        self.owner.send(f"🛠 {self.site.name} · {cid}: {msg}")
                    except Exception as e:  # noqa: BLE001
                        log.warning("сообщение владельцу не отправлено: %s", e)
        lt = self._local_now()
        if due(lt, self.site.alerts.shift_summary_at, self._shift_day):
            self._shift_day = time.strftime("%Y-%m-%d", lt)
            try:
                text = self.shift_summary(lt)
                (self.run_dir / f"shift_{self._shift_day}.txt").write_text(text, encoding="utf-8")
                if self.venue is not None:
                    self.venue.send(text)
            except Exception as e:  # noqa: BLE001
                log.warning("сводка смены не отправлена: %s", e)
        if due(lt, self.site.health.daily_at, self._daily_day):
            self._daily_day = time.strftime("%Y-%m-%d", lt)
            daily = build_daily(self.run_dir, self.site.name, st, self.sampler.pick(self.site.health.daily_frames),
                                self.perf, since=now - 86400 if self.time_base == "epoch" else None,
                                tz=self._tz())
            if self.owner is not None:
                try:
                    send_daily(self.owner, daily)
                except Exception as e:  # noqa: BLE001
                    log.warning("ежедневная сводка не отправлена: %s", e)
        return st

    def shift_summary(self, lt=None) -> str:
        """Сводка смены владельцу заведения (S4): события с местной полуночи (живой поток) или весь прогон."""
        from qonaq.metrics.compute import compute_metrics
        from qonaq.metrics.period import slice_store
        from qonaq.metrics.report import shift_summary_text

        lt = lt or self._local_now()
        now = time.time()
        if self.time_base == "epoch":
            since_midnight = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
            store = slice_store(self.store, now - since_midnight, now)
        else:
            store = self.store
        m = compute_metrics(store, self.site)
        months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
                  "октября", "ноября", "декабря"]
        return shift_summary_text(m, self.site.business.currency, f"{lt.tm_mday} {months[lt.tm_mon - 1]}")

    def perf_tick(self, final: bool = False) -> dict[str, Any]:
        """Строка live-статистики (S1): fps и задержка по камерам за окно, переподключения, CPU, RSS."""
        import json

        from qonaq.health.proc import rss_mb

        now = time.time()
        window = max(1.0, now - self._last_stats_wall)
        self._last_stats_wall = now
        cams = {}
        for cid, src in self.pipeline.sources.items():
            cs = self.pipeline.stats.cams.get(cid)
            row = cs.window(now, window) if cs else {"fps": 0.0, "latency_ms_avg": None, "latency_ms_p95": None}
            row.update({"frames": cs.frames if cs else 0, "reconnects": getattr(src, "reconnects", 0),
                        "connected": getattr(src, "connected", True)})
            cams[cid] = row
        rss, kind = rss_mb()
        line = {"wall": round(now, 1), "uptime_s": round(now - self.started_wall, 1), "window_s": round(window, 1),
                "cpu_pct": self._cpu.sample(), "rss_mb": round(rss, 1) if rss is not None else None,
                "rss_kind": kind, "threads": threading.active_count(), "cameras": cams,
                "alerts_pending": self.alerts.pending(),
            "perf": self.perf, "events_pending": self.writer.pending()}
        self.perf = line
        if self.time_base == "epoch":  # B3/B8: «система видела» — для сверки с учётом (перезапуски, обрывы)
            from qonaq.logic.base import Event

            # при остановке источники уже закрыты (connected=False) — считаем по кадрам за окно; `final` отличает
            # штатную остановку от аварийной в простоях (qonaq/health/downtime.py)
            up = [cid for cid, row in cams.items() if row.get("fps", 0) > 0 and (final or row.get("connected", True))]
            payload = {"window_s": round(window, 1), "up": up}
            if final:
                payload["final"] = True
            self.writer.add([Event(now, "", "heartbeat", payload)])
        with (self.run_dir / "live_stats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return line

    def _health_loop(self) -> None:
        next_health = time.time() + self.site.health.check_every_s
        next_stats = time.time() + self.site.health.stats_every_s
        while not self._stop.wait(1.0):
            now = time.time()
            if self.site.health.stats_every_s > 0 and now >= next_stats:
                next_stats = now + self.site.health.stats_every_s
                try:
                    self.perf_tick()
                except Exception:  # noqa: BLE001
                    log.exception("live-статистика")
            if self.site.health.enabled and now >= next_health:
                next_health = now + self.site.health.check_every_s
                try:
                    self.health_tick()
                except Exception:  # noqa: BLE001
                    log.exception("монитор здоровья")

    def start(self) -> LiveService:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()
        if self.site.health.enabled or self.site.health.stats_every_s > 0:
            self._health_thread = threading.Thread(target=self._health_loop, name="health", daemon=True)
            self._health_thread.start()
        from qonaq.alerts.telegram import FeedbackPoller, TelegramSender

        if isinstance(self.alerts.sender, TelegramSender):
            self._feedback = FeedbackPoller(self.alerts.sender, self.store).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._feedback is not None:
            self._feedback.stop()
        if self._thread is not None:
            self._thread.join(timeout=15)
        try:
            self.perf_tick(final=True)  # последняя строка live-статистики
        except Exception:  # noqa: BLE001
            pass
        try:
            (self.run_dir / "health.json").write_text(__import__("json").dumps(
                self.health.status(live=False), ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        self.writer.close()
        last = max((s["ts"] for s in self._state.values()), default=0.0)
        self.store.set_meta(t_end=last, scene_s=self.pipeline.stats.scene_s)
        self.alerts.close()
        self.store.close()
        import json

        stats = self.pipeline.stats
        (self.run_dir / "run.json").write_text(json.dumps({
            "site_id": self.site.site_id, "name": self.site.name, "profile": self.site.profile, "mode": "live",
            "cameras": [c.id for c in self.site.cameras], "detector": getattr(self.pipeline.detector, "name", "?"),
            "frames": stats.frames, "wall_s": round(stats.wall_s, 1), "fps_achieved": round(stats.fps, 2),
            "zones": {cid: [z.id for z in cam.zones.zones] for cid, cam in self.pipeline.cameras.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ read side (API)
    def jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self._jpeg.get(camera_id)

    def state(self) -> dict[str, Any]:
        with self._lock:
            cams = [dict(s) for s in self._state.values()]
        stats = self.pipeline.stats
        elapsed = max(time.time() - self.started_wall, 1e-6)
        return {
            "site": {"id": self.site.site_id, "name": self.site.name, "profile": self.site.profile},
            "cameras": cams,
            "camera_ids": [c.id for c in self.site.cameras],
            "alerts": [a.to_dict() for a in list(self.alerts.feed)[-30:]][::-1],
            "telegram": self.alerts.sender is not None,
            "alerts_held_back": self.alerts.held_back,
            "health": self.health.status(live=True)["status"],
            "alerts_pending": self.alerts.pending(),
            "perf": self.perf,
            "time_base": self.time_base,
            "fps": round(stats.frames / elapsed, 2),
            "running": not self.finished.is_set(),
            "error": self.error,
            "server_time": time.time(),
        }

    def events(self, since_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        evs = self.store.events(since_id=since_id, limit=limit, newest_first=since_id == 0)
        return [e for e in evs if e["type"] != "heartbeat"]  # служебные отметки «система видела» — не для ленты
