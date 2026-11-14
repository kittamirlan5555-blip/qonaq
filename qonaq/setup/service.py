"""`python -m qonaq service install|status|uninstall|restart|logs` (S3): live как служба с автоперезапуском.

Linux — systemd: системный unit (под root) или пользовательский (`systemctl --user` + linger, без root).
Windows — Планировщик задач: задача «при запуске системы» от SYSTEM запускает cmd-обёртку, которая
перезапускает live через 10 с после любого падения; лог — runs/service_<объект>.log (ротация 10 МБ × 5).
`--dry-run` — только показать, что будет создано.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qonaq.config import PROJECT_ROOT


def service_name(site_id: str) -> str:
    return f"qonaq-{site_id}".replace("_", "-")


def live_args(site: str, host: str | None, port: int | None) -> list[str]:
    a = ["-m", "qonaq", "live", "--site", site]
    if host:
        a += ["--host", host]
    if port:
        a += ["--port", str(port)]
    return a


def systemd_unit(site_id: str, site: str, python: str, root: Path, user_mode: bool, user: str | None,
                 host: str | None = None, port: int | None = None) -> str:
    args = " ".join(live_args(site, host, port))
    lines = [
        "[Unit]",
        f"Description=Qonaq Vision — {site_id}",
        "Wants=network-online.target",
        "After=network-online.target",
        "StartLimitIntervalSec=0",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={root}",
        f"ExecStart={python} {args}",
        "Restart=always",
        "RestartSec=10",
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=QONAQ_LOG_FILE={root / 'runs' / f'service_{site_id}.log'}",
        "TimeoutStopSec=30",
        # мягкие лимиты: служба не должна съесть всю память мини-ПК при утечке
        "MemoryMax=3G",
        "Nice=5",
    ]
    if not user_mode and user:
        lines.append(f"User={user}")
    lines += ["", "[Install]", "WantedBy=default.target" if user_mode else "WantedBy=multi-user.target", ""]
    return "\n".join(lines)


def windows_wrapper(site_id: str, site: str, python: str, root: Path, host: str | None = None,
                    port: int | None = None) -> str:
    args = " ".join(f'"{a}"' if " " in a else a for a in live_args(site, host, port))
    log = root / "runs" / f"service_{site_id}.log"
    return "\r\n".join([
        "@echo off",
        "chcp 65001 >nul",
        f'cd /d "{root}"',
        "set PYTHONUNBUFFERED=1",
        "set PYTHONIOENCODING=utf-8",
        f'set QONAQ_LOG_FILE={log}',
        ":loop",
        f'"{python}" {args}',
        f'echo %date% %time% live завершился с кодом %errorlevel%, перезапуск через 10 с >> "{log}"',
        # `timeout` без консоли (задача планировщика) сразу завершается — пауза через ping
        "ping -n 11 127.0.0.1 >nul",
        "goto loop",
        "",
    ])


def _python() -> str:
    return sys.executable


def _run(cmd: list[str], dry: bool) -> int:
    print("$ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, check=False).returncode


def install(args) -> int:
    from qonaq.config import load_site

    site_path = Path(args.site)
    site = load_site(site_path)
    rel = site_path.resolve().relative_to(PROJECT_ROOT).as_posix() if site_path.resolve().is_relative_to(
        PROJECT_ROOT) else str(site_path.resolve())
    name = service_name(site.site_id)
    (PROJECT_ROOT / "runs").mkdir(exist_ok=True)
    if platform.system() == "Windows":
        wrapper = PROJECT_ROOT / "runs" / f"{name}.cmd"
        text = windows_wrapper(site.site_id, rel, _python(), PROJECT_ROOT, args.host, args.port)
        print(f"Обёртка {wrapper}:\n{text}")
        if not args.dry_run:
            wrapper.write_text(text, encoding="utf-8")
        rc = _run(["schtasks", "/Create", "/F", "/TN", name, "/SC", "ONSTART", "/RU", "SYSTEM", "/RL", "HIGHEST",
                   "/TR", f'"{wrapper}"'], args.dry_run)
        if rc == 0 and args.now:
            rc = _run(["schtasks", "/Run", "/TN", name], args.dry_run)
        if rc != 0:
            print("Не получилось: запустите PowerShell «от имени администратора» и повторите.")
        return rc
    user_mode = args.user_mode or (hasattr(os, "geteuid") and os.geteuid() != 0)
    unit = systemd_unit(site.site_id, rel, _python(), PROJECT_ROOT, user_mode, args.run_as, args.host, args.port)
    unit_dir = Path.home() / ".config" / "systemd" / "user" if user_mode else Path("/etc/systemd/system")
    path = unit_dir / f"{name}.service"
    print(f"Unit {path}:\n{unit}")
    if not args.dry_run:
        unit_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(unit, encoding="utf-8")
    sc = ["systemctl", "--user"] if user_mode else ["systemctl"]
    rc = _run(sc + ["daemon-reload"], args.dry_run)
    rc = rc or _run(sc + ["enable", "--now" if args.now else "--quiet", name], args.dry_run)
    if user_mode and shutil.which("loginctl"):
        # без linger пользовательские службы останавливаются при выходе из сессии и не стартуют при загрузке
        _run(["loginctl", "enable-linger", os.environ.get("USER", "")], args.dry_run)
    if rc == 0:
        print(f"\nГотово. Статус: python -m qonaq service status --site {rel}")
    return rc


def _target(args) -> tuple[str, bool]:
    from qonaq.config import load_site

    site = load_site(args.site)
    return service_name(site.site_id), args.user_mode or (hasattr(os, "geteuid") and os.geteuid() != 0)


def status(args) -> int:
    name, user_mode = _target(args)
    if platform.system() == "Windows":
        return _run(["schtasks", "/Query", "/TN", name, "/V", "/FO", "LIST"], args.dry_run)
    rc = _run((["systemctl", "--user"] if user_mode else ["systemctl"]) + ["status", "--no-pager", name], args.dry_run)
    health = sorted(PROJECT_ROOT.glob("runs/live_*/health.json"), key=lambda p: p.stat().st_mtime)
    if health:
        print(f"\nПоследний health.json: {health[-1]}\n{health[-1].read_text(encoding='utf-8')[:1500]}")
    return rc


def uninstall(args) -> int:
    name, user_mode = _target(args)
    if platform.system() == "Windows":
        _run(["schtasks", "/End", "/TN", name], args.dry_run)
        return _run(["schtasks", "/Delete", "/F", "/TN", name], args.dry_run)
    sc = ["systemctl", "--user"] if user_mode else ["systemctl"]
    _run(sc + ["disable", "--now", name], args.dry_run)
    unit_dir = Path.home() / ".config" / "systemd" / "user" if user_mode else Path("/etc/systemd/system")
    p = unit_dir / f"{name}.service"
    if p.exists() and not args.dry_run:
        p.unlink()
    return _run(sc + ["daemon-reload"], args.dry_run)


def restart(args) -> int:
    name, user_mode = _target(args)
    if platform.system() == "Windows":
        _run(["schtasks", "/End", "/TN", name], args.dry_run)
        return _run(["schtasks", "/Run", "/TN", name], args.dry_run)
    return _run((["systemctl", "--user"] if user_mode else ["systemctl"]) + ["restart", name], args.dry_run)


def logs(args) -> int:
    from qonaq.config import load_site

    site = load_site(args.site)
    log = PROJECT_ROOT / "runs" / f"service_{site.site_id}.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lines:]
        print("\n".join(lines))
        return 0
    print(f"нет файла {log}")
    return 1


def add_parser(sub) -> None:
    sp = sub.add_parser("service", help="live как служба: install | status | uninstall | restart | logs (S3)")
    s = sp.add_subparsers(dest="svc_cmd", required=True)
    for cmd, hlp in [("install", "установить и запустить"), ("status", "состояние службы и health"),
                     ("uninstall", "остановить и удалить"), ("restart", "перезапустить (после правки .env/зон)"),
                     ("logs", "последние строки лога")]:
        p = s.add_parser(cmd, help=hlp)
        p.add_argument("--site", required=True, help="YAML объекта")
        p.add_argument("--dry-run", action="store_true", help="только показать команды и файлы")
        p.add_argument("--user-mode", action="store_true", help="Linux: пользовательская служба (без root)")
        if cmd == "install":
            p.add_argument("--run-as", help="Linux, системная служба: от какого пользователя запускать")
            p.add_argument("--host", help="адрес панели (0.0.0.0 — видно в локальной сети)")
            p.add_argument("--port", type=int)
            p.add_argument("--no-now", dest="now", action="store_false", help="не запускать сразу")
        if cmd == "logs":
            p.add_argument("--lines", type=int, default=80)


def run(args) -> int:
    return {"install": install, "status": status, "uninstall": uninstall, "restart": restart,
            "logs": logs}[args.svc_cmd](args)
