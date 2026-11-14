"""Быстрая проверка RTSP-адреса без декодирования видео (S3): TCP-подключение + RTSP DESCRIBE с Basic/Digest.

Зачем: OpenCV на неверном пути/пароле молча возвращает «не открылось» через несколько секунд. DESCRIBE за
доли секунды отвечает 200 (поток есть), 401 (неверный логин/пароль), 404 (нет такого канала) — установщик
сразу видит, что не так, и перебор шаблонов регистраторов идёт быстро.
"""
from __future__ import annotations

import base64
import hashlib
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit


@dataclass
class RtspProbe:
    url: str
    status: int  # 200 | 401 | 404 | ... ; 0 — нет TCP-соединения; -1 — не RTSP-ответ/таймаут
    reason: str = ""
    server: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200

    def explain(self) -> str:
        return {
            200: "поток есть",
            401: "неверный логин или пароль",
            403: "доступ запрещён (пользователю не разрешён RTSP?)",
            404: "нет такого канала/пути",
            454: "нет такого канала (сессия не найдена)",
            0: "порт закрыт или адрес недоступен",
            -1: "не отвечает как RTSP",
        }.get(self.status, f"ответ {self.status} {self.reason}")


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _split(url: str) -> tuple[str, int, str, str, str]:
    """-> host, port, user, password, url без логина/пароля."""
    u = urlsplit(url)
    host = u.hostname or ""
    port = u.port or 554
    user = unquote(u.username or "")
    pw = unquote(u.password or "")
    netloc = host if port == 554 else f"{host}:{port}"
    clean = f"{u.scheme}://{netloc}{u.path or '/'}" + (f"?{u.query}" if u.query else "")
    return host, port, user, pw, clean


def _request(sock: socket.socket, url: str, cseq: int, auth: str | None) -> tuple[int, str, dict[str, str]]:
    lines = [f"DESCRIBE {url} RTSP/1.0", f"CSeq: {cseq}", "Accept: application/sdp", "User-Agent: qonaq-setup"]
    if auth:
        lines.append(f"Authorization: {auth}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    first, *rest = head.split("\r\n") if head else [""]
    m = re.match(r"RTSP/1\.\d\s+(\d{3})\s*(.*)", first)
    if not m:
        return -1, "", {}
    headers: dict[str, str] = {}
    for ln in rest:
        if ":" in ln:
            k, v = ln.split(":", 1)
            key = k.strip().lower()
            headers[key] = (headers[key] + ", " + v.strip()) if key in headers else v.strip()
    return int(m.group(1)), m.group(2), headers


def _digest(user: str, pw: str, url: str, challenge: str) -> str:
    params = dict(re.findall(r'(\w+)="?([^",]*)"?', challenge))
    realm, nonce = params.get("realm", ""), params.get("nonce", "")
    ha1 = hashlib.md5(f"{user}:{realm}:{pw}".encode()).hexdigest()  # noqa: S324 — так требует RTSP Digest
    ha2 = hashlib.md5(f"DESCRIBE:{url}".encode()).hexdigest()  # noqa: S324
    resp = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # noqa: S324
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{url}", response="{resp}"'


def describe(url: str, timeout: float = 3.0) -> RtspProbe:
    """DESCRIBE с авторизацией (Basic или Digest — как попросит сервер)."""
    host, port, user, pw, clean = _split(url)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as e:
        return RtspProbe(url, 0, str(e))
    try:
        sock.settimeout(timeout)
        code, reason, h = _request(sock, clean, 1, None)
        if code == 401 and user:
            chal = h.get("www-authenticate", "")
            if "digest" in chal.lower():
                dig = next((c for c in re.split(r",\s*(?=Basic|Digest)", chal) if c.lower().startswith("digest")),
                           chal)
                auth = _digest(user, pw, clean, dig)
            else:
                auth = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
            code, reason, h = _request(sock, clean, 2, auth)
        return RtspProbe(url, code, reason, h.get("server", ""))
    except (OSError, ValueError) as e:
        return RtspProbe(url, -1, str(e))
    finally:
        sock.close()
