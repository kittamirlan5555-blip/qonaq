"""Поиск камер (S3): ONVIF WS-Discovery в локальной сети + перебор RTSP-шаблонов регистраторов по каналам."""
from __future__ import annotations

import logging
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

from qonaq.setup.rtsp import RtspProbe, describe, tcp_open

log = logging.getLogger("qonaq.setup")

# Субпоток (второй поток) — его хватает и он не грузит регистратор. Шаблоны — docs/KZ_REALITIES.md.
TEMPLATES: dict[str, str] = {
    "hikvision": "rtsp://{u}:{p}@{ip}:{port}/Streaming/Channels/{ch}02",  # и HiWatch
    "dahua": "rtsp://{u}:{p}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",  # и Imou
    "uniview": "rtsp://{u}:{p}@{ip}:{port}/unicast/c{ch}/s1/live",
    "xmeye": "rtsp://{ip}:{port}/user={u}&password={p}&channel={ch}&stream=1.sdp?real_stream",  # и iCSee
}
VENDOR_RU = {"hikvision": "Hikvision / HiWatch", "dahua": "Dahua / Imou", "uniview": "Uniview",
             "xmeye": "XMeye / iCSee"}
# Шаблоны для конфига: логин/пароль/адрес — из .env
ENV_TEMPLATES = {k: v.replace("{u}", "${QONAQ_NVR_USER}").replace("{p}", "${QONAQ_NVR_PASSWORD}")
                 .replace("{ip}", "${QONAQ_NVR_IP}") for k, v in TEMPLATES.items()}


# В URL кодируем только то, что ломает разбор адреса (@ : / ? # % пробел); прочие символы пароля — как есть,
# потому что не каждая прошивка/FFmpeg раскодирует %XX в логине и пароле.
URL_SAFE = "!$&'()*+,;=-._~"


def enc(s: str) -> str:
    return quote(s, safe=URL_SAFE)


def build_url(vendor: str, ip: str, user: str, password: str, ch: int, port: int = 554, env: bool = False) -> str:
    tpl = ENV_TEMPLATES[vendor] if env else TEMPLATES[vendor]
    return tpl.format(u=enc(user), p=enc(password), ip=ip, port=port, ch=ch)


@dataclass
class ScanResult:
    ip: str
    port: int
    vendor: str | None = None
    channels: list[int] = field(default_factory=list)
    probes: list[RtspProbe] = field(default_factory=list)
    error: str = ""

    def url(self, ch: int, user: str, password: str, env: bool = False) -> str:
        return build_url(self.vendor or "hikvision", "${QONAQ_NVR_IP}" if env else self.ip, user, password, ch,
                         self.port, env)


def detect_vendor(ip: str, user: str, password: str, port: int = 554, vendor: str = "auto",
                  probe=describe) -> ScanResult:
    """Канал 1 по шаблонам. Два ответа 401 подряд — стоп: неверный пароль (регистраторы блокируют
    пользователя после 5–7 неудачных входов, перебирать дальше опасно)."""
    res = ScanResult(ip, port)
    if not tcp_open(ip, port):
        res.error = f"{ip}:{port} не отвечает (порт RTSP закрыт, неверный адрес или другой порт)"
        return res
    order = list(TEMPLATES) if vendor == "auto" else [vendor]
    unauthorized = 0
    for v in order:
        r = probe(build_url(v, ip, user, password, 1, port))
        res.probes.append(r)
        log.info("  %-10s канал 1: %s", v, r.explain())
        if r.ok:
            res.vendor = v
            return res
        if r.status == 401 and v != "xmeye":
            unauthorized += 1
            if unauthorized >= 2 or vendor != "auto":
                res.error = "неверный логин или пароль регистратора (дальше не перебираю, чтобы его не заблокировать)"
                return res
    res.error = "ни один шаблон не подошёл: проверьте адрес/порт или пришлите модель регистратора"
    return res


def scan_channels(res: ScanResult, user: str, password: str, max_channels: int = 16, stop_after_misses: int = 3,
                  probe=describe) -> ScanResult:
    """Каналы 1..max подряд; после stop_after_misses отсутствующих подряд — стоп."""
    misses = 0
    for ch in range(1, max_channels + 1):
        r = probe(build_url(res.vendor, res.ip, user, password, ch, res.port))
        if r.ok:
            res.channels.append(ch)
            misses = 0
        else:
            misses += 1
            if r.status == 401:
                break
            if misses >= stop_after_misses:
                break
    return res


# ---------------------------------------------------------------- ONVIF WS-Discovery
PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{mid}</w:MessageID>
<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"""


def parse_probe_match(xml: str) -> list[dict[str, str]]:
    """Из ответа ProbeMatch: адреса сервиса (XAddrs) и scopes (часто там модель/производитель)."""
    out = []
    for xaddrs in re.findall(r"<[^>]*XAddrs>([^<]+)</[^>]*XAddrs>", xml):
        scopes = " ".join(re.findall(r"<[^>]*Scopes[^>]*>([^<]+)</[^>]*Scopes>", xml))
        hw = re.search(r"onvif://www\.onvif\.org/hardware/([^\s]+)", scopes)
        name = re.search(r"onvif://www\.onvif\.org/name/([^\s]+)", scopes)
        for x in xaddrs.split():
            host = urlsplit(x).hostname
            if host:
                out.append({"ip": host, "xaddr": x, "hardware": hw.group(1) if hw else "",
                            "name": name.group(1) if name else ""})
    return out


def onvif_discover(timeout: float = 3.0) -> list[dict[str, str]]:
    """Multicast Probe 239.255.255.250:3702. Находит IP-камеры и регистраторы с включённым ONVIF в этой подсети."""
    msg = PROBE.format(mid=uuid.uuid4()).encode()
    found: dict[str, dict[str, str]] = {}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.settimeout(0.5)
        s.sendto(msg, ("239.255.255.250", 3702))
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                data, addr = s.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            for d in parse_probe_match(data.decode("utf-8", "replace")) or [{"ip": addr[0], "xaddr": "",
                                                                             "hardware": "", "name": ""}]:
                found.setdefault(d["ip"], d)
        s.close()
    except OSError as e:
        log.info("ONVIF-поиск недоступен: %s", e)
    return sorted(found.values(), key=lambda d: tuple(int(x) if x.isdigit() else 0 for x in d["ip"].split(".")))
