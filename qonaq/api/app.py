"""FastAPI-панель: `/` — страница, `/video/<camera>` — MJPEG (с блюром), `/api/state`, `/api/events`,
`/api/stream` — SSE с состоянием раз в секунду."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from qonaq.api.service import LiveService

STATIC = Path(__file__).parent / "static"


def create_app(service: LiveService) -> FastAPI:
    app = FastAPI(title="Qonaq Vision", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/state")
    def state():
        return JSONResponse(service.state())

    @app.get("/api/events")
    def events(since_id: int = 0, limit: int = 100):
        return JSONResponse(service.events(since_id=since_id, limit=min(max(limit, 1), 1000)))

    @app.get("/api/stream")
    async def stream():
        async def gen():
            while True:
                yield f"data: {json.dumps(service.state(), ensure_ascii=False)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/frame/{camera_id}.jpg")
    def frame(camera_id: str):
        data = service.jpeg(camera_id)
        if data is None:
            raise HTTPException(404, "кадра ещё нет")
        from fastapi.responses import Response

        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/video/{camera_id}")
    async def video(camera_id: str):
        if camera_id not in service.pipeline.cameras:
            raise HTTPException(404, "нет такой камеры")

        async def gen():
            last = None
            while True:
                data = service.jpeg(camera_id)
                if data is not None and data is not last:
                    last = data
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(0.05)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
