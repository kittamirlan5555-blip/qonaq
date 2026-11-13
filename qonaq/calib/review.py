"""`calibrate review`: локальная страница проверки — сетка треков (лица размыты), клики «персонал / гость / пропустить».

    python -m qonaq calibrate review --site-id cafe1      # http://127.0.0.1:8010

Сначала — треки без авторазметки (неуверенные), затем — авторазмеченные для выборочной проверки.
Метки сохраняются сразу в data/calib/<site>/labels.json.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from qonaq.calib.store import calib_dir, load_labels, load_tracks, save_label

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qonaq — калибровка зала</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#f5f5f4;color:#1c1917}
h1{font-size:20px;margin:0 0 4px} .muted{color:#57534e;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:12px}
.card{background:#fff;border-radius:10px;padding:8px;border:2px solid transparent}
.card.staff{border-color:#2563eb}.card.guest{border-color:#16a34a}.card.skip{opacity:.45}
.thumbs{display:flex;gap:4px;overflow:hidden;height:120px}.thumbs img{height:120px;border-radius:4px}
.btns{display:flex;gap:6px;margin-top:6px}.btns button{flex:1;padding:8px 0;border:0;border-radius:6px;
font-size:14px;cursor:pointer;background:#e7e5e4}.btns .s{background:#dbeafe}.btns .g{background:#dcfce7}
.meta{font-size:12px;color:#57534e;margin-top:4px} .bar{position:sticky;top:0;background:#f5f5f4;padding:6px 0}
</style></head><body><div class="bar"><h1>Калибровка зала: <span id="site"></span></h1>
<div class="muted">Клавиши: <b>1</b> — персонал, <b>2</b> — гость, <b>0</b> — пропустить (для карточки под курсором).
Размечено: <b id="done">0</b> из <b id="total">0</b>. Лица размыты; смотрите на одежду.</div></div>
<div class="grid" id="grid"></div>
<script>
let hover=null;
async function load(){const r=await fetch('api/tracks');const d=await r.json();
document.getElementById('site').textContent=d.site;document.getElementById('total').textContent=d.tracks.length;
const g=document.getElementById('grid');g.innerHTML='';let done=0;
for(const t of d.tracks){const c=document.createElement('div');c.className='card '+(t.label||'');c.id=t.key;
if(t.label)done++;
c.innerHTML='<div class="thumbs">'+t.thumbs.map(n=>'<img loading="lazy" src="crops/'+n+'">').join('')+'</div>'+
'<div class="btns"><button class="s">1 персонал</button><button class="g">2 гость</button><button>0 —</button></div>'+
'<div class="meta">'+t.key+' · '+(t.auto?('авто: '+(t.auto=='staff'?'персонал':'гость')):'неуверенный')+
' · сидел '+Math.round(t.seated_dwell_s)+' с · столов '+t.max_tables+
' · стоя '+Math.round(100*t.standing_share)+'%</div>';
const b=c.querySelectorAll('button');b[0].onclick=()=>lab(t.key,'staff');b[1].onclick=()=>lab(t.key,'guest');
b[2].onclick=()=>lab(t.key,'skip');c.onmouseenter=()=>hover=t.key;g.appendChild(c);}
document.getElementById('done').textContent=done;}
async function lab(k,l){await fetch('api/label',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({key:k,label:l})});const c=document.getElementById(k);c.className='card '+l;
document.getElementById('done').textContent=document.querySelectorAll('.card.staff,.card.guest,.card.skip').length;}
document.addEventListener('keydown',e=>{if(!hover)return;if(e.key=='1')lab(hover,'staff');
if(e.key=='2')lab(hover,'guest');if(e.key=='0')lab(hover,'skip');});
load();
</script></body></html>"""


def review_items(site_id: str, root: Path | None = None, per_track: int = 4) -> list[dict]:
    labels = load_labels(site_id, root)
    items = []
    for t in load_tracks(site_id, root):
        th = t.get("thumbs") or []
        if not th:
            continue
        step = max(1, len(th) // per_track)
        items.append({**{k: t[k] for k in ("key", "auto", "seated_dwell_s", "max_tables", "standing_share")},
                      "thumbs": th[::step][:per_track], "label": labels.get(t["key"])})
    # неуверенные — первыми, затем длинные треки
    items.sort(key=lambda t: (t["auto"] is not None, -len(t["thumbs"])))
    return items


class Lab(BaseModel):
    key: str
    label: str


def create_app(site_id: str, root: Path | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    app = FastAPI(title="Qonaq calibrate")
    crops = calib_dir(site_id, root) / "crops"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/tracks")
    def tracks():
        return {"site": site_id, "tracks": review_items(site_id, root)}

    @app.post("/api/label")
    def label(b: Lab):
        if b.label not in ("staff", "guest", "skip"):
            raise HTTPException(400, "label: staff | guest | skip")
        save_label(site_id, b.key, b.label, root)
        return {"ok": True}

    @app.get("/crops/{name}")
    def crop(name: str):
        p = (crops / name).resolve()
        if p.parent != crops.resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    return app


def serve(site_id: str, host: str = "127.0.0.1", port: int = 8010) -> None:
    import uvicorn

    print(f"Страница проверки: http://{host}:{port}   (Ctrl+C — закончить; метки уже сохранены)")
    uvicorn.run(create_app(site_id), host=host, port=port, log_level="warning")


__all__ = ["create_app", "review_items", "serve"]
