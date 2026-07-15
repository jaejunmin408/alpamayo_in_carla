#!/usr/bin/env python
"""디버그용 2D 맵 viz 서버.

차량이 처음 스폰된 위치를 맵상 (0,0) 으로 잡고, 이후 이동을 top-down 맵에
실시간으로 그린다. tesla_camera.py 등 시뮬 루프에서 매 프레임 update() 로
현재 pose(CARLA world x,y[m], yaw[deg])를 넘기면:

  - 첫 update 의 위치를 원점으로 기록하고 이후 좌표를 원점 기준으로 rebase
  - 최신 pose + 이동 궤적(trail)을 thread-safe 하게 보관
  - HTTP 로 HTML 맵( / )과 JSON pose( /pose )를 제공

브라우저에서 http://localhost:<port>/ 를 열면 canvas 맵이 /pose 를
폴링하며 차를 움직인다. alpamayo_bridge.py 의 ThreadingHTTPServer 패턴과 동일.

사용 예 (tesla_camera.py 안):
    viz = VizServer(port=8091); viz.start()
    ...
    viz.update(tr.location.x, tr.location.y, tr.rotation.yaw, speed_mps)
    ...
    viz.stop()
"""
from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 궤적 링버퍼 최대 점 수. 초과 시 앞에서부터 버린다.
MAX_TRAIL = 8000
# trail 에 점을 추가하는 최소 이동거리(m). 이보다 덜 움직였으면 최신 pose 만 갱신.
TRAIL_MIN_STEP_M = 0.25


class VizServer:
    """스폰 원점 기준 2D 맵 viz 를 제공하는 경량 HTTP 서버."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8091):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._origin: tuple[float, float] | None = None  # (x0, y0) world[m]
        self._x = 0.0            # 원점 기준 현재 위치[m]
        self._y = 0.0
        self._yaw = 0.0          # deg (CARLA world yaw)
        self._speed = 0.0        # m/s
        self._trail: list[tuple[float, float]] = []
        # 고정 경로: 받은 순간의 CARLA world 좌표로 박제. snapshot 에서 원점 기준 rebase.
        self._fixed_path: list[tuple[float, float]] = []
        self._seq = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def update(self, x: float, y: float, yaw_deg: float,
               speed_mps: float = 0.0) -> None:
        """시뮬 루프에서 매 프레임 호출. world 좌표를 원점 기준으로 rebase."""
        with self._lock:
            if self._origin is None:
                self._origin = (float(x), float(y))
            rx = float(x) - self._origin[0]
            ry = float(y) - self._origin[1]
            self._x, self._y = rx, ry
            self._yaw = float(yaw_deg)
            self._speed = float(speed_mps)
            self._seq += 1
            if not self._trail:
                self._trail.append((rx, ry))
            else:
                lx, ly = self._trail[-1]
                if math.hypot(rx - lx, ry - ly) >= TRAIL_MIN_STEP_M:
                    self._trail.append((rx, ry))
                    if len(self._trail) > MAX_TRAIL:
                        del self._trail[0:len(self._trail) - MAX_TRAIL]

    def set_fixed_path(self, world_points) -> None:
        """추종할 경로를 CARLA world 좌표 [(wx, wy), ...] 로 '고정' 저장.

        받은 순간의 좌표에 박제되므로, 차가 이 경로 위를 지나가는 것으로
        추종 여부를 눈으로 확인할 수 있다. 새 경로가 오면 교체, 빈 값이면 지움.
        (원점 기준 rebase 는 snapshot 에서 수행 → trail/차량과 동일 프레임)
        """
        with self._lock:
            if not world_points:
                self._fixed_path = []
            else:
                self._fixed_path = [(float(p[0]), float(p[1]))
                                    for p in world_points if len(p) >= 2]

    def reset(self) -> None:
        """원점/궤적 초기화. 다음 update 위치가 새 (0,0) 이 된다."""
        with self._lock:
            self._origin = None
            self._trail = []
            self._fixed_path = []
            self._x = self._y = self._yaw = self._speed = 0.0

    def _snapshot(self) -> dict:
        with self._lock:
            ox, oy = self._origin if self._origin is not None else (0.0, 0.0)
            return {
                "seq": self._seq,
                "has_origin": self._origin is not None,
                "x": round(self._x, 3),
                "y": round(self._y, 3),
                "yaw_deg": round(self._yaw, 2),
                "speed_mps": round(self._speed, 3),
                "speed_kmh": round(self._speed * 3.6, 2),
                "dist_m": round(math.hypot(self._x, self._y), 2),
                "trail": [[round(px, 2), round(py, 2)] for px, py in self._trail],
                "path": [[round(px - ox, 3), round(py - oy, 3)]
                         for px, py in self._fixed_path],
            }

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 콘솔 조용히
                pass

            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0].rstrip("/")
                if path in ("", "/index.html"):
                    self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/pose":
                    body = json.dumps(server._snapshot()).encode("utf-8")
                    self._send(200, body, "application/json")
                elif path == "/reset":
                    server.reset()
                    self._send(200, b'{"ok":true}', "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        shown = "localhost" if self.host in ("0.0.0.0", "") else self.host
        print(f"[viz] 맵 viz 서버 시작: http://{shown}:{self.port}/")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            print("[viz] 맵 viz 서버 종료")


_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CARLA 차량 위치 맵 (디버그)</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #0e1116; color: #e6edf3;
               font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  #wrap { position: fixed; inset: 0; }
  canvas { display: block; width: 100%; height: 100%; }
  #hud { position: fixed; top: 12px; left: 12px; padding: 10px 14px;
         background: rgba(20,24,31,.82); border: 1px solid #2b333d;
         border-radius: 8px; font-size: 13px; line-height: 1.6;
         pointer-events: none; white-space: pre; }
  #hud b { color: #58a6ff; }
  #hint { position: fixed; bottom: 10px; left: 12px; font-size: 12px;
          color: #8b949e; }
  #dot { color: #f0b429; }
  .warn { color: #f85149; }
</style>
</head>
<body>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="hud">연결 중…</div>
<div id="hint">마우스휠 줌 · 드래그 이동 · F 팔로우 · G 그리드 · R 원점리셋 · C 궤적중앙</div>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const hud = document.getElementById('hud');

let scale = 6;              // px per meter
let follow = true;         // 차를 화면 중앙에 고정
let showGrid = true;
let cam = {x: 0, y: 0};    // 화면 중앙이 바라보는 월드 좌표(m)
let drag = null;
let state = {has_origin:false, x:0, y:0, yaw_deg:0, speed_kmh:0,
             dist_m:0, trail:[], path:[], seq:0};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.floor(innerWidth * dpr);
  cv.height = Math.floor(innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener('resize', resize);
resize();

// world(m) -> screen(px). CARLA 는 좌수 좌표계(+y 오른쪽), canvas 도 y 아래로
// 증가하므로 (x,y) 를 그대로 쓰면 top-down 이 자연스럽다.
function w2s(wx, wy) {
  return [innerWidth / 2 + (wx - cam.x) * scale,
          innerHeight / 2 + (wy - cam.y) * scale];
}

cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  scale = Math.min(60, Math.max(0.5, scale * f));
}, {passive: false});

cv.addEventListener('mousedown', (e) => {
  drag = {sx: e.clientX, sy: e.clientY, cx: cam.x, cy: cam.y};
  follow = false;
});
addEventListener('mousemove', (e) => {
  if (!drag) return;
  cam.x = drag.cx - (e.clientX - drag.sx) / scale;
  cam.y = drag.cy - (e.clientY - drag.sy) / scale;
});
addEventListener('mouseup', () => drag = null);

addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'f') follow = !follow;
  else if (k === 'g') showGrid = !showGrid;
  else if (k === 'r') fetch('/reset').catch(()=>{});
  else if (k === 'c') fitTrail();
});

function fitTrail() {
  if (!state.trail.length) return;
  let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
  for (const [x,y] of state.trail) {
    minx=Math.min(minx,x); maxx=Math.max(maxx,x);
    miny=Math.min(miny,y); maxy=Math.max(maxy,y);
  }
  cam.x = (minx+maxx)/2; cam.y = (miny+maxy)/2;
  const w = Math.max(maxx-minx, 5), h = Math.max(maxy-miny, 5);
  scale = Math.min(60, Math.max(0.5,
      0.85 * Math.min(innerWidth / w, innerHeight / h)));
  follow = false;
}

function niceStep(px) {
  // 화면상 대략 목표 픽셀 간격이 되도록 1/2/5 * 10^n 격자 간격(m) 선택
  const target = 90;
  let m = target / scale;
  const p = Math.pow(10, Math.floor(Math.log10(m)));
  const c = m / p;
  const mult = c < 1.5 ? 1 : c < 3.5 ? 2 : c < 7.5 ? 5 : 10;
  return mult * p;
}

function drawGrid() {
  const step = niceStep();
  const x0 = cam.x - innerWidth / 2 / scale;
  const x1 = cam.x + innerWidth / 2 / scale;
  const y0 = cam.y - innerHeight / 2 / scale;
  const y1 = cam.y + innerHeight / 2 / scale;
  ctx.lineWidth = 1;
  ctx.font = '11px monospace';
  for (let gx = Math.ceil(x0 / step) * step; gx <= x1; gx += step) {
    const [sx] = w2s(gx, 0);
    ctx.strokeStyle = Math.abs(gx) < 1e-6 ? '#3d4a5c' : '#1b2029';
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, innerHeight); ctx.stroke();
    ctx.fillStyle = '#4b5563';
    ctx.fillText(gx.toFixed(0) + 'm', sx + 3, 12);
  }
  for (let gy = Math.ceil(y0 / step) * step; gy <= y1; gy += step) {
    const [, sy] = w2s(0, gy);
    ctx.strokeStyle = Math.abs(gy) < 1e-6 ? '#3d4a5c' : '#1b2029';
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(innerWidth, sy); ctx.stroke();
    ctx.fillStyle = '#4b5563';
    ctx.fillText(gy.toFixed(0) + 'm', 3, sy - 3);
  }
}

function drawOrigin() {
  const [ox, oy] = w2s(0, 0);
  ctx.strokeStyle = '#f0b429'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(ox - 8, oy); ctx.lineTo(ox + 8, oy);
  ctx.moveTo(ox, oy - 8); ctx.lineTo(ox, oy + 8);
  ctx.stroke();
  ctx.fillStyle = '#f0b429'; ctx.font = '11px monospace';
  ctx.fillText('spawn (0,0)', ox + 10, oy - 6);
}

function drawTrail() {
  if (state.trail.length < 2) return;
  ctx.strokeStyle = '#2f81f7'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < state.trail.length; i++) {
    const [sx, sy] = w2s(state.trail[i][0], state.trail[i][1]);
    i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
  }
  ctx.stroke();
}

// 고정 경로: 받은 순간의 map 좌표(원점 기준)에 박제된 경로를 그대로 그린다.
// 차가 이 위를 지나가면 추종 성공. (trail 과 같은 프레임이라 직접 w2s)
function drawPlan() {
  const p = state.path;
  if (!state.has_origin || !p || p.length < 2) return;
  // 경로선
  ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  for (let i = 0; i < p.length; i++) {
    const [sx, sy] = w2s(p[i][0], p[i][1]);
    i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // 경로 점(듬성듬성)
  ctx.fillStyle = '#f0883e';
  const stepd = Math.max(1, Math.floor(p.length / 24));
  for (let i = 0; i < p.length; i += stepd) {
    const [sx, sy] = w2s(p[i][0], p[i][1]);
    ctx.beginPath(); ctx.arc(sx, sy, 2.2, 0, 2 * Math.PI); ctx.fill();
  }
  // 경로 끝점 강조(도달 목표)
  const [ex, ey] = w2s(p[p.length - 1][0], p[p.length - 1][1]);
  ctx.beginPath(); ctx.arc(ex, ey, 4.5, 0, 2 * Math.PI); ctx.fill();
}

function drawCar() {
  if (!state.has_origin) return;
  const [sx, sy] = w2s(state.x, state.y);
  const yaw = state.yaw_deg * Math.PI / 180;  // 0=+x(오른쪽), +는 시계방향
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(yaw);
  // 진행방향(+x)을 향한 삼각형
  ctx.fillStyle = '#3fb950';
  ctx.strokeStyle = '#0e1116'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(13, 0); ctx.lineTo(-8, 7); ctx.lineTo(-8, -7); ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.restore();
}

function render() {
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, innerWidth, innerHeight);
  if (follow && state.has_origin) { cam.x = state.x; cam.y = state.y; }
  if (showGrid) drawGrid();
  drawOrigin();
  drawTrail();
  drawPlan();
  drawCar();
  const ok = state.has_origin;
  const np = (state.path && state.path.length) || 0;
  hud.innerHTML =
    (ok ? '' : '<span class="warn">차량 pose 대기 중…</span>\n') +
    `x   <b>${state.x.toFixed(2)}</b> m\n` +
    `y   <b>${state.y.toFixed(2)}</b> m\n` +
    `yaw <b>${state.yaw_deg.toFixed(1)}</b>°\n` +
    `속도 <b>${state.speed_kmh.toFixed(1)}</b> km/h\n` +
    `원점거리 <b>${state.dist_m.toFixed(1)}</b> m\n` +
    `<span style="color:#2f81f7">━ 주행궤적</span>  ` +
    `<span style="color:#f0883e">┅ 고정경로(${np})</span>\n` +
    `zoom ${scale.toFixed(1)} px/m · follow ${follow ? 'ON' : 'off'}`;
  requestAnimationFrame(render);
}
render();

async function poll() {
  try {
    const r = await fetch('/pose', {cache: 'no-store'});
    state = await r.json();
  } catch (e) { /* 서버 잠깐 끊김 무시 */ }
}
setInterval(poll, 66);  // ~15 Hz
poll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 단독 실행: 원을 그리며 도는 가짜 차량으로 맵 동작 확인.
    import time

    srv = VizServer(port=8091)
    srv.start()
    print("브라우저에서 http://localhost:8091/ 를 열어보세요. Ctrl+C 종료.")
    t = 0.0
    try:
        while True:
            t += 0.1
            R = 20.0
            x = 300.0 + R * math.cos(t * 0.3)   # 임의 world 원점(300, 200)
            y = 200.0 + R * math.sin(t * 0.3)
            yaw = math.degrees(t * 0.3 + math.pi / 2)
            srv.update(x, y, yaw, speed_mps=R * 0.3)
            time.sleep(0.1)
    except KeyboardInterrupt:
        srv.stop()
