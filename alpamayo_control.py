#!/usr/bin/env python
"""Alpamayo -> CARLA 받는 쪽: UDP trajectory 수신 + world-frame 경로추종 제어.

Thor의 planner_live_service 가 `--enable-udp-bridge --udp-payload-mode text_json`
로 쏘는 UDP JSON 패킷을 받아, ego-local 경로(pred_xyz)를 받은 순간 차 pose 기준
world 좌표로 고정(→ tesla_camera.ego_path_to_world)한 뒤, 실제 후륜축 pose 로
closed-loop pure pursuit 추종한다.

UDP text_json 패킷 주요 필드 (planner_live/result_bridge.build_text_result_payload):
    pred_xyz        : [[x_m, y_m, 0.0], ...]  ego-local 미래 경로 (origin 제외)
    pred_v_mps      : [v, ...]                점별 목표 속도(m/s)
    pred_yaw_rad    : [yaw, ...]
    plan_dt_s       : 점 간 시간 간격(초, 보통 0.1)
    t0_us, inference_time_s : 타이밍

좌표계: alpamayo_bridge.py 가 보낸 것과 동일한 ego-local (x 전방, y 좌측+, z 위).

제어:
  lateral      : world-frame closed-loop pure pursuit (실제 후륜축 pose 기준)
  longitudinal : 모델 속도(pred_v_mps) 목표 P 제어 (없으면 고정속도 fallback)
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time

DEFAULT_WHEELBASE_M = 2.875  # Tesla Model 3
DEFAULT_MAX_STEER_DEG = 70.0

# 종방향 P 제어 게인
KP_THROTTLE = 0.5
KP_BRAKE = 0.5
SPEED_DEADBAND_MPS = 0.3

# 종방향: 모델 속도(pred_v_mps) 대신 고정 목표속도 사용 (횡방향만 경로 추종).
TARGET_SPEED_KMH = 20.0


class AlpamayoControlReceiver:
    """UDP text_json 패킷을 받아 최신 plan 을 보관."""

    def __init__(self, host="0.0.0.0", port=5005):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._plan: dict | None = None
        self._plan_recv_t: float = 0.0
        self._packets = 0
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[alpamayo] 제어 UDP 수신 시작: {self.host}:{self.port}")

    def _loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            plan = self._parse_plan(payload)
            if plan is not None:
                with self._lock:
                    self._plan = plan
                    self._plan_recv_t = time.monotonic()
                    self._packets += 1

    @staticmethod
    def _parse_plan(payload: dict) -> dict | None:
        pred_xyz = payload.get("pred_xyz")
        if not isinstance(pred_xyz, list) or not pred_xyz:
            return None
        pts = [(float(p[0]), float(p[1])) for p in pred_xyz if len(p) >= 2]
        if not pts:
            return None
        return {
            "points": pts,                                  # [(x,y), ...] ego-local
            "v_mps": payload.get("pred_v_mps") or [],
            "yaw": payload.get("pred_yaw_rad") or [],
            "curvature": payload.get("pred_curvature") or [],
            "plan_dt_s": float(payload.get("plan_dt_s") or 0.1),
            "t0_us": payload.get("t0_us"),
            "inference_time_s": float(payload.get("inference_time_s") or 0.0),
            "seq": payload.get("sample_id"),
        }

    def latest(self) -> tuple[dict | None, float]:
        """(plan, age_s). plan 없으면 (None, inf)."""
        with self._lock:
            if self._plan is None:
                return None, float("inf")
            return self._plan, time.monotonic() - self._plan_recv_t

    def stats(self) -> dict:
        plan, age = self.latest()
        return {"packets": self._packets, "has_plan": plan is not None,
                "age_s": None if plan is None else age}

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        print("[alpamayo] 제어 UDP 수신 종료")


class WorldPathFollower:
    """고정된 world 경로를 '실제 차 pose' 로 추종하는 순정 pure pursuit.

    후륜축 기준 world frame 에서 내 차 위치를 정확히 아는 상태로, 들어온 경로를
    위치 기반으로 따라간다. dead-reckoning open-loop 가 아니라, 매 프레임 차의
    실제 후륜축 (x,y,yaw) 로 경로에서 목표점을 찾으므로 cross-track/heading
    오차를 그대로 보정한다:

      1) 후륜축에서 경로상 '가장 가까운 점'(앞으로만 탐색)을 찾고
      2) 거기서부터 lookahead 거리 Ld=clamp(k*v+L0, min, max) 이상 앞의 점을 목표로
      3) 목표점을 차체좌표로 변환 -> α -> δ=atan2(2L·sinα, dist) -> steer

    부호규약: local_y = +y_world 성분(=CARLA 우측),
    α>0 -> δ>0 -> steer>0 = CARLA 우회전.
    종방향: 모델 속도(pred_v_mps) 목표 P 제어. set_path 에 velocities 를 주면
    현재 위치의 점별 목표속도를 추종하고, 없으면 고정 목표속도(TARGET_SPEED_KMH).
    use_model_speed=False 로 강제 고정속도 전환 가능(키 V).
    """

    def __init__(self, wheelbase_m=DEFAULT_WHEELBASE_M,
                 max_steer_deg=DEFAULT_MAX_STEER_DEG,
                 ld_gain=0.6, ld_l0=3.0, ld_min=4.0, ld_max=10.0, reach_r=2.0):
        self.wheelbase = float(wheelbase_m)
        self.max_steer_rad = math.radians(float(max_steer_deg))
        self.ld_gain = ld_gain
        self.ld_l0 = ld_l0
        self.ld_min = ld_min
        self.ld_max = ld_max
        self.reach_r = reach_r
        self._path: list[tuple[float, float]] = []   # world 좌표
        self._vel: list[float] | None = None         # 점별 목표속도(m/s), 경로와 1:1
        self._idx = 0                                # 앞으로만 진행하는 최근접 인덱스
        self.use_model_speed = True  # True 면 모델 pred_v_mps 를 종방향 목표로 사용
        # HUD/로깅
        self.last_i_goal = 0
        self.last_cte = 0.0        # 경로까지 최단거리(cross-track 크기)
        self.last_ld = 0.0
        self.last_v_target = 0.0   # 최근 종방향 목표속도(m/s)
        self.last_v_source = "fixed"  # "model" | "fixed"
        self.finished = False

    @property
    def has_path(self) -> bool:
        return len(self._path) >= 2

    @property
    def n_points(self) -> int:
        return len(self._path)

    def set_path(self, world_points, velocities=None) -> None:
        """추종할 경로를 CARLA world 좌표 [(x,y)..] 로 설정(새 경로마다 호출).

        velocities 를 주면(경로 점과 1:1 정렬된 목표속도 m/s) 종방향 제어에
        모델 속도를 쓴다. 없으면(map route 등) 고정 목표속도로 fallback.
        """
        vel = list(velocities) if velocities is not None else None
        path, vout = [], []
        for i, p in enumerate(world_points):
            if len(p) < 2:
                continue
            path.append((float(p[0]), float(p[1])))
            if vel is not None and i < len(vel):
                vout.append(float(vel[i]))
        self._path = path
        self._vel = vout if (vel is not None and len(vout) == len(path) and path) else None
        self._idx = 0
        self.finished = False

    def _closest_ahead(self, x: float, y: float) -> tuple[int, float]:
        """앞으로만(단조 증가) 최근접 인덱스 갱신 후 (idx, 거리) 반환."""
        best_i, best_d = self._idx, float("inf")
        for i in range(self._idx, len(self._path)):
            px, py = self._path[i]
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d, best_i = d, i
        self._idx = best_i
        return best_i, best_d

    def compute(self, rear_x: float, rear_y: float, yaw_deg: float,
                speed_mps: float) -> tuple[float, float, float]:
        if not self.has_path:
            return 0.0, 0.0, 0.0

        ci, cdist = self._closest_ahead(rear_x, rear_y)
        self.last_cte = cdist

        # 끝 도달 판정(한 번 True 면 유지): 마지막 점 근접 또는 인덱스 소진
        ex, ey = self._path[-1]
        if (math.hypot(ex - rear_x, ey - rear_y) < self.reach_r
                or ci >= len(self._path) - 1):
            self.finished = True

        # lookahead 거리 Ld = clamp(k*v + L0, min, max)
        ld = min(self.ld_max, max(self.ld_min, self.ld_gain * speed_mps + self.ld_l0))
        self.last_ld = ld

        # 최근접점부터 Ld 이상 떨어진 첫 점을 목표로(앞으로만). 없으면 마지막 점.
        gi = len(self._path) - 1
        for i in range(ci, len(self._path)):
            px, py = self._path[i]
            if math.hypot(px - rear_x, py - rear_y) >= ld:
                gi = i
                break
        self.last_i_goal = gi
        tx, ty = self._path[gi]

        # 목표점을 차체좌표로 변환 (local_y=+y_world=CARLA 우측)
        yaw = math.radians(yaw_deg)
        dx, dy = tx - rear_x, ty - rear_y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy    # 전방(+)
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy   # +y_world(=CARLA 우측)
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            steer_cmd = 0.0
        else:
            alpha = math.atan2(local_y, local_x)
            delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), dist)
            steer_cmd = max(-1.0, min(1.0, delta / self.max_steer_rad))

        # 종방향: 모델 속도(pred_v_mps)를 목표로 P 제어. 현재 위치(최근접점 ci)의
        # 점별 목표속도를 사용한다. 속도가 없으면(map route 등) 고정 목표속도.
        if self.use_model_speed and self._vel is not None:
            v_target = self._vel[min(ci, len(self._vel) - 1)]
            self.last_v_source = "model"
        else:
            v_target = TARGET_SPEED_KMH / 3.6
            self.last_v_source = "fixed"
        self.last_v_target = v_target
        err = v_target - speed_mps
        throttle = brake = 0.0
        if err > SPEED_DEADBAND_MPS:
            throttle = min(1.0, KP_THROTTLE * err)
        elif err < -SPEED_DEADBAND_MPS:
            brake = min(1.0, KP_BRAKE * (-err))
        return round(steer_cmd, 4), round(throttle, 4), round(brake, 4)
