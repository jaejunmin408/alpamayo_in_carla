#!/usr/bin/env python
"""Alpamayo -> CARLA 받는 쪽: UDP trajectory 수신 + 경로추종 제어.

Thor의 planner_live_service 가 `--enable-udp-bridge --udp-payload-mode text_json`
로 쏘는 UDP JSON 패킷을 받아, ego-local 경로(pred_xyz)와 목표속도(pred_v_mps)를
CARLA 차량 제어(steer/throttle/brake)로 변환한다.

UDP text_json 패킷 주요 필드 (planner_live/result_bridge.build_text_result_payload):
    pred_xyz        : [[x_m, y_m, 0.0], ...]  ego-local 미래 경로 (origin 제외)
    pred_v_mps      : [v, ...]                점별 목표 속도(m/s)
    pred_curvature  : [k, ...]                점별 곡률 (이미 제어 부호로 반전됨)
    pred_yaw_rad    : [yaw, ...]
    plan_dt_s       : 점 간 시간 간격(초, 보통 0.1)
    t0_us, inference_time_s : 타이밍(지연 보정용)

좌표계: alpamayo_bridge.py 가 보낸 것과 동일한 ego-local (x 전방, y 좌측+, z 위).
CARLA steer 는 +1 이 우회전이므로 좌(+y)로 가려면 음수.

제어:
  lateral      : pure pursuit (lookahead 점으로 조향각 산출)
  longitudinal : 목표속도 P 제어 (throttle/brake)
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time

# pred_xyz 의 y 가 좌측(+)인지. alpamayo_bridge.Y_SIGN 과 일관되게 둔다.
# 실주행에서 조향이 반대로 먹으면 STEER_SIGN 을 뒤집어 본다.
STEER_SIGN = -1.0          # 좌(+y) -> CARLA steer 음수
DEFAULT_WHEELBASE_M = 2.875  # Tesla Model 3
DEFAULT_MAX_STEER_DEG = 70.0

# pure pursuit lookahead 방식 선택
#   True  : i_now 이후 남은 경로 인덱스의 일정 퍼센트 지점을 목표점으로 사용
#   False : arc length 기반 lookahead 거리(ld)로 목표점 선택
USE_INDEX_LOOKAHEAD = True
# 전체 경로(0..n-1) 중 이 비율 지점의 인덱스를 목표로. 0~1.
LOOKAHEAD_INDEX_PCT = 0.5
# pure pursuit 특성상 목표점은 현재(i_now)보다 앞서야 하므로 최소 앞선 점 수.
LOOKAHEAD_INDEX_MIN_STEP = 2

# arc length 기반 lookahead = clamp(k_v * speed + L0, MIN, MAX)
LOOKAHEAD_K = 0.6
LOOKAHEAD_L0 = 4.0
LOOKAHEAD_MIN = 4.0
LOOKAHEAD_MAX = 20.0

# 종방향 P 제어 게인
KP_THROTTLE = 0.5
KP_BRAKE = 0.5
SPEED_DEADBAND_MPS = 0.3

# 종방향: 모델 속도(pred_v_mps) 대신 고정 목표속도 사용.
#   - 횡방향(조향)만 Alpamayo 경로로 제어
#   - 항상 움직이므로 ego-history 정지 트랩 회피
USE_FIXED_TARGET_SPEED = True
TARGET_SPEED_KMH = 10.0

# plan 은 t0 기준 6.4초(64점) full plan 이므로, 추론 공백 동안에도 직전 plan 을
# "경과시간만큼 진행시켜" 계속 따라간다. STALE 은 그래도 너무 오래된 plan 차단용.
STALE_PLAN_S = 3.0          # 받은 지 이보다 오래되면 정지(coast+brake)


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


class PathFollower:
    """ego-local 경로 + 목표속도 -> (steer, throttle, brake)."""

    def __init__(self, wheelbase_m=DEFAULT_WHEELBASE_M,
                 max_steer_deg=DEFAULT_MAX_STEER_DEG):
        self.wheelbase = float(wheelbase_m)
        self.max_steer_rad = math.radians(float(max_steer_deg))

    def _now_index(self, plan: dict, age_s: float) -> int:
        """plan(t0 기준)에서 '지금' 차가 위치한 점 인덱스.

        t0 이후 경과시간 ~= 추론시간 + 받은 뒤 경과(age). 그만큼 plan 을
        진행시켜, 이미 지나온 앞부분을 건너뛰고 현재 지점부터 추종한다.
        (latency 보상 + 추론 공백 동안 직전 plan 을 계속 따라가는 효과)
        """
        n = len(plan["points"])
        dt = max(plan["plan_dt_s"], 1e-3)
        elapsed = max(0.0, age_s + float(plan.get("inference_time_s") or 0.0))
        return max(0, min(n - 1, int(elapsed / dt)))

    def _heading_at(self, points, yaws, i):
        """점 i 에서의 진행 방향(rad). yaw 있으면 사용, 없으면 다음 점으로 추정."""
        if i < len(yaws):
            return float(yaws[i])
        if i + 1 < len(points):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            return math.atan2(dy, dx)
        return 0.0

    def compute(self, plan: dict, speed_mps: float,
                age_s: float = 0.0) -> tuple[float, float, float]:
        points = plan["points"]
        yaws = plan.get("yaw") or []
        n = len(points)

        i_now = self._now_index(plan, age_s)
        cx, cy = points[i_now]                       # 현재 차 위치(plan t0 프레임)
        cyaw = self._heading_at(points, yaws, i_now)  # 현재 진행 방향
        cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)

        if USE_INDEX_LOOKAHEAD:
            # 전체 경로(0..n-1) 인덱스의 LOOKAHEAD_INDEX_PCT 지점을 목표로.
            # 단, pure pursuit 은 목표가 현재보다 앞서야 하므로 i_now 기준 하한을 둔다.
            ti = int(round(LOOKAHEAD_INDEX_PCT * (n - 1)))
            ti = max(ti, min(n - 1, i_now + LOOKAHEAD_INDEX_MIN_STEP))
        else:
            # i_now 부터 호길이(arc length)로 lookahead 거리 이상인 점을 목표로
            ld = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN,
                                        LOOKAHEAD_K * speed_mps + LOOKAHEAD_L0))
            acc = 0.0
            ti = n - 1
            for i in range(i_now, n - 1):
                acc += math.hypot(points[i + 1][0] - points[i][0],
                                  points[i + 1][1] - points[i][1])
                if acc >= ld:
                    ti = i + 1
                    break
        tx, ty = points[ti]

        # 목표점을 '현재 차 프레임'으로 변환 (현재 위치 기준 평행이동 + -cyaw 회전)
        dx, dy = tx - cx, ty - cy
        fx = cos_y * dx + sin_y * dy      # 전방 성분
        fy = -sin_y * dx + cos_y * dy     # 좌측(+) 성분
        dist = math.hypot(fx, fy)

        # pure pursuit: kappa = 2*fy / dist^2  (fy>0 좌측 -> 좌회전)
        if dist < 1e-3:
            steer_cmd = 0.0
        else:
            kappa = 2.0 * fy / (dist * dist)
            delta = math.atan(self.wheelbase * kappa)
            steer_cmd = STEER_SIGN * delta / self.max_steer_rad
            steer_cmd = max(-1.0, min(1.0, steer_cmd))

        # 종방향: 고정 목표속도(기본 10km/h) P 제어. (모델 속도는 무시)
        if USE_FIXED_TARGET_SPEED:
            v_target = TARGET_SPEED_KMH / 3.6
        else:
            v_list = plan.get("v_mps") or []
            v_target = float(v_list[min(i_now, len(v_list) - 1)]) if v_list else speed_mps
        err = v_target - speed_mps
        throttle = brake = 0.0
        if err > SPEED_DEADBAND_MPS:
            throttle = min(1.0, KP_THROTTLE * err)
        elif err < -SPEED_DEADBAND_MPS:
            brake = min(1.0, KP_BRAKE * (-err))
        return round(steer_cmd, 4), round(throttle, 4), round(brake, 4)
