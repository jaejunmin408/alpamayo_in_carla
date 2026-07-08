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
  lateral      : 거리 기반 slice + re-zero 후 pure pursuit (openpilot udp_bridge 방식)
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

# 경로 갱신(slice) 방식: openpilot selfdrive/modeld/udp_bridge.py 와 동일하게
# "거리(arc-length) 기반" 으로 통일한다. 매 프레임 이동거리(v_ego·dt)를 누적해
# path 를 그만큼 잘라 새 원점·전방축으로 re-zero 한 뒤 pure pursuit 을 적용한다.
# (기존 시간-인덱스(elapsed/dt) 진행 방식은 폐기)

# pure pursuit lookahead 방식 선택 (re-zero 된 '남은 path' 기준)
#   True  : 남은 경로 인덱스의 일정 퍼센트(LOOKAHEAD_INDEX_PCT) 지점을 목표점으로
#   False : arc length 기반 lookahead 거리(ld)로 목표점 선택
USE_INDEX_LOOKAHEAD = True
# 남은 경로(0..ns-1) 중 이 비율 지점의 인덱스를 목표로. 0~1. (udp_bridge PP_INDEX_FRAC)
LOOKAHEAD_INDEX_PCT = 0.5

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


def _interp1(xq, xs, ys):
    """xs(오름차순)에 대한 ys 의 선형보간. 범위 밖은 양끝값으로 clamp."""
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xq <= xs[i]:
            d = xs[i] - xs[i - 1]
            t = (xq - xs[i - 1]) / d if d > 1e-12 else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def slice_and_rezero(points, s_offset):
    """ego-local path(points=[(x,y)..], x 전방/y 좌측)를 arc-length s_offset 지점에서
    잘라, 그 지점을 새 원점(0,0)·전방축(+x)으로 re-zero 한 (xs, ys) 를 반환.

    openpilot selfdrive/modeld/udp_bridge.slice_and_rezero 와 동일한 방식.
      - s_offset 만큼 진행한 점을 보간으로 구해 새 원점으로
      - 그 점의 진행방향(yaw0)을 +x 축에 맞추도록 전체 -yaw0 회전
      - s_offset 이후 점들만 유지 (원점 점을 맨 앞에 prepend)
    s_offset 이 path 길이를 넘으면 마지막 점으로 clamp(→ 사실상 직진).
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    s = [0.0]
    for i in range(1, len(points)):
        s.append(s[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    s_total = s[-1]
    s_off = min(max(s_offset, 0.0), s_total)

    x0 = _interp1(s_off, s, xs)
    y0 = _interp1(s_off, s, ys)
    s_ahead = min(s_off + 1e-3, s_total)
    x1 = _interp1(s_ahead, s, xs)
    y1 = _interp1(s_ahead, s, ys)
    yaw0 = math.atan2(y1 - y0, x1 - x0)
    c, sn = math.cos(-yaw0), math.sin(-yaw0)

    out_x, out_y = [0.0], [0.0]                 # 새 원점
    for i in range(len(points)):
        if s[i] > s_off:
            dx, dy = xs[i] - x0, ys[i] - y0
            out_x.append(c * dx - sn * dy)
            out_y.append(sn * dx + c * dy)
    return out_x, out_y


class PathFollower:
    """ego-local 경로 + 목표속도 -> (steer, throttle, brake)."""

    def __init__(self, wheelbase_m=DEFAULT_WHEELBASE_M,
                 max_steer_deg=DEFAULT_MAX_STEER_DEG):
        self.wheelbase = float(wheelbase_m)
        self.max_steer_rad = math.radians(float(max_steer_deg))
        # 거리 기반 slice 상태
        self._last_plan = None       # plan 객체 동일성으로 새 plan 감지
        self._slice_s = 0.0          # 현재 path 상 누적 전진 offset (m)
        self._last_mono = None       # dt 계산용 직전 호출 시각
        # 로깅/HUD 용 최근 값
        self.last_slice_s = 0.0
        self.last_i_goal = 0

    def _advance_slice(self, plan: dict, speed_mps: float, age_s: float) -> float:
        """거리 기반 slice offset 갱신 (openpilot udp_bridge 방식).

        - 새 plan 이면 offset 을 '추론+수신 지연 동안 이동한 거리' 로 seed.
          (plan 원점은 t0 시점 위치이므로 그만큼 이미 지나왔다)
        - 이후 매 호출마다 v_ego·dt 를 누적해 path 위를 전진.
        """
        now = time.monotonic()
        if plan is not self._last_plan:
            self._last_plan = plan
            lat = max(0.0, age_s + float(plan.get("inference_time_s") or 0.0))
            self._slice_s = speed_mps * lat
            self._last_mono = now
        else:
            dt = (now - self._last_mono) if self._last_mono is not None else 0.0
            self._last_mono = now
            self._slice_s += speed_mps * max(0.0, dt)
        return self._slice_s

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

    def compute(self, plan: dict, speed_mps: float,
                age_s: float = 0.0) -> tuple[float, float, float]:
        points = plan["points"]                       # [(x,y)..] ego-local @ t0

        # 1. 거리 기반 slice offset 갱신 후 path 를 re-zero (→ 현재 차 프레임)
        s_off = self._advance_slice(plan, speed_mps, age_s)
        sx, sy = slice_and_rezero(points, s_off)
        ns = len(sx)
        self.last_slice_s = s_off

        # 2. 목표점 선택 (re-zero 된 '남은 path' 기준)
        if ns < 2:
            # 남은 경로 없음(plan 소진) → 직진
            self.last_i_goal = 0
            steer_cmd = 0.0
        else:
            if USE_INDEX_LOOKAHEAD:
                # 남은 경로(0..ns-1) 인덱스의 LOOKAHEAD_INDEX_PCT 지점을 목표로.
                gi = int(round(LOOKAHEAD_INDEX_PCT * (ns - 1)))
                gi = max(1, min(ns - 1, gi))
            else:
                # 원점부터 호길이(arc length)로 lookahead 거리 이상인 점을 목표로
                ld = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN,
                                            LOOKAHEAD_K * speed_mps + LOOKAHEAD_L0))
                acc = 0.0
                gi = ns - 1
                for i in range(ns - 1):
                    acc += math.hypot(sx[i + 1] - sx[i], sy[i + 1] - sy[i])
                    if acc >= ld:
                        gi = i + 1
                        break
            self.last_i_goal = gi

            # re-zero 로 이미 현재 차 프레임: fx 전방(+), fy 좌측(+)
            fx, fy = sx[gi], sy[gi]
            dist = math.hypot(fx, fy)

            # pure pursuit: kappa = 2*fy / dist^2  (fy>0 좌측 -> 좌회전)
            if dist < 1e-3:
                steer_cmd = 0.0
            else:
                kappa = 2.0 * fy / (dist * dist)
                delta = math.atan(self.wheelbase * kappa)
                steer_cmd = STEER_SIGN * delta / self.max_steer_rad
                steer_cmd = max(-1.0, min(1.0, steer_cmd))

        # 3. 종방향: 고정 목표속도(기본 10km/h) P 제어. (모델 속도는 무시)
        if USE_FIXED_TARGET_SPEED:
            v_target = TARGET_SPEED_KMH / 3.6
        else:
            v_list = plan.get("v_mps") or []
            i_now = self._now_index(plan, age_s)
            v_target = float(v_list[min(i_now, len(v_list) - 1)]) if v_list else speed_mps
        err = v_target - speed_mps
        throttle = brake = 0.0
        if err > SPEED_DEADBAND_MPS:
            throttle = min(1.0, KP_THROTTLE * err)
        elif err < -SPEED_DEADBAND_MPS:
            brake = min(1.0, KP_BRAKE * (-err))
        return round(steer_cmd, 4), round(throttle, 4), round(brake, 4)
