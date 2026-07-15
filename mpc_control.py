#!/usr/bin/env python
"""횡방향 LTV-MPC 제어기 (순수 numpy).

WorldPathFollower(pure pursuit)의 대체 횡방향 제어기. 들어온 world 경로를
kinematic bicycle **오차 동역학**으로 예측하고, cross-track / heading 오차를
최소화하는 조향 시퀀스를 매 프레임 QP 로 풀어 첫 스텝만 적용한다(receding horizon).
속도 v(t) 와 경로 곡률 κ(t) 는 경로에서 뽑아 시변 파라미터로 넣으므로 LTV.

좌표/부호 규약 (alpamayo_control.WorldPathFollower 와 동일, CARLA 좌수계):
    상태 x = [e1, e2]
      e1 : 경로 대비 차의 '우측(+)' 횡오차 (e1>0 = 차가 경로 오른쪽으로 이탈)
      e2 : wrap(yaw - ψ_path) 헤딩오차
    입력  s : CARLA steer ∈ [-1, 1] (s>0 = 우회전), 물리 조향각 δ = s·max_steer
    이산 모델 (Euler, dt):
      e1' = e1 + dt·v·e2
      e2' = e2 + dt·(v·max_steer/L)·s - dt·v·κ
    → e1>0(우측 이탈)이면 MPC 는 s<0(좌조향)을 내므로 CARLA 부호와 자동 일치.

종방향은 WorldPathFollower 와 동일하게 모델 속도(pred_v_mps) 목표 P 제어
(alpamayo_control.longitudinal_throttle_brake 공유). 즉 이 클래스는 '횡방향만'
교체한다. tesla_camera 에서 키 M 으로 pure pursuit ↔ MPC 토글.
"""
from __future__ import annotations

import math

import numpy as np

from alpamayo_control import (
    DEFAULT_MAX_STEER_DEG,
    DEFAULT_WHEELBASE_M,
    TARGET_SPEED_KMH,
    longitudinal_throttle_brake,
)


def _wrap_pi(a: float) -> float:
    """각도를 (-π, π] 로 wrap."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def solve_box_qp(H, g, lb, ub, *, iters=120, tol=1e-6, warm=None):
    """min 0.5 uᵀH u + gᵀu  s.t. lb ≤ u ≤ ub  를 FISTA(가속 투영경사)로 근사.

    H: (n,n) 대칭 PD, g:(n,), lb/ub: (n,) 또는 스칼라. warm: 초기값(n,).
    box 제약이라 투영 = clip. 반환 u:(n,).
    """
    n = g.shape[0]
    lb = np.full(n, lb) if np.isscalar(lb) else np.asarray(lb, float)
    ub = np.full(n, ub) if np.isscalar(ub) else np.asarray(ub, float)
    u = np.clip(warm.copy(), lb, ub) if warm is not None else np.zeros(n)
    y = u.copy()
    # Lipschitz 상수 = H 최대 고유값 (H 작아서 eigvalsh 저렴)
    lmax = float(np.linalg.eigvalsh(H)[-1])
    step = 1.0 / max(lmax, 1e-9)
    t = 1.0
    for _ in range(iters):
        u_new = np.clip(y - step * (H @ y + g), lb, ub)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = u_new + ((t - 1.0) / t_new) * (u_new - u)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u, t = u_new, t_new
    return u


class LateralMPC:
    """world 경로를 오차 동역학 LTV-MPC 로 추종하는 횡방향 제어기.

    WorldPathFollower 와 동일한 인터페이스(set_path / compute / has_path /
    n_points / finished / use_model_speed + HUD 속성)를 제공해 tesla_camera 에서
    드롭인 교체가 가능하다.
    """

    def __init__(self, wheelbase_m=DEFAULT_WHEELBASE_M,
                 max_steer_deg=DEFAULT_MAX_STEER_DEG,
                 dt=0.1, horizon=25,
                 q_ey=1.0, q_epsi=0.15, r_steer=0.3, r_dsteer=12.0,
                 qf_scale=3.0, v_floor=1.0, reach_r=2.0,
                 max_dsteer=0.04, kappa_smooth=5):
        self.wheelbase = float(wheelbase_m)
        self.max_steer_rad = math.radians(float(max_steer_deg))
        self.dt = float(dt)
        self.N = int(horizon)
        self.q_ey = float(q_ey)
        self.q_epsi = float(q_epsi)
        self.r_steer = float(r_steer)
        self.r_dsteer = float(r_dsteer)
        self.qf_scale = float(qf_scale)     # 종단 가중 = qf_scale · 스테이지 가중
        self.v_floor = float(v_floor)       # 모델 v 하한(정지 시 제어 가능성 유지)
        self.reach_r = float(reach_r)
        # 진동 억제: 프레임당 조향 변화 하드 제한(액추에이터 지연/고루프율 보상) +
        #            곡률 이동평균 스무딩(feedforward 지터 감소)
        self.max_dsteer = float(max_dsteer)
        self.kappa_smooth = int(kappa_smooth)
        self._last_steer = 0.0              # 직전 적용 조향(slew 기준 + rate 페널티 앵커)
        self.use_model_speed = True

        # 경로 상태 (world)
        self._path: list[tuple[float, float]] = []
        self._vel: list[float] | None = None   # 점별 목표속도(m/s)
        self._psi: np.ndarray | None = None     # 점별 경로 heading(rad)
        self._kappa: np.ndarray | None = None    # 점별 곡률(우측+, 1/m)
        self._s: np.ndarray | None = None        # 누적 호길이(m)
        self._idx = 0
        self._warm: np.ndarray | None = None     # QP warm-start
        self.finished = False

        # HUD/로깅 (WorldPathFollower 와 호환되는 이름)
        self.last_i_goal = 0
        self.last_cte = 0.0
        self.last_ld = 0.0
        self.last_lookahead_mode = "mpc"
        self.last_v_target = 0.0
        self.last_v_source = "fixed"
        # MPC 전용
        self.last_e1 = 0.0
        self.last_e2 = 0.0

    @property
    def has_path(self) -> bool:
        return len(self._path) >= 2

    @property
    def n_points(self) -> int:
        return len(self._path)

    def set_path(self, world_points, velocities=None) -> None:
        """world 경로 [(x,y)..] + (선택) 점별 목표속도 설정. heading/곡률/호길이 선계산."""
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
        self._warm = None
        self.finished = False
        self._precompute_geometry()

    def _precompute_geometry(self) -> None:
        n = len(self._path)
        if n < 2:
            self._psi = self._kappa = self._s = None
            return
        xy = np.asarray(self._path, dtype=float)      # (n,2)
        d = np.diff(xy, axis=0)                        # (n-1,2)
        seg = np.hypot(d[:, 0], d[:, 1])               # (n-1,)
        s = np.concatenate([[0.0], np.cumsum(seg)])    # (n,)
        psi = np.arctan2(d[:, 1], d[:, 0])             # (n-1,) 세그먼트 heading
        psi = np.concatenate([psi, psi[-1:]])          # (n,) 마지막은 복제
        # 곡률 κ = dψ/ds (우측+; ψ 증가=우회전). 세그먼트 heading 차 / 평균 세그먼트 길이
        dpsi = np.array([_wrap_pi(psi[i + 1] - psi[i]) for i in range(n - 1)])
        ds = np.maximum(seg, 1e-3)
        kappa = np.concatenate([dpsi / ds, [0.0]])     # (n,)
        # 곡률 이동평균 스무딩(feedforward 지터 → 진동 억제). 홀수 window.
        w = self.kappa_smooth
        if w >= 3 and n >= w:
            w = w if w % 2 == 1 else w + 1
            ker = np.ones(w) / w
            kappa = np.convolve(kappa, ker, mode="same")
        self._s, self._psi, self._kappa = s, psi, kappa

    def _closest_ahead(self, x: float, y: float) -> tuple[int, float]:
        best_i, best_d = self._idx, float("inf")
        for i in range(self._idx, len(self._path)):
            px, py = self._path[i]
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d, best_i = d, i
        self._idx = best_i
        return best_i, best_d

    def _sample_horizon(self, ci: int):
        """현재 호위치에서 dt 마다 v·dt 만큼 전진하며 v[k], κ[k] 샘플 (LTV 파라미터)."""
        s_arr, kappa = self._s, self._kappa
        if self._vel is not None:
            vprof = np.asarray(self._vel, dtype=float)
        else:
            vprof = np.full(len(self._path), TARGET_SPEED_KMH / 3.6)
        vk = np.empty(self.N)
        kk = np.empty(self.N)
        sp = float(s_arr[ci])
        for k in range(self.N):
            v = max(self.v_floor, float(np.interp(sp, s_arr, vprof)))
            vk[k] = v
            kk[k] = float(np.interp(sp, s_arr, kappa))
            sp += v * self.dt
        return vk, kk

    def _build_qp(self, x0, vk, kk, prev_s):
        """LTV 오차모델 condensed QP (H, g) 생성. 상태 dim=2, 입력 dim=1, N 스텝."""
        N, dt, L, ms = self.N, self.dt, self.wheelbase, self.max_steer_rad
        # A[k], B[k], d[k]
        A = [np.array([[1.0, dt * vk[k]], [0.0, 1.0]]) for k in range(N)]
        B = [np.array([0.0, dt * vk[k] * ms / L]) for k in range(N)]
        dvec = [np.array([0.0, -dt * vk[k] * kk[k]]) for k in range(N)]

        # 자유(무입력) 궤적 Xfree = [x1..xN], (2N,)
        Xfree = np.empty(2 * N)
        xp = x0.copy()
        for i in range(N):
            xp = A[i] @ xp + dvec[i]
            Xfree[2 * i:2 * i + 2] = xp
        # Gamma (2N x N): u[j] → x[1..N]
        Gamma = np.zeros((2 * N, N))
        for j in range(N):
            xp = np.zeros(2)
            for i in range(N):
                xp = A[i] @ xp + (B[i] if i == j else 0.0)
                if i >= j:
                    Gamma[2 * i:2 * i + 2, j] = xp
        # 가중 (스테이지 x1..x_{N-1}, 종단 xN)
        qdiag = np.tile([self.q_ey, self.q_epsi], N)
        qdiag[-2:] = [self.q_ey * self.qf_scale, self.q_epsi * self.qf_scale]
        # rate: (u_k - u_{k-1}), u_{-1}=prev_s
        Lmat = np.eye(N) - np.eye(N, k=-1)
        b = np.zeros(N)
        b[0] = prev_s
        GtQ = Gamma.T * qdiag                      # (N x 2N)
        H = 2.0 * (GtQ @ Gamma + self.r_steer * np.eye(N)
                   + self.r_dsteer * (Lmat.T @ Lmat))
        g = 2.0 * (GtQ @ Xfree - self.r_dsteer * (Lmat.T @ b))
        return H, g

    def compute(self, rear_x: float, rear_y: float, yaw_deg: float,
                speed_mps: float) -> tuple[float, float, float]:
        if not self.has_path:
            return 0.0, 0.0, 0.0

        ci, cdist = self._closest_ahead(rear_x, rear_y)
        self.last_i_goal = min(ci + self.N, len(self._path) - 1)

        # 끝 도달 판정 (WorldPathFollower 와 동일)
        ex, ey = self._path[-1]
        if (math.hypot(ex - rear_x, ey - rear_y) < self.reach_r
                or ci >= len(self._path) - 1):
            self.finished = True

        # 현재 오차 [e1(우측+ 횡오차), e2(헤딩오차)]
        psi_p = float(self._psi[ci])
        px, py = self._path[ci]
        # e1 = (차-경로점) 벡터를 경로 우측 법선(-sinψ, cosψ)에 투영
        e1 = -(rear_x - px) * math.sin(psi_p) + (rear_y - py) * math.cos(psi_p)
        e2 = _wrap_pi(math.radians(yaw_deg) - psi_p)
        self.last_e1, self.last_e2, self.last_cte = e1, e2, abs(e1)

        # 종방향 목표속도 (WorldPathFollower 와 동일)
        if self.use_model_speed and self._vel is not None:
            v_target = self._vel[min(ci, len(self._vel) - 1)]
            self.last_v_source = "model"
        else:
            v_target = TARGET_SPEED_KMH / 3.6
            self.last_v_source = "fixed"
        self.last_v_target = v_target

        # 경로 소진 시 직진 유지
        if ci >= len(self._path) - 1:
            self.last_ld = 0.0
            self._last_steer = 0.0
            throttle, brake = longitudinal_throttle_brake(v_target, speed_mps)
            return 0.0, round(throttle, 4), round(brake, 4)

        # LTV-MPC QP 풀이 → 첫 스텝 조향. rate 페널티 앵커 = 실제 직전 적용 조향.
        vk, kk = self._sample_horizon(ci)
        H, g = self._build_qp(np.array([e1, e2]), vk, kk, self._last_steer)
        warm = None
        if self._warm is not None:                  # 직전 해를 한 칸 shift 해 warm-start
            warm = np.concatenate([self._warm[1:], self._warm[-1:]])
        u = solve_box_qp(H, g, -1.0, 1.0, warm=warm)
        self._warm = u
        raw = float(np.clip(u[0], -1.0, 1.0))
        # 프레임당 조향 slew-rate 하드 제한(진동/급변 억제, 루프율 무관하게 작동)
        lo, hi = self._last_steer - self.max_dsteer, self._last_steer + self.max_dsteer
        steer_cmd = min(hi, max(lo, raw))
        self._last_steer = steer_cmd
        self.last_ld = float(np.sum(vk) * self.dt)   # 예측 호라이즌 거리(참고용)

        throttle, brake = longitudinal_throttle_brake(v_target, speed_mps)
        return round(steer_cmd, 4), round(throttle, 4), round(brake, 4)
