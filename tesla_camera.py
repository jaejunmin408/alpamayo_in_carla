#!/usr/bin/env python
"""테슬라 차량에 카메라 여러 대 + IMU/GNSS 센서를 달아, 카메라는 격자로
보고 좌우 카메라 각도를 별도 컨트롤 창(슬라이더)에서 실시간 조절하는 스크립트.

IMU(가속도/각속도/방위)와 GNSS(위/경/고도)는 차량에 부착되어 각 10Hz로
값을 받아 카메라 창 좌하단 HUD에 표시한다. (나중에 자율주행 모델 입력으로 사용)

서버(start_server.sh)가 떠 있는 상태에서 venv 파이썬으로 실행:
    ./venv/bin/python tesla_camera.py

옵션:
    --cameras front,tele,left,right    띄울 카메라 목록 (쉼표 구분)
                                       기본: front,tele,left,right
    --side-yaw N / --side-pitch N      좌우 카메라 시작 좌우각/상하각(도)
    --fov / --tele-fov                 일반 화각(기본 90) / 망원 화각(기본 30)
    --autopilot / --no-autopilot       자동 주행 (기본 켜짐)
    --width / --height                 카메라 창 크기 (기본 1280x720)

카메라 종류:
    front  전방        tele  전방 망원(좁은 화각으로 멀리)
    left   좌측        right 우측
    rear   후방        chase 3인칭 추적뷰   top 탑뷰

실시간 조작 (카메라 격자 창에 포커스를 두고):
    [운전]  W 전진  S 후진  A/D 좌/우 조향  SPACE 핸드브레이크
            P  자동주행 <-> 수동운전 전환
    [카메라] ← / →  좌우(yaw)    ↑ / ↓  상하(pitch)
            컨트롤 창의 슬라이더 클릭/드래그로 직접 지정
            R  각도 리셋(yaw 90°, pitch 0°)
    ESC / Q  종료
"""
import argparse
import math
import sys
import time
import weakref
from collections import deque

import carla
import numpy as np
import pygame
from pygame._sdl2.video import Window, Renderer, Texture

from alpamayo_bridge import AlpamayoBridge, REQUIRED_CAMERAS, IMAGE_W, IMAGE_H
from alpamayo_control import AlpamayoControlReceiver, WorldPathFollower
from mpc_control import LateralMPC
from viz_server import VizServer


SIDE_LOCATIONS = {
    "left":  carla.Location(y=-0.9, z=1.4),
    "right": carla.Location(y=0.9, z=1.4),
}

# 각도 조절과 무관한 고정 카메라들의 위치/방향. (tele는 front와 동일 방향)
FIXED_TRANSFORMS = {
    "front": carla.Transform(carla.Location(x=1.6, z=1.4)),
    "tele":  carla.Transform(carla.Location(x=1.6, z=1.4)),
    "rear":  carla.Transform(carla.Location(x=-1.8, z=1.4),
                             carla.Rotation(yaw=180.0)),
    "chase": carla.Transform(carla.Location(x=-5.5, z=2.8),
                             carla.Rotation(pitch=-12.0)),
    "top":   carla.Transform(carla.Location(z=12.0),
                             carla.Rotation(pitch=-90.0)),
}
CAMERA_NAMES = tuple(FIXED_TRANSFORMS) + tuple(SIDE_LOCATIONS)


def side_transform(name, yaw, pitch):
    sign = -1.0 if name == "left" else 1.0
    return carla.Transform(SIDE_LOCATIONS[name],
                           carla.Rotation(yaw=sign * yaw, pitch=pitch))


def parse_args():
    p = argparse.ArgumentParser(description="CARLA Tesla 멀티 카메라 뷰어")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", default="90")
    p.add_argument("--tele-fov", default="30")
    p.add_argument("--side-yaw", type=float, default=90.0)
    p.add_argument("--side-pitch", type=float, default=0.0)
    p.add_argument("--cameras", default="front,tele,left,right",
                   help="쉼표로 구분: " + ", ".join(CAMERA_NAMES))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--autopilot", dest="autopilot", action="store_true")
    g.add_argument("--no-autopilot", dest="autopilot", action="store_false")
    p.set_defaults(autopilot=True)
    # --- Alpamayo 연동 (Thor planner_live 가 polling 하는 /latest 서버) ---
    p.add_argument("--alpamayo", action="store_true",
                   help="Alpamayo 입력 /latest HTTP 서버를 띄운다 "
                        "(모델 카메라 4대를 576x320 으로 렌더)")
    p.add_argument("--alpamayo-host", default="0.0.0.0",
                   help="/latest 서버 bind 주소 (기본 0.0.0.0)")
    p.add_argument("--alpamayo-port", type=int, default=18080,
                   help="/latest 서버 포트 (기본 18080)")
    p.add_argument("--alpamayo-clip-id", default="carla_live",
                   help="샘플 clip_id")
    # --- Alpamayo 결과 수신(제어) ---
    p.add_argument("--alpamayo-control", action="store_true",
                   help="Alpamayo UDP 경로를 받아 자율주행(키 O로 토글)")
    p.add_argument("--alpamayo-control-host", default="0.0.0.0",
                   help="제어 UDP 수신 bind 주소")
    p.add_argument("--alpamayo-control-port", type=int, default=5005,
                   help="제어 UDP 수신 포트 (Thor --udp-port 와 일치)")
    # --- 맵 ground-truth 경로 추종 (Alpamayo 없이 도로망 route 따라감) ---
    p.add_argument("--map-route", action="store_true",
                   help="맵 차선을 따라 앞으로 route를 뽑아 추종(키 G로 토글). "
                        "제어기는 Alpamayo와 동일한 WorldPathFollower 재사용")
    p.add_argument("--map-route-dist", type=float, default=200.0,
                   help="맵 route 생성 거리(m, 현재 차선 따라 앞으로)")
    p.add_argument("--map-route-step", type=float, default=2.0,
                   help="맵 route waypoint 간격(m)")
    # --- 디버그용 2D 맵 viz (스폰 위치를 0,0 으로 그린다) ---
    p.add_argument("--viz", action="store_true",
                   help="차량 위치 2D 맵 viz HTTP 서버를 띄운다 "
                        "(브라우저에서 http://localhost:<viz-port>/ )")
    p.add_argument("--viz-host", default="0.0.0.0",
                   help="viz 서버 bind 주소 (기본 0.0.0.0)")
    p.add_argument("--viz-port", type=int, default=8091,
                   help="viz 서버 포트 (기본 8091)")
    return p.parse_args()


def grid_shape(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def rear_axle_xy(transform, wheelbase):
    """차량 transform(월드)에서 후륜축 위치(x, y[m])와 yaw(deg)를 계산.

    후륜축 = 차량중심 - (L/2)*전방벡터. pure pursuit 제어(alpamayo_control)가
    쓰는 기준점과 동일하게 맞춰 viz 도 후륜축을 그린다.
    """
    yaw_deg = transform.rotation.yaw
    yaw = math.radians(yaw_deg)
    half = wheelbase / 2.0
    x = transform.location.x - half * math.cos(yaw)
    y = transform.location.y - half * math.sin(yaw)
    return x, y, yaw_deg


def ego_path_to_world_rax(rax, ray, yaw_deg, points):
    """후륜축 world pose(rax, ray, yaw_deg)에 ego-local 경로(x전방, y좌측+)를 붙여
    world 좌표로 변환.

      wx = rax + fx*cosφ + fy*sinφ
      wy = ray + fx*sinφ - fy*cosφ
    (ego +y=좌측이 φ=0 에서 world -y 로 가는 CARLA 좌수 좌표계 규약)
    """
    phi = math.radians(yaw_deg)
    c, s = math.cos(phi), math.sin(phi)
    out = []
    for p in points:
        fx, fy = float(p[0]), float(p[1])
        out.append((rax + fx * c + fy * s, ray + fx * s - fy * c))
    return out


def ego_path_to_world(transform, wheelbase, points):
    """ego-local 경로를 '현재 차 transform' 의 후륜축에 붙여 world 좌표로 변환."""
    rax, ray, yaw_deg = rear_axle_xy(transform, wheelbase)
    return ego_path_to_world_rax(rax, ray, yaw_deg, points)


def _wrap_deg(d):
    """각도차를 (-180, 180] 로 wrap."""
    return (d + 180.0) % 360.0 - 180.0


class PoseHistory:
    """월클럭(us) 타임스탬프별 후륜축 pose(rax, ray, yaw_deg) 이력 버퍼.

    plan["t0_us"] 는 alpamayo_bridge 가 time.time() 으로 찍어 Thor 가 그대로
    되돌려준 값이라 이 프로세스의 월클럭과 동일하다. 그래서 t0_us 시점의 ego
    pose 를 이 버퍼에서 조회하면, 들어온 경로를 '추론 입력 프레임(t0) 시점의
    내 차 위치' 에 정확히 앵커할 수 있다 (= inference time 전 위치).
    """

    def __init__(self, keep_s=3.0):
        self._keep_us = int(keep_s * 1e6)
        self._buf = deque()  # (t_us, rax, ray, yaw_deg) 시간 오름차순

    def add(self, t_us, rax, ray, yaw_deg):
        self._buf.append((int(t_us), rax, ray, yaw_deg))
        cutoff = int(t_us) - self._keep_us
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def lookup(self, t_us):
        """t_us 시점 pose(rax, ray, yaw_deg) 를 선형보간으로 반환. 버퍼 범위를
        벗어나면(너무 오래됐거나 미래) None."""
        if t_us is None or not self._buf:
            return None
        t_us = int(t_us)
        if t_us < self._buf[0][0] or t_us > self._buf[-1][0]:
            return None
        prev = self._buf[0]
        for cur in self._buf:
            if cur[0] >= t_us:
                span = cur[0] - prev[0]
                if span <= 0:
                    return prev[1], prev[2], prev[3]
                r = (t_us - prev[0]) / span
                rax = prev[1] + r * (cur[1] - prev[1])
                ray = prev[2] + r * (cur[2] - prev[2])
                yaw = prev[3] + r * _wrap_deg(cur[3] - prev[3])
                return rax, ray, yaw
            prev = cur
        return prev[1], prev[2], prev[3]


def lane_route_ahead(carla_map, transform, dist_m=200.0, step_m=2.0):
    """현재 위치의 차선을 따라 앞으로 dist_m 만큼 waypoint world 경로를 생성.

    map.get_waypoint 로 현재 pose를 도로(주행 차선)에 스냅한 뒤 wp.next(step)로
    차선을 따라 전진한다. WorldPathFollower.set_path 에 바로 넣을 수 있는
    [(x, y)..] world 좌표 리스트를 반환. (목적지 없이 도로 따라 주행)

    교차로 등에서 wp.next 가 여러 갈래를 주면 진행방향(차 yaw)에 가장 가까운
    분기를 골라 '직진 경향'으로 이어붙인다. (분기 없으면 그대로 진행)
    """
    wp = carla_map.get_waypoint(transform.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return []
    pts = [(wp.transform.location.x, wp.transform.location.y)]
    acc, guard = 0.0, 0
    max_pts = int(dist_m / max(step_m, 0.1)) + 10  # 무한루프 방지 상한
    while acc < dist_m and guard < max_pts:
        guard += 1
        nxts = wp.next(step_m)
        if not nxts:
            break
        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            # 여러 갈래: 현 waypoint 진행방향과 yaw 차가 가장 작은 분기 선택(직진)
            base_yaw = math.radians(wp.transform.rotation.yaw)
            base = (math.cos(base_yaw), math.sin(base_yaw))
            nxt = max(nxts, key=lambda w: (
                math.cos(math.radians(w.transform.rotation.yaw)) * base[0]
                + math.sin(math.radians(w.transform.rotation.yaw)) * base[1]))
        wp = nxt
        pts.append((wp.transform.location.x, wp.transform.location.y))
        acc += step_m
    return pts


class CameraView:
    """카메라 센서 콜백 → 최신 프레임을 pygame surface로 보관.

    bridge 가 주어지면 RGB(HWC) 프레임을 Alpamayo producer 로도 push 한다.
    """

    def __init__(self, name, bridge=None):
        self.name = name
        self.surface = None
        self.bridge = bridge  # AlpamayoBridge | None

    @staticmethod
    def on_image(weak_self, image):
        self = weak_self()
        if self is None:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        arr = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB (HWC)
        if self.bridge is not None:
            self.bridge.push_camera(self.name, arr)
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))


def draw_text(ren, font, text, color, pos):
    """Renderer에 텍스트를 그린다."""
    surf = font.render(text, True, color)
    tex = Texture.from_surface(ren, surf)
    tex.draw(dstrect=pygame.Rect(pos[0], pos[1],
                                 surf.get_width(), surf.get_height()))


class Slider:
    """컨트롤 창 안의 마우스 드래그 슬라이더 (창 로컬 좌표 기준)."""

    def __init__(self, x, w, y, lo, hi, label):
        self.x, self.w, self.y = x, w, y
        self.lo, self.hi = lo, hi
        self.label = label
        self.h = 10
        self.dragging = False

    def value_to_x(self, v):
        t = (v - self.lo) / (self.hi - self.lo)
        return int(self.x + t * self.w)

    def x_to_value(self, px):
        t = max(0.0, min(1.0, (px - self.x) / self.w))
        return self.lo + t * (self.hi - self.lo)

    def hit(self, pos):
        px, py = pos
        return (self.x - 12 <= px <= self.x + self.w + 12
                and self.y - 16 <= py <= self.y + self.h + 16)

    def draw(self, ren, value, font):
        draw_text(ren, font, self.label.format(value), (235, 235, 235),
                  (self.x, self.y - 26))
        ren.draw_color = pygame.Color(90, 90, 96)
        ren.fill_rect(pygame.Rect(self.x, self.y, self.w, self.h))
        ren.draw_color = pygame.Color(70, 160, 250)
        ren.fill_rect(pygame.Rect(self.x, self.y,
                                  self.value_to_x(value) - self.x, self.h))
        kx = self.value_to_x(value)
        ren.draw_color = pygame.Color(255, 255, 255)
        ren.fill_rect(pygame.Rect(kx - 7, self.y - 5, 14, self.h + 10))


def main():
    args = parse_args()

    names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    bad = [c for c in names if c not in CAMERA_NAMES]
    if bad:
        print(f"알 수 없는 카메라: {bad}. 사용 가능: {list(CAMERA_NAMES)}",
              file=sys.stderr)
        return
    if not names:
        print("카메라를 하나 이상 지정하세요.", file=sys.stderr)
        return

    cols, rows = grid_shape(len(names))
    tile_w = args.width // cols
    tile_h = args.height // rows

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    side_yaw = float(args.side_yaw)
    side_pitch = float(args.side_pitch)
    has_sides = any(n in SIDE_LOCATIONS for n in names)

    # --- Alpamayo producer (옵션) ---
    bridge = None
    if args.alpamayo:
        missing = [c for c in REQUIRED_CAMERAS if c not in names]
        if missing:
            print(f"[alpamayo] 모델 카메라 누락: {missing}. "
                  f"--cameras 에 {list(REQUIRED_CAMERAS)} 가 모두 있어야 합니다.",
                  file=sys.stderr)
            return
        bridge = AlpamayoBridge(host=args.alpamayo_host, port=args.alpamayo_port,
                                clip_id=args.alpamayo_clip_id)

    # --- Alpamayo 결과 수신/제어 (옵션) ---
    control_rx = None
    follower = None
    alpamayo_drive = False
    map_route_drive = False
    if args.alpamayo_control:
        control_rx = AlpamayoControlReceiver(host=args.alpamayo_control_host,
                                             port=args.alpamayo_control_port)

    # --- 디버그 맵 viz (옵션) ---
    viz = None
    if args.viz:
        viz = VizServer(host=args.viz_host, port=args.viz_port)

    actors = []
    side_cams = {}
    try:
        # --- 테슬라 스폰 ---
        tesla_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle = None
        for sp in world.get_map().get_spawn_points():
            vehicle = world.try_spawn_actor(tesla_bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            print("테슬라 스폰 실패.", file=sys.stderr)
            return
        actors.append(vehicle)
        print(f"테슬라 스폰 완료: {vehicle.type_id}")
        autopilot = bool(args.autopilot)
        vehicle.set_autopilot(autopilot)
        print("자동주행 켜짐" if autopilot else "수동운전 모드 (WASD)")

        # 차량 물리에서 휠베이스/최대조향각 1회 추출 (viz·제어 공통 기준점).
        # 후륜축 = center - L/2·forward. viz·제어·경로고정 모두 같은 L 사용.
        wheelbase_m, max_steer_deg = 2.875, 70.0
        try:
            wheels = vehicle.get_physics_control().wheels
            # 앞/뒤축 위치차(월드 cm)로 휠베이스 추정 (orientation 무관)
            fr, re = wheels[0].position, wheels[2].position
            wheelbase_m = math.hypot(fr.x - re.x, fr.y - re.y) / 100.0 or 2.875
            max_steer_deg = max(w.max_steer_angle for w in wheels) or 70.0
        except Exception:
            pass

        # 맵 viz: 후륜축 기준점으로 그린다.
        if viz is not None:
            viz.start()
            print(f"[viz] 기준점=후륜축 (wheelbase={wheelbase_m:.2f}m)")
            rx, ry, yaw0 = rear_axle_xy(vehicle.get_transform(), wheelbase_m)
            viz.update(rx, ry, yaw0, 0.0)

        # 제어기: closed-loop world-frame 횡방향 추종 (실제 차 pose 기준).
        # 두 횡방향 제어기를 준비해 키 M 으로 토글한다:
        #   pp_follower  : pure pursuit (기본)
        #   mpc_follower : LTV-MPC
        # 종방향(모델 속도 P)은 두 제어기가 공유. Alpamayo(--alpamayo-control)와
        # 맵 route(--map-route)가 같은 제어기를 재사용(한 번에 한 소스만 활성).
        carla_map = world.get_map()
        pp_follower = mpc_follower = follower = None
        use_mpc = False
        if control_rx is not None or args.map_route:
            pp_follower = WorldPathFollower(wheelbase_m=wheelbase_m,
                                            max_steer_deg=max_steer_deg)
            mpc_follower = LateralMPC(wheelbase_m=wheelbase_m,
                                      max_steer_deg=max_steer_deg)
            follower = pp_follower
            if control_rx is not None:
                control_rx.start()
                print(f"[alpamayo] 제어 준비 (횡방향 pure pursuit/MPC, "
                      f"wheelbase={wheelbase_m:.2f}m, max_steer={max_steer_deg:.1f}deg). "
                      f"키 O 주행, T 앵커(현재/t0), V 종방향속도(모델/고정), "
                      f"L lookahead(%/거리), M 횡방향(PP/MPC), [ ] MPC 조향 slew 조정")
            if args.map_route:
                print(f"[map-route] 맵 차선 추종 준비 (dist={args.map_route_dist:.0f}m, "
                      f"step={args.map_route_step:.1f}m). 키 G 로 주행 토글")

        def set_follower_path(points, velocities=None):
            """pp/mpc 두 제어기에 동일 경로를 설정(활성 토글 시 즉시 사용 가능)."""
            if pp_follower is not None:
                pp_follower.set_path(points, velocities)
                mpc_follower.set_path(points, velocities)

        # --- 카메라 부착 (tele는 좁은 화각) ---
        # Alpamayo 모델 카메라는 계약 해상도(576x320)로 네이티브 렌더 → 리사이즈 불필요.
        # 화면 격자는 텍스처가 타일 크기로 자동 스케일하므로 표시엔 문제 없음.
        cam_bp = bp_lib.find("sensor.camera.rgb")

        views = []
        for name in names:
            is_model_cam = bridge is not None and name in REQUIRED_CAMERAS
            if is_model_cam:
                cam_bp.set_attribute("image_size_x", str(IMAGE_W))
                cam_bp.set_attribute("image_size_y", str(IMAGE_H))
            else:
                cam_bp.set_attribute("image_size_x", str(tile_w))
                cam_bp.set_attribute("image_size_y", str(tile_h))
            cam_bp.set_attribute(
                "fov", str(args.tele_fov) if name == "tele" else str(args.fov))
            if name in SIDE_LOCATIONS:
                transform = side_transform(name, side_yaw, side_pitch)
            else:
                transform = FIXED_TRANSFORMS[name]
            view = CameraView(name, bridge=bridge if is_model_cam else None)
            cam = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
            actors.append(cam)
            if name in SIDE_LOCATIONS:
                side_cams[name] = cam
            wv = weakref.ref(view)
            cam.listen(lambda image, wv=wv: CameraView.on_image(wv, image))
            views.append(view)
        print(f"카메라 {len(views)}대 부착 완료: {names}")
        if bridge is not None:
            bridge.start()

        # --- IMU / GNSS 센서 부착 (10Hz, 노이즈 없음) ---
        telemetry = {"imu": None, "gnss": None}
        imu_bp = bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.1")   # 10Hz
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        actors.append(imu)
        imu.listen(lambda m: telemetry.__setitem__("imu", m))

        gnss_bp = bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", "0.1")  # 10Hz
        gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        actors.append(gnss)
        gnss.listen(lambda m: telemetry.__setitem__("gnss", m))
        print("IMU/GNSS 센서 부착 완료 (각 10Hz, 노이즈 없음)")

        def apply_sides():
            for nm, cam in side_cams.items():
                cam.set_transform(side_transform(nm, side_yaw, side_pitch))

        # --- 창 두 개: 카메라 격자 + 컨트롤 ---
        pygame.init()
        pygame.font.init()
        main_win = Window(f"CARLA Tesla 멀티 카메라 {cols}x{rows}",
                          size=(args.width, args.height))
        main_ren = Renderer(main_win)
        font = pygame.font.SysFont("monospace", 18, bold=True)
        hud_font = pygame.font.SysFont("monospace", 15, bold=True)

        ctrl_win = ctrl_ren = None
        sliders = []
        yaw_slider = pitch_slider = None
        if has_sides:
            cw, ch = 430, 230
            ctrl_win = Window("카메라 각도 조절", size=(cw, ch))
            ctrl_win.position = (60, 60)
            ctrl_ren = Renderer(ctrl_win)
            pitch_slider = Slider(24, cw - 48, 86, -90.0, 90.0,
                                  "상하(pitch): {:5.1f}  (-위 +아래)")
            yaw_slider = Slider(24, cw - 48, 150, 0.0, 180.0,
                                "좌우(yaw): {:5.1f}  (0앞 90옆 180뒤)")
            sliders = [yaw_slider, pitch_slider]
        cfont = pygame.font.SysFont("monospace", 16, bold=True)
        chint = pygame.font.SysFont("monospace", 14)

        main_id = main_win.id
        ctrl_id = ctrl_win.id if ctrl_win else None
        active_id = main_id
        steer = 0.0  # 현재 조향각(부드러운 전환용)
        next_alpamayo_tick = time.monotonic()  # 10Hz 샘플 적재 시점
        last_viz_plan_id = None  # viz 고정 경로 갱신용(새 plan 감지)
        anchor_t0 = False        # True 면 경로를 t0(추론 입력 프레임) pose 에 앵커
        pose_hist = PoseHistory(keep_s=3.0)  # t0 pose 조회용 월클럭 pose 이력

        def evt_win_id(event):
            ew = getattr(event, "window", None)
            return getattr(ew, "id", None) if ew is not None else None

        print("창 2개가 떴습니다. 컨트롤 창에서 각도 조절, ESC/Q 종료.")
        running = True
        while running:
            for event in pygame.event.get():
                ewid = evt_win_id(event)
                if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                    running = False
                elif event.type == pygame.WINDOWFOCUSGAINED and ewid is not None:
                    active_id = ewid
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_p:
                        autopilot = not autopilot
                        if autopilot:
                            alpamayo_drive = False  # 모드 충돌 방지
                            map_route_drive = False
                        vehicle.set_autopilot(autopilot)
                        print("자동주행" if autopilot else "수동운전(WASD)")
                    elif event.key == pygame.K_o and control_rx is not None:
                        alpamayo_drive = not alpamayo_drive
                        if alpamayo_drive:
                            autopilot = False
                            vehicle.set_autopilot(False)
                            map_route_drive = False  # 모드 충돌 방지
                        print("Alpamayo 주행 ON" if alpamayo_drive
                              else "Alpamayo 주행 OFF (수동)")
                    elif event.key == pygame.K_t and control_rx is not None:
                        anchor_t0 = not anchor_t0
                        print(f"[anchor] 경로 앵커 = "
                              f"{'t0 pose(추론 입력 시점)' if anchor_t0 else '현재 pose'}")
                    elif event.key == pygame.K_v and follower is not None:
                        # 종방향은 두 제어기 공유 → 둘 다 반영
                        ums = not follower.use_model_speed
                        pp_follower.use_model_speed = ums
                        mpc_follower.use_model_speed = ums
                        print(f"[속도] 종방향 목표 = "
                              f"{'모델 pred_v_mps' if ums else '고정 목표속도'}")
                    elif event.key == pygame.K_l and pp_follower is not None:
                        # lookahead 방식은 pure pursuit 전용 설정
                        pp_follower.use_index_lookahead = not pp_follower.use_index_lookahead
                        print(f"[lookahead] (PP) 목표점 선택 = "
                              f"{'남은경로 %s%% 인덱스' % int(pp_follower.pp_index_pct * 100) if pp_follower.use_index_lookahead else '속도기반 거리(Ld)'}")
                    elif event.key == pygame.K_m and mpc_follower is not None:
                        use_mpc = not use_mpc
                        follower = mpc_follower if use_mpc else pp_follower
                        print(f"[횡방향] 제어기 = {'LTV-MPC' if use_mpc else 'pure pursuit'}")
                    elif event.key == pygame.K_LEFTBRACKET and mpc_follower is not None:
                        mpc_follower.max_dsteer = max(0.005, mpc_follower.max_dsteer - 0.005)
                        print(f"[MPC] 조향 slew 제한 = {mpc_follower.max_dsteer:.3f}/frame (더 부드럽게)")
                    elif event.key == pygame.K_RIGHTBRACKET and mpc_follower is not None:
                        mpc_follower.max_dsteer = min(0.5, mpc_follower.max_dsteer + 0.005)
                        print(f"[MPC] 조향 slew 제한 = {mpc_follower.max_dsteer:.3f}/frame (더 반응성)")
                    elif event.key == pygame.K_g and follower is not None \
                            and args.map_route:
                        map_route_drive = not map_route_drive
                        if map_route_drive:
                            autopilot = False
                            vehicle.set_autopilot(False)
                            alpamayo_drive = False  # 모드 충돌 방지
                            # 지금 pose 기준으로 앞 차선 route 새로 생성 후 고정
                            pts = lane_route_ahead(
                                carla_map, vehicle.get_transform(),
                                args.map_route_dist, args.map_route_step)
                            set_follower_path(pts)
                            if viz is not None:
                                viz.set_fixed_path(pts)
                            print(f"[map-route] 주행 ON — 차선 route {len(pts)}점 고정")
                        else:
                            print("[map-route] 주행 OFF (수동)")
                    elif has_sides and event.key == pygame.K_LEFT:
                        side_yaw = max(0.0, side_yaw - 2.0); apply_sides()
                    elif has_sides and event.key == pygame.K_RIGHT:
                        side_yaw = min(180.0, side_yaw + 2.0); apply_sides()
                    elif has_sides and event.key == pygame.K_UP:
                        side_pitch = max(-90.0, side_pitch - 2.0); apply_sides()
                    elif has_sides and event.key == pygame.K_DOWN:
                        side_pitch = min(90.0, side_pitch + 2.0); apply_sides()
                    elif has_sides and event.key == pygame.K_r:
                        side_yaw, side_pitch = 90.0, 0.0; apply_sides()
                elif has_sides and event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1:
                    on_ctrl = (ewid == ctrl_id) or (ewid is None
                                                    and active_id == ctrl_id)
                    if on_ctrl:
                        if yaw_slider.hit(event.pos):
                            yaw_slider.dragging = True
                            side_yaw = yaw_slider.x_to_value(event.pos[0])
                            apply_sides()
                        elif pitch_slider.hit(event.pos):
                            pitch_slider.dragging = True
                            side_pitch = pitch_slider.x_to_value(event.pos[0])
                            apply_sides()
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    for s in sliders:
                        s.dragging = False
                elif has_sides and event.type == pygame.MOUSEMOTION:
                    if yaw_slider.dragging:
                        side_yaw = yaw_slider.x_to_value(event.pos[0])
                        apply_sides()
                    elif pitch_slider.dragging:
                        side_pitch = pitch_slider.x_to_value(event.pos[0])
                        apply_sides()

            alpa_hud = None  # HUD 표시용 (steer/throttle/brake/cte)
            # --- 새 경로 수신 감지: 받은 경로를 world 에 고정 ---
            # 앵커 pose = 현재(기본) 또는 t0(추론 입력 프레임 시점, 키 T). follower 는
            # 이후 실제 차 pose 로 이 world 경로를 closed-loop 추종한다.
            if control_rx is not None and not map_route_drive:
                # t0 앵커 조회용으로 매 프레임 현재 후륜축 pose 를 월클럭으로 적재
                nrax, nray, nyaw = rear_axle_xy(vehicle.get_transform(), wheelbase_m)
                pose_hist.add(time.time() * 1e6, nrax, nray, nyaw)

                newplan, _age = control_rx.latest()
                if newplan is not None and id(newplan) != last_viz_plan_id:
                    last_viz_plan_id = id(newplan)
                    anchor_src = "현재"
                    anchored = (pose_hist.lookup(newplan.get("t0_us"))
                                if anchor_t0 else None)
                    if anchored is not None:
                        wpts = ego_path_to_world_rax(anchored[0], anchored[1],
                                                     anchored[2], newplan["points"])
                        anchor_src = "t0"
                    else:
                        wpts = ego_path_to_world(vehicle.get_transform(),
                                                 wheelbase_m, newplan["points"])
                        if anchor_t0:
                            anchor_src = "현재(t0 조회실패)"
                    set_follower_path(wpts, newplan.get("v_mps"))
                    if viz is not None:
                        viz.set_fixed_path(wpts)
                    seq = newplan.get("seq")
                    now = time.time()
                    ts = time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}"
                    print(f"[{ts}] 새 경로 seq={seq} 고정 ({follower.n_points}점, "
                          f"앵커={anchor_src}, "
                          f"{'주행중' if alpamayo_drive else '대기(O로 시작)'})", flush=True)

            # --- 자율주행: 고정 world 경로 closed-loop 추종 ---
            # (Alpamayo=O 또는 맵 route=G, 둘 다 같은 WorldPathFollower 사용)
            if (alpamayo_drive or map_route_drive) and follower is not None:
                vel = vehicle.get_velocity()
                speed_mps = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                tr = vehicle.get_transform()
                rax, ray, ryaw = rear_axle_xy(tr, wheelbase_m)
                control = carla.VehicleControl()
                if not follower.has_path:
                    control.throttle, control.brake, control.steer = 0.0, 0.3, 0.0
                    alpa_hud = "Alpamayo 경로 대기"
                else:
                    st, th, br = follower.compute(rax, ray, ryaw, speed_mps)
                    if follower.finished:
                        control.throttle, control.brake, control.steer = 0.0, 0.5, 0.0
                        alpa_hud = (f"경로 끝 도달 정지 "
                                    f"(i={follower.last_i_goal}/{follower.n_points})")
                    else:
                        control.steer, control.throttle, control.brake = st, th, br
                        alpa_hud = (f"{'MPC' if use_mpc else 'PP'} "
                                    f"steer={st:+.2f} thr={th:.2f} brk={br:.2f} "
                                    f"cte={follower.last_cte:.2f}m "
                                    f"ld={follower.last_ld:.1f}m[{follower.last_lookahead_mode}] "
                                    f"v*={follower.last_v_target * 3.6:.0f}kph({follower.last_v_source}) "
                                    f"gi={follower.last_i_goal}/{follower.n_points}")
                vehicle.apply_control(control)
            # --- 수동운전: 눌린 키로 차량 제어 ---
            elif not autopilot:
                keys = pygame.key.get_pressed()
                control = carla.VehicleControl()
                if keys[pygame.K_w] and not keys[pygame.K_s]:
                    control.throttle, control.reverse = 0.6, False
                elif keys[pygame.K_s] and not keys[pygame.K_w]:
                    control.throttle, control.reverse = 0.6, True
                if keys[pygame.K_a] and not keys[pygame.K_d]:
                    steer = max(-1.0, steer - 0.05)
                elif keys[pygame.K_d] and not keys[pygame.K_a]:
                    steer = min(1.0, steer + 0.05)
                else:
                    steer *= 0.6  # 키를 떼면 중앙으로 복귀
                control.steer = round(steer, 3)
                control.hand_brake = keys[pygame.K_SPACE]
                vehicle.apply_control(control)

            # --- Alpamayo 샘플 적재 (10Hz): 카메라 4장 + ego 포즈 ---
            if bridge is not None and time.monotonic() >= next_alpamayo_tick:
                next_alpamayo_tick += 0.1
                tr = vehicle.get_transform()
                fwd, right = tr.get_forward_vector(), tr.get_right_vector()
                pose = ((tr.location.x, tr.location.y, tr.location.z),
                        (fwd.x, fwd.y, fwd.z),
                        (right.x, right.y, right.z),
                        math.radians(tr.rotation.yaw))
                bridge.tick(pose)

            # --- 맵 viz 갱신 (스폰 원점 기준 후륜축 위치/궤적) ---
            # 고정 경로(set_fixed_path)는 위 '새 경로 수신 감지' 블록에서 설정됨.
            if viz is not None:
                vv = vehicle.get_velocity()
                spd = math.sqrt(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z)
                rx, ry, ryaw = rear_axle_xy(vehicle.get_transform(), wheelbase_m)
                viz.update(rx, ry, ryaw, spd)

            # --- 카메라 격자 그리기 ---
            main_ren.draw_color = pygame.Color(20, 20, 20)
            main_ren.clear()
            for i, view in enumerate(views):
                x = (i % cols) * tile_w
                y = (i // cols) * tile_h
                if view.surface is not None:
                    tex = Texture.from_surface(main_ren, view.surface)
                    tex.draw(dstrect=pygame.Rect(x, y, tile_w, tile_h))
                tag = view.name
                if view.name == "tele":
                    tag = f"tele (FOV {args.tele_fov}°)"
                elif view.name in SIDE_LOCATIONS:
                    tag = f"{view.name} (yaw {side_yaw:.0f}° pitch {side_pitch:.0f}°)"
                draw_text(main_ren, font, tag, (255, 255, 0), (x + 6, y + 4))
            # 주행 상태 + 센서 HUD (좌하단, 아래에서 위로 쌓음)
            v = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)  # km/h
            if alpamayo_drive:
                mode = "Alpamayo 주행 (O:해제)"
            elif autopilot:
                mode = "자동주행 (P:수동)"
            else:
                mode = "수동운전 WASD (P:자동)"
            lines = [(font, f"{mode}   {speed:5.1f} km/h")]
            if alpa_hud is not None:
                lines.append((hud_font, alpa_hud))
            imu_m = telemetry["imu"]
            if imu_m is not None:
                a, g = imu_m.accelerometer, imu_m.gyroscope
                comp = math.degrees(imu_m.compass)
                lines.append((hud_font,
                    f"IMU accel[{a.x:+5.2f} {a.y:+5.2f} {a.z:+5.2f}]m/s^2"
                    f"  gyro[{g.x:+5.2f} {g.y:+5.2f} {g.z:+5.2f}]rad/s"
                    f"  compass {comp:5.1f}deg"))
            gnss_m = telemetry["gnss"]
            if gnss_m is not None:
                lines.append((hud_font,
                    f"GNSS lat {gnss_m.latitude:+.6f}  lon {gnss_m.longitude:+.6f}"
                    f"  alt {gnss_m.altitude:6.1f}m"))
            y = args.height - 8 - 22 * len(lines)
            for f, text in lines:
                draw_text(main_ren, f, text, (120, 255, 160), (8, y))
                y += 22
            main_ren.present()

            # --- 컨트롤 창 그리기 ---
            if ctrl_ren is not None:
                ctrl_ren.draw_color = pygame.Color(28, 28, 34)
                ctrl_ren.clear()
                draw_text(ctrl_ren, cfont, "좌우 카메라 각도 조절",
                          (120, 200, 255), (24, 16))
                yaw_slider.draw(ctrl_ren, side_yaw, cfont)
                pitch_slider.draw(ctrl_ren, side_pitch, cfont)
                draw_text(ctrl_ren, chint,
                          "키: <-/-> 좌우  ↑/↓ 상하  R 리셋  ESC/Q 종료",
                          (190, 190, 190), (24, 196))
                ctrl_ren.present()

            pygame.time.wait(16)  # ~60fps

    finally:
        print("정리 중...")
        if viz is not None:
            viz.stop()
        if control_rx is not None:
            control_rx.stop()
        if bridge is not None:
            bridge.stop()
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass
        pygame.quit()
        print("종료 완료 ✅")


if __name__ == "__main__":
    main()
