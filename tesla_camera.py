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

import carla
import numpy as np
import pygame
from pygame._sdl2.video import Window, Renderer, Texture

from alpamayo_bridge import AlpamayoBridge, REQUIRED_CAMERAS, IMAGE_W, IMAGE_H
from alpamayo_control import AlpamayoControlReceiver, PathFollower, STALE_PLAN_S


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
    return p.parse_args()


def grid_shape(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


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
    last_logged_seq = None       # 새 plan 로깅용
    last_plan_log_t = 0.0        # 직전 plan 로그 시각(간격 측정)
    if args.alpamayo_control:
        control_rx = AlpamayoControlReceiver(host=args.alpamayo_control_host,
                                             port=args.alpamayo_control_port)

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

        # 제어기: 차량 물리에서 휠베이스/최대조향각 추출
        if control_rx is not None:
            try:
                pc = vehicle.get_physics_control()
                wheels = pc.wheels
                max_steer = max(w.max_steer_angle for w in wheels) or 70.0
                # 앞/뒤 축 위치차로 휠베이스 추정 (cm -> m)
                xs = [w.position.x for w in wheels]
                wheelbase = abs(max(xs) - min(xs)) / 100.0 or 2.875
            except Exception:
                max_steer, wheelbase = 70.0, 2.875
            follower = PathFollower(wheelbase_m=wheelbase, max_steer_deg=max_steer)
            control_rx.start()
            print(f"[alpamayo] 제어 준비 (wheelbase={wheelbase:.2f}m, "
                  f"max_steer={max_steer:.1f}deg). 키 O 로 Alpamayo 주행 토글")

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
                        if autopilot and alpamayo_drive:
                            alpamayo_drive = False  # 모드 충돌 방지
                        vehicle.set_autopilot(autopilot)
                        print("자동주행" if autopilot else "수동운전(WASD)")
                    elif event.key == pygame.K_o and control_rx is not None:
                        alpamayo_drive = not alpamayo_drive
                        if alpamayo_drive:
                            autopilot = False
                            vehicle.set_autopilot(False)
                        print("Alpamayo 주행 ON" if alpamayo_drive
                              else "Alpamayo 주행 OFF (수동)")
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

            alpa_hud = None  # HUD 표시용 (steer/throttle/brake/age)
            # --- Alpamayo 주행: UDP 경로 추종 ---
            if alpamayo_drive and follower is not None:
                vel = vehicle.get_velocity()
                speed_mps = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                plan, age = control_rx.latest()
                control = carla.VehicleControl()
                if plan is None or age > STALE_PLAN_S:
                    # plan 없음/오래됨 -> 감속 정지
                    control.throttle, control.brake, control.steer = 0.0, 0.3, 0.0
                    alpa_hud = f"Alpamayo 대기/stale (age={age:.2f}s)"
                else:
                    st, th, br = follower.compute(plan, speed_mps, age)
                    control.steer, control.throttle, control.brake = st, th, br
                    alpa_hud = (f"Alpamayo steer={st:+.2f} thr={th:.2f} brk={br:.2f} "
                                f"age={age:.2f}s pkt={control_rx.stats()['packets']}")
                    # 새 plan(seq)이 들어왔을 때만 시간/지연/제어값 한 줄 기록
                    seq = plan.get("seq")
                    if seq != last_logged_seq:
                        now = time.time()
                        ts = time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}"
                        t0_us = plan.get("t0_us") or 0
                        e2e_ms = (now * 1e6 - t0_us) / 1000.0 if t0_us else float("nan")
                        infer_ms = (plan.get("inference_time_s") or 0.0) * 1000.0
                        gap_ms = (now - last_plan_log_t) * 1000.0 if last_plan_log_t else 0.0
                        print(f"[{ts}] plan seq={seq} e2e={e2e_ms:6.0f}ms "
                              f"infer={infer_ms:5.0f}ms gap={gap_ms:6.0f}ms "
                              f"s={follower.last_slice_s:4.1f}m gi={follower.last_i_goal:2d} "
                              f"steer={st:+.2f} thr={th:.2f} "
                              f"brk={br:.2f} v={speed_mps * 3.6:4.1f}km/h",
                              flush=True)
                        last_logged_seq = seq
                        last_plan_log_t = now
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
