#!/usr/bin/env python
"""맵에 waypoint 경로를 생성하고 추종하는 첫 자율주행 스크립트.

서버(start_server.sh)가 떠 있는 상태에서 venv 파이썬으로 실행:
    ./venv/bin/python follow_waypoints.py                      # pure pursuit (기본)
    ./venv/bin/python follow_waypoints.py --controller pid     # 내장 heading-PID
    ./venv/bin/python follow_waypoints.py --speed 30 --ld-gain 0.6

구성:
  1) GlobalRoutePlanner 로 출발지→목적지 waypoint 경로 생성
  2) 횡(조향) 제어: pure pursuit  또는  CARLA 내장 heading-error PID
     종(속도) 제어: PIDLongitudinalController (둘 다 공통)
  3) 동기 모드(fixed_delta) 루프에서 경로를 끝까지 추종

나중에 2)를 openpilot 출력으로 교체하면 "외부 경로 주입 → 제어만 테스트" 목표로 이어집니다.
"""
import argparse
import math
import os
import sys

import carla

# CARLA가 제공하는 agents 패키지(PythonAPI/carla 안)는 pip에 안 깔리므로 경로 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "CARLA_0.9.16", "PythonAPI", "carla"))

from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402
from agents.navigation.controller import (  # noqa: E402
    VehiclePIDController,
    PIDLongitudinalController,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--controller", choices=["pure_pursuit", "pid"],
                   default="pure_pursuit", help="횡(조향) 제어 방식")
    p.add_argument("--speed", type=float, default=20.0, help="목표 속도(km/h)")
    p.add_argument("--dist", type=float, default=150.0, help="목적지까지 대략 거리(m)")
    p.add_argument("--sampling", type=float, default=2.0, help="waypoint 간격(m)")
    p.add_argument("--ld-gain", type=float, default=0.5,
                   help="pure pursuit lookahead 게인 k (L_d = k*속도[m/s])")
    p.add_argument("--ld-min", type=float, default=4.0, help="최소 lookahead 거리(m)")
    p.add_argument("--reach", type=float, default=3.0, help="목적지 도달 판정 반경(m)")
    p.add_argument("--dt", type=float, default=0.05, help="시뮬레이션 스텝(s)")
    return p.parse_args()


def pick_destination(carla_map, start_loc, want_dist):
    """출발점에서 want_dist(m) 정도 떨어진 스폰 지점을 목적지로 고른다."""
    best, best_diff = None, float("inf")
    for sp in carla_map.get_spawn_points():
        d = sp.location.distance(start_loc)
        if d < 10.0:  # 출발점 자기 자신 제외
            continue
        diff = abs(d - want_dist)
        if diff < best_diff:
            best, best_diff = sp, diff
    return best


def draw_route(world, route, life=60.0):
    for wp, _ in route:
        world.debug.draw_point(
            wp.transform.location + carla.Location(z=0.3),
            size=0.08, color=carla.Color(0, 200, 255), life_time=life,
        )


class PurePursuitLateral:
    """기하학적 pure pursuit 횡제어.

    후륜축 기준으로 경로상 lookahead 거리(L_d = k*v) 앞의 점을 목표로 잡고,
    그 점을 지나는 원호의 곡률로 조향각을 계산한다:
        δ = atan2(2*L*sin(α), L_d)        (L=축거, α=차체기준 목표점 방위각)
    """

    def __init__(self, vehicle, route, k=0.5, ld_min=4.0):
        self._vehicle = vehicle
        self._pts = [wp.transform.location for wp, _ in route]
        self._k = k
        self._ld_min = ld_min
        self._idx = 0  # 직전 목표 인덱스(앞으로만 검색)

        phys = vehicle.get_physics_control()
        # 축거 L: 앞바퀴와 뒷바퀴 위치 차로 추정 (cm → m)
        wheels = phys.wheels
        fr = wheels[0].position
        re = wheels[2].position
        self._L = math.hypot(fr.x - re.x, fr.y - re.y) / 100.0 or 2.9
        # 조향 정규화용 최대 조향각(deg)
        self._max_steer = max(w.max_steer_angle for w in wheels) or 70.0

    def _speed_ms(self):
        v = self._vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def run_step(self):
        tf = self._vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        fwd = (math.cos(yaw), math.sin(yaw))
        # 후륜축 위치 = 차량중심 - (L/2)*진행방향
        rear = (tf.location.x - (self._L / 2.0) * fwd[0],
                tf.location.y - (self._L / 2.0) * fwd[1])

        ld = max(self._ld_min, self._k * self._speed_ms())

        # 후륜축에서 ld 이상 떨어진 첫 경로점을 목표로(앞으로만 탐색)
        target = self._pts[-1]
        for i in range(self._idx, len(self._pts)):
            dx = self._pts[i].x - rear[0]
            dy = self._pts[i].y - rear[1]
            if math.hypot(dx, dy) >= ld:
                self._idx = i
                target = self._pts[i]
                break

        # 차체 좌표계에서 목표점 방위각 α
        dx = target.x - rear[0]
        dy = target.y - rear[1]
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy   # 전방(+)
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy   # 좌(+)
        alpha = math.atan2(local_y, local_x)

        delta = math.atan2(2.0 * self._L * math.sin(alpha), ld)  # rad
        steer = delta / math.radians(self._max_steer)
        return max(-1.0, min(1.0, steer))


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    carla_map = world.get_map()
    print(f"연결 성공! 맵: {carla_map.name} / 횡제어: {args.controller}")

    original_settings = world.get_settings()
    vehicle = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.dt
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        start_tf = carla_map.get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, start_tf)
        print("차량 스폰:", vehicle.type_id)

        grp = GlobalRoutePlanner(carla_map, args.sampling)
        dest_tf = pick_destination(carla_map, start_tf.location, args.dist)
        route = grp.trace_route(start_tf.location, dest_tf.location)
        goal_loc = route[-1][0].transform.location
        print(f"경로 waypoint 수: {len(route)} (간격 {args.sampling}m)")
        draw_route(world, route)

        # 종방향(속도) PID — 두 방식 공통
        lon = PIDLongitudinalController(
            vehicle, K_P=1.0, K_I=0.05, K_D=0.0, dt=args.dt)

        # 횡방향 제어기 선택
        if args.controller == "pure_pursuit":
            lat = PurePursuitLateral(vehicle, route, k=args.ld_gain, ld_min=args.ld_min)
            pid_full = None
        else:
            lat = None
            pid_full = VehiclePIDController(
                vehicle,
                args_lateral={"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": args.dt},
                args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": args.dt},
            )

        world.tick()  # 첫 스텝 안정화

        spectator = world.get_spectator()
        max_steps = int(60.0 / args.dt) * 5  # 안전 상한(약 5분)
        for step in range(max_steps):
            veh_loc = vehicle.get_location()
            if veh_loc.distance(goal_loc) < args.reach:
                print("목적지 도착 ✅")
                break

            if args.controller == "pure_pursuit":
                steer = lat.run_step()
                accel = lon.run_step(args.speed)  # throttle(+)/brake(-) 신호
                control = carla.VehicleControl()
                control.steer = steer
                if accel >= 0.0:
                    control.throttle = min(accel, 0.75)
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = min(-accel, 0.3)
            else:
                # 내장 PID: 가장 가까운(앞쪽) waypoint를 목표로 단순 추적
                control = pid_full.run_step(args.speed, route[min(step, len(route) - 1)][0])

            vehicle.apply_control(control)

            tf = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(z=25),
                carla.Rotation(pitch=-70, yaw=tf.rotation.yaw),
            ))

            world.tick()

            if step % 40 == 0:
                v = vehicle.get_velocity()
                spd = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
                print(f"  step={step:4d}  goal_d={veh_loc.distance(goal_loc):5.1f}m  "
                      f"속도={spd:5.1f}km/h  thr={control.throttle:.2f} "
                      f"steer={control.steer:+.2f} brake={control.brake:.2f}")
        else:
            print("최대 스텝 도달(목적지 미도착)")

    finally:
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)  # 비동기 모드로 복구
        print("정리 완료.")


if __name__ == "__main__":
    main()
