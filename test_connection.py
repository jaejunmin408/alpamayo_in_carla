#!/usr/bin/env python
"""CARLA 서버 연결 및 기본 동작 테스트.

서버(start_server.sh)가 떠 있는 상태에서 venv 파이썬으로 실행:
    ./venv/bin/python test_connection.py
"""
import carla
import random
import time


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print("연결 성공!")
    print("  맵:", world.get_map().name)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    print("  스폰 지점 수:", len(spawn_points))

    # 차량 한 대 스폰
    vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))
    transform = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, transform)
    print("  차량 스폰:", vehicle.type_id)

    # 자동주행(트래픽 매니저)에 맡겨 5초간 주행
    vehicle.set_autopilot(True)
    for i in range(5):
        loc = vehicle.get_location()
        print(f"  t={i}s  위치=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
        time.sleep(1.0)

    vehicle.destroy()
    print("정리 완료. 테스트 성공 ✅")


if __name__ == "__main__":
    main()
