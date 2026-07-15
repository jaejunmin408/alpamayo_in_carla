#!/usr/bin/env python
"""수동 경로 주입기: 좌회전/우회전/직진 명령 -> Alpamayo text_json UDP 로 전송.

tesla_camera.py 를 `--alpamayo-control` 로 띄우고 창에서 O 키로 Alpamayo 주행을
켜 두면, 이 스크립트에서 친 명령대로 ego-local 경로를 만들어 UDP(기본 5005)로
쏜다. 즉 실제 모델 대신 "내가 만든 경로"로 제어(steer/throttle/brake)가 잘 먹는지
테스트하는 용도.

전송 방식: 명령 한 번 = 패킷 한 번 (one-shot).
    한 번 보낸 경로(기본 15m)를 끝까지 추종하고, 경로 끝에 도달하면 정지한다.
    (tesla_camera 가 stale 을 무시하고 follower.finished 로 종료를 판정)
    다른 경로로 가려면 다시 명령을 친다 -> 그 순간 차 위치를 기준으로 재고정.

좌표계 (alpamayo_bridge / alpamayo_control 과 동일):
    ego-local, x 전방(+), y 좌측(+).  좌회전 = +y, 우회전 = -y, 직진 = y 0.
    (제어기 STEER_SIGN=-1 이 +y 를 CARLA 좌회전으로 매핑)

명령 (대소문자/한영 무관, 엔터로 실행 -> 즉시 1회 전송):
    직진 / s / straight       직진 경로 1회 전송
    좌 / 좌회전 / l / left      좌회전 경로 1회 전송
    우 / 우회전 / r / right     우회전 경로 1회 전송
    <명령> <숫자>              그 명령의 회전반경(m) 지정. 예) "l 8", "우 12"
    radius <숫자> / R<숫자>     기본 회전반경 변경
    st / status               현재 상태
    ? / help                  도움말
    q / quit / exit           종료

실행:
    ./venv/bin/python path_commander.py                 # localhost:5005
    ./venv/bin/python path_commander.py --port 5005 --radius 10
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="tesla_camera 제어 UDP 수신 호스트 (기본 127.0.0.1)")
    p.add_argument("--port", type=int, default=5005,
                   help="제어 UDP 포트 (tesla_camera --alpamayo-control-port 와 일치)")
    p.add_argument("--radius", type=float, default=10.0,
                   help="기본 회전반경(m). 작을수록 급회전 (기본 10)")
    p.add_argument("--length", type=float, default=15.0,
                   help="경로 길이(m, 호길이). 이 거리만큼 가고 멈춘다 (기본 15)")
    p.add_argument("--plan-dt", type=float, default=0.1,
                   help="경로 점 간 시간 간격(초, 기본 0.1). 점 간격=speed*plan_dt")
    p.add_argument("--speed", type=float, default=10.0,
                   help="경로 점 간격 산정용 속도(km/h). 실제 목표속도는 "
                        "수신측 TARGET_SPEED_KMH 이 결정 (기본 10)")
    return p.parse_args()


def build_path(kappa: float, ds: float, n: int):
    """등곡률 원호를 ego-local 점열로 생성.

    kappa(1/m): +면 좌회전(+y), -면 우회전(-y), 0이면 직진.
    시작점은 원점(0,0), 시작 진행방향은 +x.
      직진: x=s, y=0
      원호: x=sin(k s)/k, y=(1-cos(k s))/k, yaw=k s   (s=호길이=i*ds)
    반환: (pts[[x,y,0.0]..], yaws[rad])
    """
    pts, yaws = [], []
    for i in range(n):
        s = i * ds
        if abs(kappa) < 1e-6:
            x, y, yaw = s, 0.0, 0.0
        else:
            x = math.sin(kappa * s) / kappa
            y = (1.0 - math.cos(kappa * s)) / kappa
            yaw = kappa * s
        pts.append([round(x, 4), round(y, 4), 0.0])
        yaws.append(round(yaw, 5))
    return pts, yaws


class Commander:
    def __init__(self, args):
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (args.host, args.port)
        self._radius = args.radius
        self._seq = 0
        self._last = "-"             # 마지막 전송 명령 라벨
        # 점 간격 ds = v[m/s]*dt, 점 개수 n 은 총 호길이(length)에 맞춘다.
        self.ds = (args.speed / 3.6) * args.plan_dt or 0.25
        self.n = max(2, int(round(args.length / self.ds)) + 1)

    def set_radius(self, r: float):
        self._radius = max(1.0, r)

    def send(self, label: str, kappa: float):
        """현재 명령의 경로를 한 번 만들어 UDP 로 1회 전송."""
        pts, yaws = build_path(kappa, self.ds, self.n)
        v_plan = self.args.speed / 3.6
        payload = {
            "pred_xyz": pts,
            "pred_yaw_rad": yaws,
            "pred_v_mps": [round(v_plan, 3)] * self.n,
            "pred_curvature": [round(kappa, 6)] * self.n,
            "plan_dt_s": self.args.plan_dt,
            "t0_us": int(time.time() * 1e6),
            "inference_time_s": 0.0,
            "sample_id": self._seq,
        }
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
            self._seq += 1
            self._last = label
            return True
        except OSError as e:
            print(f"전송 실패: {e}")
            return False

    def status(self) -> str:
        return (f"마지막 전송={self._last}  기본반경={self._radius:.1f}m  "
                f"보낸패킷={self._seq}  -> {self.addr[0]}:{self.addr[1]}  "
                f"({self.n}점 x {self.args.plan_dt}s)")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


HELP = """\
명령 (엔터 = 즉시 1회 전송):
  직진 / s / straight        직진 경로 1회
  좌 / 좌회전 / l / left       좌회전 경로 1회
  우 / 우회전 / r / right      우회전 경로 1회
  <명령> <숫자>              그 회전의 반경(m). 예) "l 8", "우 12"
  radius <숫자> / R<숫자>     기본 회전반경 변경
  st / status               현재 상태
  ? / help                  도움말
  q / quit / exit           종료
* 한 번 보내면 그 경로(기본 15m) 끝까지 가고 멈춘다. 계속 가려면 다시 전송.
"""

STRAIGHT = {"직진", "s", "straight", "j", "w", "8", "ㅈ"}
LEFT = {"좌", "좌회전", "l", "left", "a", "4", "ㅗ"}
RIGHT = {"우", "우회전", "r", "right", "d", "6", "ㅜ"}
QUIT = {"q", "quit", "exit", "종료"}


def main():
    args = parse_args()
    cmd = Commander(args)
    print(f"경로 커맨더 시작 -> {args.host}:{args.port} "
          f"(반경 기본 {args.radius:.1f}m, {cmd.n}점 x {args.plan_dt}s, one-shot)")
    print("tesla_camera.py 를 --alpamayo-control 로 띄우고 창에서 'O' 키로 "
          "Alpamayo 주행을 켜세요.")
    print(HELP)
    try:
        while True:
            try:
                line = input("경로> ").strip()
            except EOFError:
                break
            if not line:
                print(cmd.status())
                continue
            toks = line.split()
            head = toks[0].lower()
            # 반경 인자 (두번째 토큰 숫자) 파싱
            arg_r = None
            if len(toks) >= 2:
                try:
                    arg_r = float(toks[1])
                except ValueError:
                    arg_r = None

            if head in QUIT:
                break
            elif head in ("?", "help", "h", "도움말"):
                print(HELP)
            elif head in ("st", "status", "상태"):
                print(cmd.status())
            elif head in ("radius",) or (head.startswith("r") and head[1:].replace(".", "", 1).isdigit()):
                r = arg_r if head == "radius" else float(head[1:])
                if r is None:
                    print("사용법: radius <숫자>")
                else:
                    cmd.set_radius(r)
                    print(f"기본 반경 = {cmd._radius:.1f}m")
            elif head in STRAIGHT:
                cmd.send("직진", 0.0)
                print("→ 직진 경로 전송")
            elif head in LEFT:
                if arg_r:
                    cmd.set_radius(arg_r)
                cmd.send("좌회전", 1.0 / cmd._radius)
                print(f"→ 좌회전 경로 전송 (반경 {cmd._radius:.1f}m)")
            elif head in RIGHT:
                if arg_r:
                    cmd.set_radius(arg_r)
                cmd.send("우회전", -1.0 / cmd._radius)
                print(f"→ 우회전 경로 전송 (반경 {cmd._radius:.1f}m)")
            else:
                print(f"알 수 없는 명령: {line!r}   ('?' 로 도움말)")
    except KeyboardInterrupt:
        pass
    finally:
        cmd.close()
        print("\n커맨더 종료.")


if __name__ == "__main__":
    main()
