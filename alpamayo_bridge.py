#!/usr/bin/env python
"""CARLA -> Alpamayo 1.5 producer ("sensor container" 역할).

tesla_camera.py 의 카메라/차량 상태를 받아, Thor의 planner_live_service 가
polling 하는 `/latest` HTTP 엔드포인트로 Alpamayo 입력 NPZ 샘플을 서빙한다.

Alpamayo 입력 계약 (planner_live/sample_contract.py 기준, 4카메라 배치):
    image_frames        : (4, 4, 3, 320, 576) uint8   [Cam, T, C, H, W] planar CHW RGB
    camera_indices      : (4,)  int32   = (0, 1, 2, 6)
    camera_order        : (4,)  str     = (left, front, right, front_tele)
    ego_history_xyz     : (1, 1, 16, 3)    float32   t0-상대 ego-local 위치
    ego_history_rot     : (1, 1, 16, 3, 3) float32   t0-상대 ego-local 회전행렬
    relative_timestamps : (4, 4) float32  프레임별 t0 기준 상대시간(초)
    absolute_timestamps : (4, 4) int64    프레임별 unix us
    t0_us               : (1,) int64
    fixed_delta_seconds : (1,) float32 (>0)
    clip_id             : (1,) str
    camera_order        : 위와 동일

T축(프레임 4장)은 oldest->newest, 마지막이 t0.
ego history 16스텝도 oldest->newest, 마지막(t0)이 위치 (0,0,0)/회전 identity.

회전행렬 convention: yaw = arctan2(R[1,0], R[0,0])  ->  Rz(yaw).

이 모듈은 carla 에 의존하지 않는다. tesla_camera.py 가 carla 좌표/벡터를
뽑아 push_camera()/tick() 으로 넘겨준다.
"""
from __future__ import annotations

import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


# --- Alpamayo 카메라 매핑 (tesla_camera 이름 -> (camera_id, semantic)) ---
# planner_live PLANNER_CAMERA_IDS=(0,1,2,6), ORDER=(left,front,right,front_tele)
CAMERA_MAP = {
    "left":  (0, "left"),
    "front": (1, "front"),
    "right": (2, "right"),
    "tele":  (6, "front_tele"),   # CARLA 'tele'(망원) == Alpamayo front_tele
}
# image_frames / camera_indices 에 들어갈 정렬 순서 (canonical)
CAMERA_ORDER = ("left", "front", "right", "front_tele")
CAMERA_INDICES = (0, 1, 2, 6)
# Alpamayo 가 기대하는 tesla_camera 카메라 이름들
REQUIRED_CAMERAS = ("left", "front", "right", "tele")

IMAGE_H = 320
IMAGE_W = 576
NUM_FRAMES = 4      # T: t0-300/-200/-100/0 ms
NUM_EGO_HISTORY = 16

# ego-local 좌표/회전 부호.  CARLA 는 left-handed(x앞 y오른쪽 z위, yaw 시계방향),
# Alpamayo ego-local 은 x앞 / y왼쪽 / z위 (오른손계) 로 가정한다.
# -> 오른쪽 성분을 뒤집어 왼쪽 양수로, yaw 부호도 뒤집어 반시계 양수로.
# 실주행 결과(대시보드)와 비교해 history 가 차량 뒤로 깔리는지 확인 후 조정한다.
Y_SIGN = -1.0      # CARLA right-component -> Alpamayo y (left positive)
YAW_SIGN = -1.0    # CARLA yaw(CW) -> Alpamayo yaw(CCW)


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


class AlpamayoBridge:
    """카메라 프레임 + ego 포즈를 모아 /latest NPZ 로 서빙하는 producer."""

    def __init__(self, host="0.0.0.0", port=18080, clip_id="carla_live",
                 fixed_delta_seconds=0.1):
        self.host = host
        self.port = port
        self.clip_id = str(clip_id)
        self.fixed_delta_seconds = float(fixed_delta_seconds)

        self._lock = threading.Lock()
        # 카메라별 최신 HWC RGB uint8 프레임 (push_camera 로 갱신)
        self._latest_frame: dict[str, np.ndarray] = {}
        # 10Hz tick 으로 쌓는 기록
        self._img_records: deque = deque(maxlen=NUM_FRAMES)      # (t_us, {name: CHW})
        self._ego_records: deque = deque(maxlen=NUM_EGO_HISTORY)  # (t_us, pose)
        self._seq = 0

        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- tesla_camera 가 호출하는 입력 API ----------------------------------

    def push_camera(self, name: str, rgb_hwc: np.ndarray) -> None:
        """카메라 콜백에서 최신 RGB(HWC, uint8) 프레임을 전달."""
        if name not in CAMERA_MAP:
            return
        arr = np.ascontiguousarray(rgb_hwc, dtype=np.uint8)
        with self._lock:
            self._latest_frame[name] = arr

    def tick(self, pose) -> bool:
        """10Hz 로 호출. 현재 4개 카메라 최신 프레임 + ego 포즈를 한 스텝으로 기록.

        pose = (loc(x,y,z), fwd(x,y,z), right(x,y,z), yaw_rad)  -- CARLA 기준.
        모든 모델 카메라 프레임이 준비됐을 때만 기록하고 True 를 반환.
        """
        t_us = int(time.time() * 1e6)
        with self._lock:
            frames = {}
            for name in REQUIRED_CAMERAS:
                f = self._latest_frame.get(name)
                if f is None or f.shape[:2] != (IMAGE_H, IMAGE_W):
                    return False
                # HWC -> CHW (planar), Alpamayo 계약은 [C,H,W]
                frames[name] = np.ascontiguousarray(
                    np.transpose(f, (2, 0, 1)), dtype=np.uint8)
            self._img_records.append((t_us, frames))
            self._ego_records.append((t_us, pose))
            self._seq += 1
            return True

    # ---- 샘플 빌드 ----------------------------------------------------------

    def _build_sample(self):
        """버퍼가 충분하면 (npz_bytes, seq, t0_us) 반환, 아니면 None."""
        with self._lock:
            if (len(self._img_records) < NUM_FRAMES
                    or len(self._ego_records) < NUM_EGO_HISTORY):
                return None
            img_records = list(self._img_records)      # oldest..newest (len 4)
            ego_records = list(self._ego_records)      # oldest..newest (len 16)
            seq = self._seq

        t0_us = int(img_records[-1][0])

        # image_frames: [Cam, T, C, H, W]
        cams = []
        for name in CAMERA_ORDER:
            # CAMERA_ORDER 는 semantic; tesla 카메라 이름으로 역매핑
            tesla_name = _semantic_to_tesla(name)
            frames_t = [rec[1][tesla_name] for rec in img_records]  # T x CHW
            cams.append(np.stack(frames_t, axis=0))
        image_frames = np.stack(cams, axis=0).astype(np.uint8)

        # timestamps: 프레임 시각 (모든 카메라 동일하게 기록됨)
        frame_times = np.array([rec[0] for rec in img_records], dtype=np.int64)  # (T,)
        rel = ((frame_times - t0_us).astype(np.float32) / 1e6)                   # (T,)
        relative_timestamps = np.tile(rel, (len(CAMERA_ORDER), 1)).astype(np.float32)
        absolute_timestamps = np.tile(frame_times, (len(CAMERA_ORDER), 1)).astype(np.int64)

        # ego history (t0-상대 ego-local)
        ego_xyz, ego_rot = _build_ego_history(ego_records)

        sample = {
            "image_frames": image_frames,
            "camera_indices": np.array(CAMERA_INDICES, dtype=np.int32),
            "ego_history_xyz": ego_xyz.reshape(1, 1, NUM_EGO_HISTORY, 3).astype(np.float32),
            "ego_history_rot": ego_rot.reshape(1, 1, NUM_EGO_HISTORY, 3, 3).astype(np.float32),
            "relative_timestamps": relative_timestamps,
            "absolute_timestamps": absolute_timestamps,
            "t0_us": np.array([t0_us], dtype=np.int64),
            "fixed_delta_seconds": np.array([self.fixed_delta_seconds], dtype=np.float32),
            "clip_id": np.array([self.clip_id]),
            "camera_order": np.array(list(CAMERA_ORDER)),
        }

        buf = io.BytesIO()
        np.savez(buf, **sample)
        return buf.getvalue(), seq, t0_us

    def status(self) -> dict:
        with self._lock:
            n_img = len(self._img_records)
            n_ego = len(self._ego_records)
            last_t0 = int(self._img_records[-1][0]) if self._img_records else 0
            seq = self._seq
        age_s = (time.time() * 1e6 - last_t0) / 1e6 if last_t0 else None
        return {
            "ready": n_img >= NUM_FRAMES and n_ego >= NUM_EGO_HISTORY,
            "img_records": n_img,
            "ego_records": n_ego,
            "sequence": seq,
            "last_t0_us": last_t0,
            "sample_age_s": age_s,
            "clip_id": self.clip_id,
        }

    # ---- HTTP 서버 ----------------------------------------------------------

    def start(self) -> None:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 콘솔 조용히
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0].rstrip("/")
                if path in ("/latest", ""):
                    built = bridge._build_sample()
                    if built is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    payload, seq, t0_us = built
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("X-Sample-Sequence", str(seq))
                    self.send_header("X-T0-US", str(t0_us))
                    self.send_header("X-Clip-ID", bridge.clip_id)
                    self.end_headers()
                    self.wfile.write(payload)
                elif path in ("/healthz", "/status"):
                    body = json.dumps(bridge.status()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"[alpamayo] /latest 서버 시작: http://{self.host}:{self.port}/latest "
              f"(clip_id={self.clip_id})")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            print("[alpamayo] /latest 서버 종료")


def _semantic_to_tesla(semantic: str) -> str:
    for tesla_name, (_id, sem) in CAMERA_MAP.items():
        if sem == semantic:
            return tesla_name
    raise KeyError(semantic)


def _build_ego_history(ego_records):
    """ego_records(oldest..newest, len 16) -> (xyz[16,3], rot[16,3,3]) t0-local."""
    (cx, cy, cz), (fx, fy, fz), (rx, ry, rz), yaw0 = ego_records[-1][1]
    xyz = np.zeros((NUM_EGO_HISTORY, 3), dtype=np.float32)
    rot = np.zeros((NUM_EGO_HISTORY, 3, 3), dtype=np.float32)
    for i, (_t, pose) in enumerate(ego_records):
        (px, py, pz), _fwd, _right, yaw = pose
        dx, dy, dz = px - cx, py - cy, pz - cz
        fwd_comp = dx * fx + dy * fy + dz * fz       # 전방 성분
        right_comp = dx * rx + dy * ry + dz * rz     # 우측 성분
        xyz[i, 0] = fwd_comp                          # x: forward
        xyz[i, 1] = Y_SIGN * right_comp               # y: left(+)
        xyz[i, 2] = dz                                # z: up
        rot[i] = _rz(YAW_SIGN * (yaw - yaw0))
    return xyz, rot
