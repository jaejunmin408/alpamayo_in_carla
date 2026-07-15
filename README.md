# CARLA + Alpamayo 자율주행 구동 환경

<img width="1288" height="742" alt="image" src="https://github.com/user-attachments/assets/3bcd5d2a-897a-4c1d-b2f4-cd064cdcb137" />
<img width="1049" height="881" alt="image" src="https://github.com/user-attachments/assets/98d2ceed-f5b8-436e-aa8c-79f8b92d4b58" />

NVIDIA **Alpamayo** 주행 모델을 **CARLA 0.9.16** 시뮬레이터 위에서 돌리기 위한 코드 모음입니다.

CARLA에 테슬라 차량 + 카메라/센서를 스폰해 **Alpamayo 입력(멀티 카메라 + ego history)** 을 만들어 서빙하고,
모델(Thor의 `planner_live_service`)이 뱉은 **미래 경로(trajectory)** 를 받아 **pure pursuit 제어**로 차를 몰게 합니다.

```
                 (1) /latest NPZ (HTTP :18080)
  ┌─────────────────────┐  ───────────────────────▶  ┌──────────────────────────┐
  │   CARLA (이 PC)      │                            │  Thor 서버 (별도 머신)    │
  │                      │                            │  planner_live_service    │
  │  tesla_camera.py     │                            │  = Alpamayo 1.5 모델      │
  │   ├ alpamayo_bridge  │◀───────────────────────    │                          │
  │   └ alpamayo_control │  (2) UDP text_json 경로     └──────────────────────────┘
  │       (WorldPathFollower)   (:5005)
  └─────────────────────┘
```

- **(1) 입력**: `alpamayo_bridge.py` 가 카메라 4장 + ego history 를 NPZ 로 만들어 `/latest` 로 서빙 → Thor가 polling
- **(2) 출력**: Thor가 추론한 ego-local 경로(`pred_xyz`)와 점별 속도(`pred_v_mps`)를 UDP(`5005`)로 전송 → `alpamayo_control.py` 가 받아 제어

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `tesla_camera.py` | **메인 실행 스크립트.** 테슬라 + 카메라(front/tele/left/right) + IMU/GNSS 스폰, pygame 창(카메라 격자 + 사이드캠 각도 조절), 자율주행 루프. 모든 키 조작이 여기 있음. |
| `alpamayo_bridge.py` | **CARLA → Alpamayo 입력 producer.** 카메라/ego pose 를 Alpamayo 계약 형식(`image_frames`, `ego_history_xyz/rot`, `t0_us` 등) NPZ 로 만들어 `/latest` HTTP(`:18080`)로 서빙. |
| `alpamayo_control.py` | **Alpamayo → CARLA 제어.** UDP(`:5005`)로 받은 경로를 world 좌표로 고정하고 `WorldPathFollower`(closed-loop pure pursuit)로 추종. 횡/종방향 제어 로직 본체. |
| `viz_server.py` | **디버그용 2D top-down 맵 viz.** 스폰 지점을 원점으로 차 위치/궤적/고정 경로를 브라우저(`http://localhost:8091`)에 실시간 렌더. |
| `path_commander.py` | **수동 경로 주입기(테스트용).** 모델 대신 "좌회전/우회전/직진" 명령을 ego-local 경로로 만들어 UDP(`5005`)로 쏨. 제어가 잘 먹는지 모델 없이 검증. |
| `test_connection.py` | CARLA 서버 연결 확인용 최소 스크립트. |
| `start_server.sh` | CARLA 서버 실행 헬퍼 (하이브리드 GPU에서 RTX 4060으로 렌더 오프로드). |
| `CARLA_0.9.16/` | CARLA 시뮬레이터 본체 (대용량, git 제외). |
| `venv/` | 파이썬 가상환경 (git 제외). |

---

## 실행 방법

### 1. CARLA 서버 켜기

```bash
./start_server.sh -RenderOffScreen -quality-level=Low
```

- `-RenderOffScreen` : 헤드리스(서버 화면 없이) 실행 — 카메라 렌더는 정상 동작
- `-quality-level=Low` : 저사양 모드 (8GB VRAM 권장)
- 하이브리드 GPU(AMD 내장 + RTX 4060)라 `start_server.sh` 가 NVIDIA로 렌더를 오프로드함 (`USE_NVIDIA=0` 으로 끌 수 있음)

### 2. 차량 스폰 + 제어 클라이언트 실행

```bash
./venv/bin/python tesla_camera.py --no-autopilot --alpamayo --alpamayo-control --viz --map-route
```

| 플래그 | 의미 |
|--------|------|
| `--no-autopilot` | CARLA 내장 autopilot 끄고 수동/제어 모드로 시작 |
| `--alpamayo` | `alpamayo_bridge` 켬 — `/latest` NPZ producer(`:18080`) 서빙 (Thor 입력) |
| `--alpamayo-control` | `AlpamayoControlReceiver` 켬 — UDP(`:5005`)로 모델 경로 수신, `O` 키로 주행 |
| `--viz` | 2D 맵 viz 서버 켬 (`http://localhost:8091`) |
| `--map-route` | 맵 차선 route 추종 모드 활성 — `G` 키로 주행 (모델 없이 차선만 따라가기) |

> Thor(모델 서버)는 별도 머신에서 `planner_live_service` 를 띄워 이 PC의 `/latest` 를 polling 하고,
> 결과 경로를 이 PC의 UDP `5005` 로 쏘도록 설정해야 합니다.

---

## 시뮬레이터 키 조작

pygame 카메라 창에 포커스를 둔 상태에서:

| 키 | 동작 |
|----|------|
| **O** | **Alpamayo 주행 토글.** 모델이 UDP로 준 경로를 pure pursuit로 추종 시작/중지 (`--alpamayo-control` 필요) |
| **T** | **경로 앵커 토글: 현재 pose ↔ t0 pose.** 받은 경로를 "지금" 위치가 아니라 **추론 입력 프레임(t0) 시점** 위치에 붙임 (아래 설명) |
| **V** | **종방향 목표속도 토글: 모델 속도 ↔ 고정 속도.** 모델 `pred_v_mps` 를 목표로 쓸지, 고정 `TARGET_SPEED_KMH` 로 쓸지 |
| **L** | **lookahead 방식 토글: 남은 경로 % 인덱스 ↔ 속도기반 거리.** 목표점을 남은 경로의 몇 % 지점으로 고를지(기본), 속도비례 거리 `Ld`로 고를지 |
| **G** | **맵 차선 route 주행 토글.** 누른 순간 앞 차선 경로를 생성해 고정 추종 (`--map-route` 필요, 모델 불필요) |
| **P** | CARLA 내장 autopilot 토글 |
| **← →** | 사이드 카메라 yaw(좌우 각도) 조절 (사이드캠 있을 때) |
| **↑ ↓** | 사이드 카메라 pitch(상하 각도) 조절 |
| **R** | 사이드 카메라 각도 리셋 (yaw 90°, pitch 0°) |
| **W / S / A / D** | 수동운전 — 전진 / 후진 / 좌·우 조향 (Space = 핸드브레이크) |
| **ESC / Q** | 종료 |

> **모드 배타**: `P`(autopilot) · `O`(Alpamayo) · `G`(map-route) 는 서로 배타적. 하나를 켜면 나머지는 자동으로 꺼집니다.
> `T`(앵커) · `V`(속도) 는 `O`/`G` 주행과 무관하게 언제든 토글하는 설정 스위치입니다.

HUD 예시: `PP steer=+0.12 thr=0.30 brk=0.00 cte=0.45m ld=6.3m v*=20kph(model) gi=8/64`

---

## 제어 로직

제어는 전부 `alpamayo_control.py` 의 `WorldPathFollower` 에서 이뤄집니다.
받은 ego-local 경로는 **world 좌표로 고정(anchor)** 한 뒤, 매 프레임 **실제 차 pose**로 추종하는 closed-loop 방식입니다.

### 횡방향 (steering) — pure pursuit

후륜축(rear axle) 기준 world frame에서, 내 차 위치를 정확히 아는 상태로 경로를 위치 기반으로 따라갑니다:

1. 후륜축에서 경로상 **가장 가까운 점**(`ci`, 앞으로만 탐색) 찾기
2. **목표점(goal) 선택** — 아래 두 방식 중 하나 (키 `L` 로 토글)
3. 목표점을 차체좌표로 변환 → `α` → `δ = atan2(2·L·sinα, dist)` → steer

dead-reckoning open-loop 가 아니라 매 프레임 실제 후륜축 (x, y, yaw)로 목표점을 다시 찾기 때문에
cross-track / heading 오차를 그대로 보정합니다.

**목표점 선택 방식** (`WorldPathFollower`, 키 `L` 로 전환):

- **① 남은 경로 % 인덱스 (기본, `use_index_lookahead=True`)**
  ```
  gi = ci + round(pp_index_pct · (n-1 - ci))     # n = 경로 점 개수
     = ci + round(0.5 · (n-1 - ci))              # 기본 pct = 0.5
  ```
  - 현재 최근접점(`ci`)부터 **경로 끝까지 중 `pp_index_pct` 지점**을 목표로 → 속도와 무관
  - 차가 진행할수록 목표가 항상 앞에 있고, 경로 끝에서 자연히 수렴
  - 최소 1점 앞(`ci+1`)·상한 `n-1`로 clamp → `pct=0/1` 이어도 안전
  - 목표 비율은 `pp_index_pct`(기본 0.5)로 조정

- **② 속도기반 거리 (fallback, `use_index_lookahead=False`)**
  ```
  Ld = clamp(ld_gain · v + ld_l0,  ld_min,  ld_max)
     = clamp(0.6 · v + 3.0,        4.0,     10.0)   [m]
  ```
  - 최근접점부터 `Ld` 이상 떨어진 첫 점을 목표로. 속도에 비례(하한 4m/상한 10m)
  - 예) 정지 시 4m, 20km/h(≈5.6m/s)에서 ≈ 6.3m

> HUD의 `ld=…m[pct|dist]` 는 실제 목표점까지의 유효 lookahead 거리와 현재 방식을 표시합니다.

### 종방향 (throttle / brake) — 모델 속도 목표 P 제어

- **목표속도 = 모델 `pred_v_mps`** (현재 위치=최근접점의 점별 속도). 기본 활성(`use_model_speed=True`)
- P 제어: `err = v_target − v_ego`
  - `err > +deadband` → `throttle = min(1, KP_THROTTLE · err)`
  - `err < −deadband` → `brake = min(1, KP_BRAKE · (−err))`
  - 게인 `KP_THROTTLE = KP_BRAKE = 0.5`, `SPEED_DEADBAND = 0.3 m/s`
- **fallback**: 경로에 속도가 없으면(예: `G` 맵 route) 고정 `TARGET_SPEED_KMH = 20` 사용
- **키 `V`** 로 언제든 모델속도 ↔ 고정속도 전환 (모델 속도가 튀거나 위험할 때 안전 스위치)

### 경로 앵커: inference time 전 ego 위치에 붙이기 (키 `T`)

모델이 경로를 만든 입력 프레임 시점(`t0`)과, 그 결과가 CARLA에 도착한 시점 사이에는
**추론 + 전송 지연**이 있고 그동안 차는 앞으로 이동합니다.
경로를 "지금" pose에 붙이면 그 지연만큼 경로가 앞으로 밀리는 공간 오차가 생깁니다.

- 그래서 경로 원점을 **`t0` 시점(=추론 입력 프레임을 찍은 순간)의 ego pose** 에 붙입니다.
- `plan["t0_us"]` 는 `alpamayo_bridge` 가 `time.time()` 월클럭으로 찍어 Thor가 그대로 돌려준 값이라
  이 프로세스의 시계와 동일 → 매 프레임 pose를 버퍼(`PoseHistory`)에 쌓아 두고 `t0_us` 시점 pose를 **선형보간**으로 조회
- 조회한 과거 pose에 경로를 붙이면, 차가 이미 지나온 앞부분은 자연스럽게 최근접점 탐색으로 건너뛰고
  현재 위치부터 추종하게 됩니다.
- **키 `T`** 로 `현재 pose` ↔ `t0 pose` 앵커를 토글해 비교 가능. `t0` 조회 실패(버퍼 밖) 시 현재 pose로 fallback.

---

## 참고

- 좌표계: ego-local 은 x 전방 / y 좌측(+) / z 위. CARLA world 는 좌수 좌표계이며 steer +1 = 우회전.
- 로그 파일(`*.log`), `venv/`, `CARLA_0.9.16/`, `__pycache__/` 는 `.gitignore` 처리됨.
