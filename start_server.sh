#!/usr/bin/env bash
# CARLA 시뮬레이터 서버 실행 헬퍼
# 사용법:
#   ./start_server.sh                    # NVIDIA GPU로 실행 (창 띄움)
#   ./start_server.sh -quality-level=Low # 저사양 모드 (8GB VRAM 권장)
#   ./start_server.sh -RenderOffScreen   # 헤드리스(화면 없이) 실행
#   USE_NVIDIA=0 ./start_server.sh       # NVIDIA 강제 끄기(내장 GPU로 실행)
#
# 이 노트북은 하이브리드 GPU(AMD 내장 + RTX 4060, prime-select=on-demand)라
# 기본값으로는 내장 GPU에 렌더링이 잡힙니다. 아래 환경변수로 NVIDIA(RTX 4060)에
# 오프로드합니다. CARLA는 Vulkan으로 렌더링하므로 NVIDIA optimus Vulkan 레이어를 사용.

if [ "${USE_NVIDIA:-1}" = "1" ]; then
  export __NV_PRIME_RENDER_OFFLOAD=1        # NVIDIA PRIME 오프로드 활성화
  export __VK_LAYER_NV_optimus=NVIDIA_only  # Vulkan 디바이스를 NVIDIA만 노출
  export __GLX_VENDOR_LIBRARY_NAME=nvidia   # (GLX 폴백용)
  echo "[start_server] NVIDIA(RTX 4060)로 렌더링 오프로드합니다. (USE_NVIDIA=0 으로 끌 수 있음)"
else
  echo "[start_server] NVIDIA 오프로드 비활성 — 내장 GPU로 실행합니다."
fi

cd "$(dirname "$0")/CARLA_0.9.16" || { echo "CARLA_0.9.16 폴더가 없습니다. 압축 해제를 먼저 하세요."; exit 1; }
echo "CARLA 서버 시작... (종료: Ctrl+C)"
./CarlaUE4.sh "$@"
