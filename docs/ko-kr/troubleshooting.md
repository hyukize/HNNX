# 문제 해결

[English](../troubleshooting.md) | 한국어

## macOS 앱 손상 또는 확인 불가 경고

HNNX는 Apple Developer ID 서명이나 공증을 사용하지 않습니다. 따라서 app bundle에
ad-hoc 서명이 있어도 인터넷에서 받은 DMG를 Gatekeeper가 차단할 수 있습니다.

Apple Silicon Mac에서 HNNX를 로컬로 빌드하는 방법을 권장합니다.

```bash
git clone https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx.git
cd hnnx
./scripts/build-macos-local.sh
```

생성된 DMG를 열어 HNNX를 Applications로 복사합니다. Gatekeeper를 시스템 전체에서
비활성화하지 않습니다.

## `externally-managed-environment` 오류

Homebrew와 많은 Linux 배포판은 PEP 668에 따라 system Python을 보호합니다.
HNNX를 설치하기 위해 `--break-system-packages`를 사용하지 않습니다.

HNNX의 **Create Recommended Environment**를 사용하거나 전용 venv를 생성합니다.

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

## ONNX GraphSurgeon 검색 실패

1. VS Code에서 **HNNX: Create GraphSurgeon Environment**를 실행하거나 데스크톱
   GraphSurgeon Settings에서 **Create Recommended Environment**를 선택합니다.
2. 기존 환경이 있다면 **HNNX: Configure GraphSurgeon Python**에서 Python
   executable을 선택합니다.
3. 다음 명령으로 직접 검증합니다.

```bash
~/.hnnx/venv/bin/python -c "import onnx, onnx_graphsurgeon"
```

Windows에서는 `~/.hnnx/venv/Scripts/python.exe`를 사용합니다. VS Code remote
session에서는 local Mac이 아니라 remote 환경에서 확인합니다.

## Remote 또는 Kubernetes에서 VSIX 미동작

HNNX는 workspace extension입니다. Remote extension host에 설치하거나 활성화한
뒤 연결된 상태에서 **HNNX: Create GraphSurgeon Environment**를 실행합니다. Model,
external data, encodings, Python 경로를 모두 해당 host에서 접근할 수 있어야 합니다.

## `ENC` 버튼 미표시

Encodings 파일이 attachment된 경우에만 버튼을 표시합니다.

- **File > Load AIMET Encodings…**에서 직접 불러옵니다.
- VS Code의 `hnnx.autoLoadEncodings`와 같은 폴더의 파일명을 확인합니다.
- 파일을 외부에서 수정했다면 **Reload Encodings**를 실행합니다.
- 현재 VSIX를 설치했는지 확인하고 오래된 확장은 삭제하거나 업데이트합니다.

## Encodings tensor mismatch

Encoding export에는 ONNX export 또는 편집 중 삭제·변경된 이름이 남을 수 있습니다.
Model statistics에서 mismatch entry를 확인합니다. HNNX는 graph가 변경되어도
encodings를 다시 작성하지 않으므로 현재 ONNX에 맞게 생성한 파일을 불러옵니다.

## External-data ONNX 열기 또는 저장 실패

ONNX와 `.data` sidecar를 같은 폴더에 둡니다. HNNX는 안전을 위해 model 폴더
밖의 external data를 읽지 않습니다. 상대 참조를 유지할 수 있도록 편집한 모델도
원본 data 파일과 같은 폴더에 저장합니다.

## Infer Shapes 실패

일반적인 원인은 다음과 같습니다.

- 필수 input 또는 Graph Output이 끊어져 있습니다.
- 연결이 cycle을 생성합니다.
- Tensor element type이 호환되지 않습니다.
- Operator attribute가 Opset 17 schema에 맞지 않습니다.
- Custom operator에 ONNX shape function이 없습니다.

편집 중 일부 invalid state는 허용하지만 필수 연결을 복구하기 전에는 Infer Shapes와
Save As를 실행할 수 없습니다. Custom operator가 유효하더라도 shape는 일부만
추론될 수 있습니다.

TopK의 두 output은 의미가 다릅니다. `values`와 integer `indices`를 각각 호환되는
destination에 연결합니다.

## 대형 모델 성능 저하

최초 로딩은 전체 graph를 parsing하고 layout과 SVG를 생성합니다. 로드 이후에는
다음 방법을 사용합니다.

- VS Code에서 ONNX tab을 고정하여 전체 reload를 피합니다.
- Find와 확장된 zoom 범위를 사용합니다.
- Incremental edit를 우선 사용하고 필요할 때만 Re-layout을 실행합니다.
- 사용하지 않는 model tab을 닫고 Electron, VS Code extension host 또는
  WindowServer의 CPU와 memory 사용량을 확인합니다.

## Save As 후 원본 파일 유지

Save As는 항상 플랫폼 파일 선택창을 엽니다. 다른 이름을 선택하면 복사본을
생성합니다. 원본을 덮어쓰려면 원본 경로를 직접 선택하고 플랫폼의 overwrite
확인을 진행합니다.

HNNX가 View mode에 있더라도 저장 파일에는 현재 편집 session이 포함됩니다.
