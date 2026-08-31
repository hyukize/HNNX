# 그래프 편집

[English](../graph-editing.md) | 한국어

HNNX는 NVIDIA ONNX GraphSurgeon 기반의 beta ONNX 편집기를 제공합니다. HNNX의
시각화 방식을 유지하면서 ONNX 메인 그래프를 편집합니다.

중요한 모델은 항상 복사본에서 먼저 편집합니다.

## GraphSurgeon 환경

편집, Save As 검증, 형상 추론에는 `onnx`와 `onnx_graphsurgeon`이 설치된
Python 3 환경이 필요합니다.

다음 메뉴에서 권장 환경을 자동으로 생성합니다.

- **macOS 데스크톱:** **HNNX > GraphSurgeon Settings… > Create Recommended Environment**
- **Windows/Linux 데스크톱:** **View > GraphSurgeon Settings… > Create Recommended Environment**
- **VS Code:** **HNNX: Create GraphSurgeon Environment**를 실행합니다.

권장 경로는 `~/.hnnx/venv`입니다. VS Code는 Remote SSH, Dev Container,
Kubernetes를 포함한 현재 extension host에 환경을 생성합니다.

macOS 또는 Linux에서 수동으로 설정하려면 다음 명령을 실행합니다.

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

Windows PowerShell에서는 다음 명령을 실행합니다.

```powershell
py -m venv $HOME\.hnnx\venv
$HOME\.hnnx\venv\Scripts\python.exe -m pip install onnx onnx_graphsurgeon `
  --extra-index-url https://pypi.ngc.nvidia.com
```

기존 interpreter를 사용하려면 GraphSurgeon Settings에서 선택하거나 VS Code에서
**HNNX: Configure GraphSurgeon Python**을 실행합니다. HNNX는 경로를 저장하기
전에 두 module을 import할 수 있는지 검증합니다.

자동 생성 명령은 전용 virtual environment 안에만 package를 설치합니다. PEP
668을 우회하거나 Homebrew/시스템 Python을 변경하지 않습니다.

## Edit 모드 시작 및 종료

- `E`를 누르거나 `EDIT · BETA`를 선택하여 Edit 모드로 들어갑니다.
- `V`를 눌러 View 모드로 돌아갑니다.
- View 모드로 돌아가도 현재 세션의 편집 내용은 유지합니다.
- `SAVE AS`는 항상 현재 세션 상태를 저장합니다.

연결된 AIMET encodings는 별도의 읽기 전용 첨부 파일입니다. Topology를 편집해도
encodings 파일의 텐서 이름이나 QParam은 변경하지 않습니다.

## 연결선 편집

연결선을 선택하면 정확한 `tensor → node.input` 관계를 표시합니다.

- **REPLACE:** 작업을 선택한 뒤 주황색 소스 출력 포트를 선택합니다.
- **DISCONNECT:** 선택한 대상 입력을 임시로 비웁니다.
- **새 분기:** 출력 포트를 클릭하거나 호환되는 입력으로 드래그합니다.
- **Fan-out:** 분기 목록에서 대상 연결을 선택합니다. 항목에 마우스를 올리면
  해당 선을 강조합니다.
- **취소:** `Q`를 누르거나 그래프 빈 공간을 우클릭합니다.

필수 입력도 임시로 끊을 수 있지만 `REQUIRED · MISSING` 상태와 빨간색으로
표시합니다. 선택 입력은 보라색으로 표시합니다. 모든 필수 입력과
`Graph Output`을 복구하기 전에는 Save As와 Infer Shapes를 실행할 수 없습니다.

## 노드 및 그래프 엔드포인트 편집

노드를 클릭하여 입력, 노드 이름, 출력 텐서 이름을 관리합니다. 검색 가능한
`+ ADD`에서 형식을 지정한 `Graph Input`, `Graph Output`, 자주 사용하는 Opset 17 연산자를
추가합니다.

`D`를 누르면 선택한 편집 가능한 연결, 엔드포인트 또는 안전하게 삭제할 수 있는
노드를 삭제합니다. 다른 노드나 그래프 출력에서 사용 중인 노드는 삭제할 수
없습니다. Initializer는 읽기 전용이며 HNNX는 임의 가중치 편집이나 학습 기능을
제공하지 않습니다.

노드와 `Graph Input`/`Graph Output` 엔드포인트를 드래그하여 화면에서 정리할 수 있습니다.
위치는 화면에만 적용하며 `RESET`은 자동 레이아웃으로 복구합니다.

## 검증, 레이아웃, 저장

- **INFER SHAPES**는 임시 모델에 아직 저장하지 않은 명령을 적용하고 ONNX checker와
  엄격한 shape inference를 실행한 뒤 type과 shape를 화면에 표시합니다.
- **RE-LAYOUT**은 전체 그래프를 다시 생성하고 배치합니다.
- **REFRESH VIEW**는 증분 미리보기가 남아 있는 경우에만 조용한 보조 기능으로
  활성화됩니다.
- **SAVE AS**는 GraphSurgeon 편집을 적용하고 위상 정렬과 검증을
  수행한 뒤 플랫폼 파일 선택창을 엽니다.
- **RESET**은 아직 저장하지 않은 전체 편집 세션을 폐기합니다. 실수를 방지하기
  위해 단축키를 제공하지 않습니다.

외부 텐서 데이터 모델은 상대 `.data` 참조를 유지할 수 있도록 원본 ONNX와
같은 폴더에 저장합니다.
형상 추론 시에는 작은 Constant와 shape/axes/sizes 텐서에 필요한 외부 구간만
제한적으로 읽습니다. 큰 가중치는 불러오지 않으며 임시로 읽은 값을 저장 모델에
포함하지 않습니다.

## 키보드 단축키

텍스트, 검색창, 편집 가능한 컨트롤에 입력하는 동안에는 단축키가 작동하지 않습니다.

| 단축키 | 동작 |
| --- | --- |
| `E` | Edit Beta로 전환합니다. |
| `V` | View 모드로 돌아갑니다. |
| `R` | 그래프를 다시 배치합니다. |
| `I` | 형상 추론을 실행합니다. |
| `Cmd+S` / `Ctrl+S` | Save As를 실행합니다. |
| `D` | 선택한 편집 가능 항목을 삭제하거나 연결을 끊습니다. |
| `Q` | 현재 편집 선택을 취소합니다. |
| `Cmd/Ctrl+Z` | 실행 취소합니다. |
| `Cmd/Ctrl+Shift+Z` | 다시 실행합니다. |

## 현재 제약사항

- 메인 그래프만 편집할 수 있으며 중첩 subgraph 편집은 지원하지 않습니다.
- 임의 initializer 값과 학습된 weight는 편집할 수 없습니다.
- 필수 입력이 비어 있는 모델은 저장하거나 형상 추론할 수 없습니다.
- ONNX shape function이 없는 custom operator는 shape가 일부만 추론될 수 있습니다.
- Topology를 편집해도 외부 AIMET encodings를 자동으로 갱신하지 않습니다.
- 대형 모델은 최초에 전체 parsing, 레이아웃, SVG 생성을 수행합니다.
