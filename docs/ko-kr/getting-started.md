# 시작하기

[English](../getting-started.md) | 한국어

이 문서는 HNNX를 설치하고 첫 ONNX 모델을 확인·편집하는 과정을 안내합니다.
운영체제별 상세 내용은 [설치](installation.md) 문서를 참고합니다.

## 1. HNNX 설치

사용 환경에 맞는 방법을 선택합니다.

- **VS Code:** [최신 릴리스](https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx/releases/tag/v0.1.19)에서
  `hnnx-0.1.19.vsix`를 설치합니다.
- **Windows x64:** `HNNX-0.1.19-x64-setup.exe`를 설치합니다.
- **Linux x64:** AppImage 또는 Debian 패키지를 사용합니다.
- **Apple Silicon macOS:** 저장소를 clone한 뒤
  `./scripts/build-macos-local.sh`를 실행합니다. Apple Developer ID 서명과
  공증을 지원하기 전까지는 로컬 빌드를 권장합니다.
- **개발 환경:** 저장소에서 `npm install`과 `npm start`를 실행합니다.

## 2. 모델 열기

데스크톱 앱에서는 `.onnx` 파일을 열거나 창으로 드래그합니다. ONNX,
외부 `.data`, AIMET encodings 파일을 한 번에 드래그할 수도 있습니다.

VS Code에서는 Explorer의 `.onnx` 파일을 선택합니다. HNNX가 기본 사용자 지정
편집기로 열립니다. 다른 편집기가 선택되어 있다면 **Open With…**를 사용합니다.

간단한 예제는 다음 두 파일을 함께 엽니다.

- `examples/hnnx-mixed-precision.onnx`
- `examples/hnnx-mixed-precision.encodings`

## 3. 그래프 탐색

- 스크롤 또는 pinch 동작으로 확대·축소하고 빈 공간을 드래그하여 이동합니다.
- Find에서 노드나 그래프 엔드포인트를 검색합니다.
- 노드, 텐서, 배지 또는 엔드포인트를 클릭하여 오른쪽 사이드바에서 확인합니다.
- 사이드바 경계를 드래그하여 긴 텐서 이름을 표시합니다.
- 좌하단 정보 버튼에서 그래프와 encoding 통계를 확인합니다.
- **View > Theme**에서 Auto, Light 또는 Dark를 선택합니다.

주요 도구 모음은 다음 순서를 유지합니다.

```text
SAVE AS → INFER SHAPES → RE-LAYOUT → EDIT → ENC
```

`ENC`는 AIMET encodings 파일을 첨부한 경우에 표시됩니다.

## 4. AIMET encodings 연결

HNNX는 같은 폴더의 일반적인 파일명을 인식합니다.

```text
model.encodings
model.onnx.encodings
model.encodings.json
```

VS Code 확장은 기본적으로 위 파일을 자동으로 불러옵니다. **File > Load
AIMET Encodings…**, **Reload Encodings**, **Detach Encodings**에서 첨부 파일을
관리합니다. `ENC` 버튼은 ONNX를 다시 열지 않고 encoding 시각화만 켜거나 끕니다.

badge와 edge label의 의미는 [AIMET Encodings](aimet-encodings.md)를 참고합니다.

## 5. 편집 및 형상 추론 설정

편집 기능은 선택 사항입니다. 다음 방법으로 권장 Python 환경을 생성합니다.

- 데스크톱: **HNNX/View > GraphSurgeon Settings… > Create Recommended Environment**
- VS Code: Command Palette에서 **HNNX: Create GraphSurgeon Environment**

VS Code 원격 세션에서는 Mac이 아니라 원격 extension host에 환경을
생성합니다.

`E`를 누르거나 `EDIT · BETA`를 선택합니다. 연결선을 선택하면 정확한 소스와
대상을 확인할 수 있습니다. 예제 모델의 복사본을 편집하고 `INFER
SHAPES`를 실행한 뒤 `SAVE AS`로 새로운 ONNX를 저장합니다.

중요한 모델을 수정하기 전에 [그래프 편집](graph-editing.md)을 확인합니다.

## 다음 문서

- [설치 및 업데이트](installation.md)
- [그래프 편집](graph-editing.md)
- [AIMET 정밀도 이해](aimet-encodings.md)
- [문제 해결](troubleshooting.md)
