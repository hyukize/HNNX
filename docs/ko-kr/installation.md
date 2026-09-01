# 설치

[English](../installation.md) | 한국어

HNNX는 데스크톱 앱, VS Code 확장, 소스 코드 형태로 제공합니다. 배포된
패키지는 [릴리스 페이지](https://github.com/hyukize/HNNX/releases)에서
확인합니다.

## Apple Silicon macOS

HNNX는 아직 Apple Developer ID 서명과 Apple 공증을 사용하지 않으므로 macOS에서는
로컬 빌드를 권장합니다.

필요한 환경은 다음과 같습니다.

- Apple Silicon Mac (`arm64`)
- Git
- Node.js와 npm
- `codesign`을 제공하는 Xcode Command Line Tools

```bash
git clone https://github.com/hyukize/HNNX.git
cd hnnx
./scripts/build-macos-local.sh
```

스크립트는 고정된 npm 의존성을 설치하고 HNNX를 빌드합니다. 전체 app bundle에
ad-hoc 서명을 적용하고 검증한 뒤 다음 파일을 생성합니다.

```text
dist/HNNX-<version>-arm64.dmg
```

DMG를 열어 HNNX를 Applications로 복사합니다. `node_modules`가 준비된 반복
빌드에서는 다음 명령을 사용합니다.

```bash
./scripts/build-macos-local.sh --skip-install
```

ad-hoc 서명은 Developer ID 서명이나 공증을 대체하지 않습니다. Gatekeeper를
시스템 전체에서 비활성화하지 않습니다. 릴리스 DMG는 제한된 테스트 용도로
제공하며 팀 설치에는 로컬 빌드를 권장합니다.

## Windows x64

1. 릴리스 페이지에서 `HNNX-<version>-x64-setup.exe`를 받습니다.
2. 설치 프로그램을 실행합니다.
3. SmartScreen 경고가 표시되면 파일을 공식 HNNX 릴리스에서 받았는지 확인한
   후 진행합니다.

Windows 설치 프로그램은 code signing을 적용하지 않았습니다.

## Linux x64

AppImage는 다음과 같이 실행합니다.

```bash
chmod +x HNNX-*-x64.AppImage
./HNNX-*-x64.AppImage
```

Debian 또는 Ubuntu에서는 다음과 같이 설치합니다.

```bash
sudo apt install ./HNNX-*-x64.deb
```

데스크톱 앱은 그래픽 세션이 필요합니다. CLI 전용 또는 원격 환경에서는
로컬 VS Code를 해당 환경에 연결하고 HNNX 확장을 사용합니다.

## VS Code 확장

다음 환경이 필요합니다.

- VS Code 1.90 이상
- ONNX 파일이 있는 로컬 또는 원격 workspace

1. `hnnx-<version>.vsix`를 받습니다.
2. VS Code에서 Extensions를 엽니다.
3. `…` > **Install from VSIX…**를 선택합니다.
4. 파일을 선택하고 필요하면 VS Code를 다시 불러옵니다.

명령행에서도 설치할 수 있습니다.

```bash
code --install-extension hnnx-*.vsix --force
```

확장은 workspace extension host에서 실행됩니다. Remote SSH, Dev Container,
Kubernetes 환경에서는 Python 설정과 모델 접근도 원격 환경에서 수행합니다.

로컬 VSIX를 업데이트할 때는 새 파일을 `--force`로 설치하거나 기존 HNNX를
삭제한 후 새로운 VSIX를 설치합니다.

## 소스 코드 실행

Git, Node.js, npm이 필요합니다.

```bash
git clone https://github.com/hyukize/HNNX.git
cd hnnx
npm install
npm start
```

`npm install`은 JavaScript 의존성만 설치합니다. 시스템 Python을 변경하거나
GraphSurgeon을 설치하지 않습니다.

## 선택 사항인 Python backend

ONNX와 AIMET encodings를 보는 기능에는 Python backend가 필요하지 않습니다.
그래프 편집, Save As 검증, 형상 추론에는 Python 3, ONNX, NVIDIA ONNX
GraphSurgeon이 필요합니다.

내장된 환경 생성 명령을 사용하거나 [그래프 편집](graph-editing.md#graphsurgeon-환경)의
수동 설정을 따릅니다.
