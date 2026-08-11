# Installation

English | [한국어](ko-kr/installation.md)

HNNX is available as a desktop application, a VS Code extension, and source.
Published packages are listed on the
[releases page](https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx/releases).

## Apple Silicon macOS

The recommended macOS installation is a local build because HNNX does not yet
use an Apple Developer ID certificate or Apple notarization.

Requirements:

- Apple Silicon Mac (`arm64`)
- Git
- Node.js and npm
- Xcode Command Line Tools for `codesign`

```bash
git clone https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx.git
cd hnnx
./scripts/build-macos-local.sh
```

The script installs locked npm dependencies, builds HNNX, ad-hoc signs and
verifies the complete app bundle, and creates:

```text
dist/HNNX-<version>-arm64.dmg
```

Open the DMG and copy HNNX to Applications. For a repeat build with an existing
`node_modules` directory, use:

```bash
./scripts/build-macos-local.sh --skip-install
```

An ad-hoc signature is not equivalent to Developer ID signing or notarization.
Do not disable Gatekeeper globally. The release DMG is provided for controlled
testing, while a local build remains the recommended team installation method.

## Windows x64

1. Download `HNNX-<version>-x64-setup.exe` from the release page.
2. Run the installer.
3. If SmartScreen appears, verify that the file came from the expected HNNX
   release before continuing.

The Windows installer is not code-signed.

## Linux x64

For AppImage:

```bash
chmod +x HNNX-*-x64.AppImage
./HNNX-*-x64.AppImage
```

For Debian or Ubuntu:

```bash
sudo apt install ./HNNX-*-x64.deb
```

The desktop application requires a graphical session. For a CLI-only or remote
environment, use the VS Code extension from a local VS Code window attached to
that environment.

## VS Code extension

Requirements:

- VS Code 1.90 or later
- A local or remote workspace containing the ONNX files

1. Download `hnnx-<version>.vsix`.
2. Open Extensions in VS Code.
3. Choose `…` > **Install from VSIX…**.
4. Select the downloaded file and reload VS Code if prompted.

Command-line installation is also supported:

```bash
code --install-extension hnnx-*.vsix --force
```

The extension runs on the workspace extension host. In Remote SSH, Dev
Container, and Kubernetes sessions, Python setup and model access therefore
occur in the remote environment.

To update a locally installed VSIX, install the newer file with `--force` or
uninstall HNNX and install the new VSIX.

## Run from source

Requirements:

- Git
- Node.js and npm

```bash
git clone https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx.git
cd hnnx
npm install
npm start
```

`npm install` installs JavaScript dependencies only. It does not modify the
system Python or install GraphSurgeon.

## Optional Python backend

Viewing ONNX and AIMET encodings does not require the Python backend. Graph
editing, Save As validation, and shape inference require Python 3, ONNX, and
NVIDIA ONNX GraphSurgeon.

Use the built-in environment command or follow the manual setup in
[Graph Editing](graph-editing.md#graphsurgeon-environment).
