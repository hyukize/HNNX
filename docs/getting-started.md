# Getting Started

English | [한국어](ko-kr/getting-started.md)

This guide takes you from installation to inspecting and editing a small ONNX
model. For platform-specific details, see [Installation](installation.md).

## 1. Install HNNX

Choose one of these paths:

- **VS Code:** install `hnnx-<version>.vsix` from the
  [releases page](https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx/releases).
- **Windows x64:** install `HNNX-<version>-x64-setup.exe`.
- **Linux x64:** use the AppImage or Debian package.
- **Apple Silicon macOS:** clone the repository and run
  `./scripts/build-macos-local.sh`. This is the recommended macOS path until
  Developer ID signing and notarization are available.
- **Development:** run `npm install` and `npm start` from the repository.

On first launch, HNNX opens a setup window for the color theme, GraphSurgeon
Python environment, AIMET encodings defaults, and optional shape inference on
open. The recommended profile keeps automatic shape inference off. Reopen the
same window later from **Settings…** (`Cmd+,` in the macOS app).

## 2. Open a model

In the desktop app, open or drag an `.onnx` file into the window. You can drag
the ONNX file, its external `.data` file, and an AIMET encodings file together.

In VS Code, select an `.onnx` file in Explorer. HNNX is the default custom
editor. Use **Open With…** if another ONNX editor is already selected.

For a compact example, open these files together:

- `examples/hnnx-mixed-precision.onnx`
- `examples/hnnx-mixed-precision.encodings`

## 3. Navigate the graph

- Scroll or pinch to zoom; drag empty graph space to pan.
- Use Find to locate a node or graph endpoint.
- Click a node, tensor, badge, or endpoint to inspect it in the right sidebar.
- Drag the sidebar boundary to make long tensor names visible.
- Use the lower-left information control for graph and encoding statistics.
- Choose **View > Theme** to select Auto, Light, or Dark.

The primary toolbar stays in this order:

```text
SAVE AS → INFER SHAPES → RE-LAYOUT → EDIT → ENC
```

`ENC` appears when an AIMET encodings file is attached.

## 4. Attach AIMET encodings

HNNX recognizes common neighboring filenames such as:

```text
model.encodings
model.onnx.encodings
model.encodings.json
```

The desktop app and VS Code extension auto-load these by default. Use **File > Load AIMET
Encodings…**, **Reload Encodings**, or **Detach Encodings** to manage the
attachment. The `ENC` button hides or shows encoding visualization without
unloading the ONNX model.

See [AIMET Encodings](aimet-encodings.md) for badge and edge-label semantics.

## 5. Enable editing and shape inference

Editing is optional. Create the recommended Python environment using one of
these commands:

- Desktop: **HNNX > Settings… > Create recommended environment**
- VS Code: **HNNX: Create GraphSurgeon Environment** from the Command Palette

In a VS Code remote session, the environment is created on the remote
extension host, not on your Mac.

**Run ONNX shape inference when a model opens** is available in Settings and
is disabled by default. If enabled, a missing Python environment or an
inference error produces a warning while the original model remains open.

Press `E` or choose `EDIT · BETA`. Select a connection to inspect its exact
source and destination. Try an edit on a copy of the example model, run
`INFER SHAPES`, and use `SAVE AS` to write a new ONNX file.

Continue with [Graph Editing](graph-editing.md) before modifying an important
model.

## Next steps

- [Install or update another package](installation.md)
- [Learn graph editing](graph-editing.md)
- [Understand AIMET precision](aimet-encodings.md)
- [Resolve common problems](troubleshooting.md)
