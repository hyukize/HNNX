# HNNX

HNNX is an ONNX graph workbench for model inspection, AIMET quantization
analysis, and NVIDIA ONNX GraphSurgeon-backed editing.

![HNNX icon](publish/pampam-hnnx-icon.png)

## Highlights

- Visualize ONNX models together with AIMET encodings
- Drop an ONNX model, its external `.data` sidecar, and encodings together
- Trace activation precision through encoding-free graph operations
- Inspect mixed precision, QParams, parameter bit widths, and encoding mismatches
- See explicit Graph Input quantizers as `Q:A8`/`Q:A16` endpoint badges
- Edit graph connections, nodes, inputs, outputs, and common Opset 17 operators
- Undo, redo, reset, shape inference, validation, and Save As
- Use the native Apple Silicon macOS app or VS Code extension
- Open models from VS Code Remote and Kubernetes workspaces

See [AIMET.md](AIMET.md) for the complete feature guide and editor behavior.
See [MANUAL_TEST.md](MANUAL_TEST.md) for the hands-on test workflow.

## Development

```bash
npm install
npm start
```

Graph editing and shape inference require a Python environment containing ONNX
and NVIDIA ONNX GraphSurgeon:

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

## Build

Apple Silicon macOS app:

```bash
npm run build:mac-hnnx
```

VS Code extension:

```bash
npm run build:vscode-hnnx
```

Generated packages are written to `dist/HNNX-0.1.6-arm64.dmg` and
`vscode-extension/hnnx-0.1.6.vsix`.

## Attribution

HNNX is derived from [Netron](https://github.com/lutzroeder/netron) and retains
Netron's MIT license and copyright notice. HNNX-specific modifications are by
Jonghyuk Park. See [NOTICE](NOTICE) and [LICENSE](LICENSE).
