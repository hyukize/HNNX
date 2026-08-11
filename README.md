# HNNX

HNNX is an ONNX graph workbench for model inspection, AIMET quantization
analysis, and NVIDIA ONNX GraphSurgeon-backed editing.

![HNNX icon](publish/pampam-hnnx-icon.png)

## Highlights

- Visualize ONNX models together with AIMET encodings
- Optionally auto-load neighboring encodings, then toggle, reload, or detach them without reopening the ONNX model
- Drop an ONNX model, its external `.data` sidecar, and encodings together
- Trace activation precision through encoding-free graph operations
- Inspect mixed precision, QParams, parameter bit widths, and encoding mismatches
- See explicit Graph Input quantizers as `A8`/`A16` endpoint badges
- Edit graph connections, nodes, inputs, outputs, and common Opset 17 operators
- Undo, redo, reset, shape inference, validation, and Save As
- Build the native Apple Silicon macOS app locally, or use the distributed Windows/Linux packages and VS Code extension
- Open models from VS Code Remote and Kubernetes workspaces
- Select Auto, Light, or Dark appearance; Auto follows the operating system or VS Code

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

Apple Silicon macOS app (recommended local build):

```bash
./scripts/build-macos-local.sh
```

The script verifies that it is running on Apple Silicon macOS, installs the
locked npm dependencies with `npm ci`, builds HNNX, ad-hoc signs the complete
app bundle, verifies the signature, and creates
`dist/HNNX-0.1.18-arm64.dmg`. For repeat builds with an existing
`node_modules`, use `./scripts/build-macos-local.sh --skip-install`.

Windows x64 installer:

```bash
npm run build:windows-hnnx
```

Linux x64 AppImage and Debian package:

```bash
npm run build:linux-hnnx
```

VS Code extension:

```bash
npm run build:vscode-hnnx
```

Generated packages are written to:

- `dist/HNNX-0.1.18-arm64.dmg` (built locally on Apple Silicon macOS)
- `dist/HNNX-0.1.18-x64-setup.exe`
- `dist/HNNX-0.1.18-x64.AppImage`
- `dist/HNNX-0.1.18-x64.deb`
- `vscode-extension/hnnx-0.1.18.vsix`

## macOS distribution policy

HNNX does not currently distribute a Developer ID-signed and Apple-notarized
macOS binary as its primary installation method. macOS users should clone this
repository and run `./scripts/build-macos-local.sh`, then open the generated DMG
and copy HNNX to Applications. This avoids distributing a downloaded,
unnotarized application and keeps the macOS build under the user's control.

The local build is ad-hoc signed and verified, but ad-hoc signing is not a
substitute for Apple Developer ID signing or notarization. Do not disable
Gatekeeper globally. Graph editing and shape inference still require the
separate Python environment described above; the build script never installs
Python packages or modifies Homebrew Python.

## Attribution

HNNX is derived from [Netron](https://github.com/lutzroeder/netron) and retains
Netron's MIT license and copyright notice. HNNX-specific modifications are by
Jonghyuk Park. The canonical project repository is
[hyukize/HNNX](https://github.com/hyukize/HNNX); organization-hosted copies may
be maintained as internal distribution mirrors. Mirroring does not replace the
copyright and license notices in this repository. See [NOTICE](NOTICE) and
[LICENSE](LICENSE).
