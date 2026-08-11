# Troubleshooting

## macOS says the app is damaged or cannot be verified

HNNX is not Apple Developer ID-signed or notarized. A downloaded DMG can
therefore be blocked by Gatekeeper even though the bundle is ad-hoc signed.

The recommended solution is to build HNNX locally on an Apple Silicon Mac:

```bash
git clone https://gitea.fde2.mrxrunway.ai/mrx-armstrong/hnnx.git
cd hnnx
./scripts/build-macos-local.sh
```

Open the generated DMG and copy HNNX to Applications. Do not disable Gatekeeper
globally.

## `externally-managed-environment`

Homebrew and many Linux distributions protect their system Python under PEP
668. Do not use `--break-system-packages` for HNNX.

Use **Create Recommended Environment** in HNNX or create a dedicated venv:

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

## HNNX cannot find ONNX GraphSurgeon

1. Run **HNNX: Create GraphSurgeon Environment** in VS Code, or open desktop
   GraphSurgeon Settings and choose **Create Recommended Environment**.
2. If an environment already exists, select its Python executable with
   **HNNX: Configure GraphSurgeon Python**.
3. Verify it manually:

```bash
~/.hnnx/venv/bin/python -c "import onnx, onnx_graphsurgeon"
```

On Windows, use `~/.hnnx/venv/Scripts/python.exe`. In a VS Code remote session,
run these checks in the remote environment rather than on the local Mac.

## The VSIX works locally but not over Remote or Kubernetes

HNNX is a workspace extension. Install or enable it on the remote extension
host, then run **HNNX: Create GraphSurgeon Environment** while connected. The
model files, external data, encodings, and configured Python path must all be
accessible from that host.

## The `ENC` button is missing

The button appears only after an encodings file is attached.

- Check **File > Load AIMET Encodings…**.
- In VS Code, verify `hnnx.autoLoadEncodings` and the neighboring filename.
- Use **Reload Encodings** after editing the file externally.
- Confirm that you installed the current VSIX rather than an older extension.

## Encodings are loaded but some tensors do not match

The encoding export may contain names removed or renamed during ONNX export or
editing. Open model statistics and inspect mismatch entries. HNNX does not
rewrite encodings when the graph changes; load an encoding file generated for
the current ONNX model.

## An external-data ONNX model does not open or save correctly

Keep the ONNX and its `.data` sidecar in the same directory. HNNX restricts
external-data resolution to the model directory for safety. Save edited models
beside the original data file so relative references remain valid.

## Infer Shapes fails

Common causes include:

- A required input or Graph Output is disconnected
- A connection creates a cycle
- Tensor element types are incompatible
- An operator's attributes do not satisfy its Opset 17 schema
- A custom operator has no registered ONNX shape function

The editor permits some invalid intermediate states, but Infer Shapes and Save
As remain blocked until required connections are restored. Custom operators
may be valid while still producing only partial inferred shapes.

TopK has two semantically different outputs: `values` and integer `indices`.
Connect each output to a compatible destination.

## A large model feels slow

The first load still parses, lays out, and creates SVG for the complete graph.
After loading:

- Keep the ONNX tab pinned in VS Code to avoid a full reload.
- Use Find and the extended zoom range for navigation.
- Prefer incremental edits; use Re-layout only when a complete rebuild is
  useful.
- Close unused model tabs if Electron, the VS Code extension host, or
  WindowServer retains significant CPU or memory.

## Save As succeeds but the original file is unchanged

Save As always opens the platform file chooser. Select another filename to
create a copy. To overwrite the original, explicitly select the original path
and confirm the platform overwrite prompt.

The saved file includes the current edit session even when HNNX is back in View
mode.
