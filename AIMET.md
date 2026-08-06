# HNNX

HNNX is an ONNX graph workbench for model inspection, AIMET encoding
visualization and NVIDIA ONNX GraphSurgeon-backed editing. It is built on
Netron's mature model visualization engine; upstream attribution is retained
in the license and project documentation.

## Included features

- AIMET encoding formats 0.6.1, 1 and 2
- Activation and parameter precision badges
- Bit-width inference through encoding-free precision-preserving operations
- Mixed-precision heatmap
- Input, parameter and output quantization details
- Explicit Graph Input QParam badges and endpoint details
- Model statistics and encoding mismatch counts
- KV-cache classification
- Resizable property sidebar
- Transparent scrollbar tracks with visible, draggable position indicators
- VS Code Webview-specific horizontal and vertical scroll thumbs
- Fine-grained zoom controls with an extended 4% overview scale
- Simultaneous ONNX, external `.data`, and encodings drag-and-drop
- VS Code ONNX Custom Editor with remote-workspace support
- VS Code ONNX tabs are automatically kept open by default when switching files
- HNNX welcome screen and product identity
- ONNX GraphSurgeon Editor with direct connection-line selection
- Edit ports aligned to the actual routed wire attachment points
- Connection replacement and temporary disconnection of any non-initializer input
- Undo, redo and full edit-session reset
- NVIDIA ONNX GraphSurgeon-backed Save As export
- Strict ONNX shape inference over the current unsaved graph edits
- External tensor-data references are preserved without copying large weight files
- Fixed Split-to-Concat edge bundles for repeated parallel tensors

## Workspace shortcuts

Shortcuts are disabled while typing in an input, search field, text area or
editable control.

- `E`: enter Edit Beta
- `V`: return to View mode
- `R`: rebuild and re-layout the current graph
- `I`: run ONNX shape inference
- `Cmd+S` on macOS or `Ctrl+S` on Windows/Linux: Save As in View or Edit mode
- `D`: delete the selected editable node, endpoint or connection
- `Q`: cancel the current edit selection
- `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`: undo and redo

`RESET` intentionally has no shortcut because it discards the complete
unsaved edit session.

`SAVE AS` always opens the platform file chooser with the current ONNX name.
Choose another name for a copy, or choose the original path and confirm the
platform overwrite prompt. Saving always includes the current session edits,
whether the interface is in View or Edit mode.

## Showcase model

`examples/hnnx-showcase.onnx` is a small Opset 17 model accompanied by
`examples/hnnx-showcase.encodings`. It covers Conv parameters, mixed
activation precision, a four-way Split-to-Concat bundle, residual fan-out,
Clip optional inputs, a removable Cast, TopK values/indices, graph outputs,
shape inference and visual editing. Regenerate both files with:

```bash
~/.hnnx/venv/bin/python scripts/create-hnnx-showcase.py
```

Activation bit width is inferred through `Identity`, `Cast`, `Transpose`, `Reshape`,
`Flatten`, `Squeeze`, `Unsqueeze`, `DepthToSpace`, `SpaceToDepth`, `Split` and
`Concat` when a connected precision-preserving chain has one unambiguous
encoded precision. Concat propagation is disabled when its inputs contain
conflicting precisions.
For `TopK`, only the first `values` output inherits activation precision; the
second `indices` output remains its ONNX integer type and is never labeled as
an AIMET activation precision. Only the display precision is inferred; scale, zero point and other QParam
details remain limited to tensors that have explicit encodings.

Direct `Split.outputs -> Concat.inputs` connections with at least three
same-shaped, same-typed tensors are collapsed into one visual edge. The label
uses `×N · shape · precision`, for example `×32 · 1×64×1×2048 · A8`, where N
is the number of actual connection lines collapsed into the bundle, including
repeated occurrences of the same tensor in the Concat input list. Bundles remain
collapsed as a single representative edge and never change the ONNX graph. The
hidden logical connections are retained for editing and serialization but do not
participate in layout, so a large bundle cannot shift or clip the graph.
In Edit mode a bundle has one source port and one target port, both labelled
`×N`. Selecting either port opens a searchable list for choosing the exact ONNX
output tensor or input slot; individual logical ports are not drawn on top of the
collapsed edge.

The beta ONNX GraphSurgeon Editor is entered with the `EDIT · BETA` toolbar
button. Select a
connection line to inspect its exact `tensor → node.input` mapping. `REPLACE`
then lets you choose an orange source output. `DISCONNECT` can temporarily
remove any non-initializer connection, including a required input or the line
feeding a graph output. A selected connection can also be disconnected with
`D` on every platform. Text fields retain their normal typing behavior.
The editor treats AIMET encodings as a separate, read-only visualization
attachment. Editing ONNX topology does not rewrite, propagate or rebind the
loaded encodings. Click or drag an output port to start a new branch. Shift has
no special connection behavior. Select a line and use the explicit `REPLACE`
action when an existing connection should be moved.
Structural edits update the ONNX model
immediately but defer the expensive full graph rebuild until `REFRESH VIEW` is
pressed. An output can also be dragged directly to a compatible input; this
updates only one temporary preview line while dragging and commits on drop.
Operator nodes and graph Input/Output endpoints can be dragged without a full
layout pass; only directly connected lines are redrawn while the surrounding
graph is visually subdued. These positions are view-only and `RESET` restores
the automatic layout. The searchable `+ ADD` dialog can add typed graph inputs,
graph outputs sourced from an existing tensor, and a curated set of common ONNX
operators. Operator forms expose the required tensor inputs and commonly used
attributes such as `axis`, `perm` and `to`. Additions participate in undo/redo,
shape inference, graph refresh and `SAVE AS`.
A
disconnected slot remains editable: click its node to search for a
tensor, use `PICK` to reconnect from the graph, or use `CLEAR` to remove the
current connection. Optional ports use purple markers, with a dashed hollow
marker for a disconnected slot. Missing required inputs use red markers and
`REQUIRED · MISSING` status. This intentionally permits an invalid intermediate
editing state, but `SAVE AS` and `INFER SHAPES` remain blocked until every
required input is reconnected. Pressing `Q` or right-clicking empty graph space
cancels the current connection selection. The menu also lists required inputs in
blue, optional inputs in purple and initializers as read-only. Required and
optional inputs can both be rewired. Required inputs use a status marker
instead of a meaningless checkbox. Focusing a tensor field opens the full
candidate list and typing filters tensor names by substring; optional inputs
alone retain an ON/OFF checkbox. Node names and output tensor names can be
renamed with undo/redo; all references inside the ONNX graph follow an output
rename, but external files are not changed. The same menu can delete an unused leaf node; deletion stays disabled
while the node still feeds another node or a graph output. The original
orange-output/blue-input workflow remains available for creating a new
residual-style connection. The editor validates tensor data types, rejects
cycles, and routes residual shortcuts outside the sequential branch.
Rewiring a graph output preserves its external output name instead of adopting
the selected source tensor name. GraphSurgeon inserts an `Identity` boundary
when required to keep the ONNX interface valid. Click the graph output
endpoint itself to edit its external name with undo/redo.
Its context menu can also disconnect the current source. A disconnected graph
output remains visible as a red port and must be reconnected before save.
It edits the main graph only. Save to another path for a copy, or select the
source path and confirm the platform overwrite prompt.
The visual editor maintains an immediate preview, while NVIDIA ONNX
GraphSurgeon performs the actual ONNX modification, topological sorting and
validation during `SAVE AS`.
Connected edit ports are positioned from the already-computed graph edge
endpoints after layout. The position index is built in one linear edge scan
per rendered graph; it does not continuously measure the DOM or run per-frame
tracking. Fan-out connections that carry the same output tensor share one
source anchor and one edit port; their downstream routes remain independently
laid out. `RESET` discards every unsaved command, clears inferred shape
overlays and restores the model to the state at which the current edit session
started.

`REFRESH VIEW` is a secondary fallback for the few edits whose incremental
preview is pending. It remains visible as a quiet disabled control beside
`RE-LAYOUT`, and becomes available only when a deferred redraw exists.
`RE-LAYOUT` remains the primary full graph rebuild.

`INFER SHAPES` applies the current unsaved edit commands to a temporary model,
runs the ONNX checker and strict `onnx.shape_inference`, then displays inferred
tensor types and dimensions in the current graph. The source ONNX and external
data files are not modified. Any later graph edit clears these inferred values
to prevent stale shapes. Passing inference means the graph is consistent with
the shape rules available to ONNX; custom operators without an ONNX shape
function may remain only partially inferred.

## ONNX GraphSurgeon setup

Editing requires Python 3, ONNX and NVIDIA ONNX GraphSurgeon. The application
does not install Python packages automatically. A dedicated environment is
recommended:

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

The macOS app discovers this environment automatically. Alternatively, set
`HNNX_PYTHON` to the desired interpreter. Team members can also open
**HNNX > GraphSurgeon Settings…** (`Cmd+,`) to select and validate a
Python executable. The path is saved in the app configuration. If no usable
environment is found during `SAVE AS`, the same setup dialog opens
automatically before a destination file is requested.

For VS Code Remote or Kubernetes, create the environment on the remote
extension host. The extension also supports the
`hnnx.pythonPath` setting and the
**HNNX: Configure GraphSurgeon Python** command. A missing environment
offers direct actions to enter the interpreter path or open Settings.

Models using external tensor data must save the edited ONNX beside the
original model so its existing `.data` files remain valid. Graph editing is
available in the macOS app and VS Code extension; a standalone browser cannot
run the Python backend.

Node badges represent explicit Graph Input/output QParams (`Q:A8`), parameter QParams
(`W8`) or quantization signatures (`Q:A8→A16`, `Q:A8/A16→A16`). Propagated precision is
shown only on graph edges as a muted dashed label such as `~A8`, so
precision-preserving operations are not mistaken for quantization points.

HNNX provides **Auto**, **Light**, and **Dark** appearance modes. In the
macOS app, choose **View > Theme**; Auto follows the macOS appearance. In the
VS Code extension, choose **View > Theme** from the HNNX menu or set
`hnnx.colorTheme`; Auto follows the active VS Code color theme, including
theme changes made while the model is open.

## Development

```bash
npm install
npm test
npm start
```

The npm install process does not install Python packages.

## Build the macOS app

Apple Silicon:

```bash
npm run build:mac-hnnx
```

The unsigned DMG is written to `dist/HNNX-0.1.10-arm64.dmg`.
The custom build uses a separate application identifier and settings folder,
and official Netron auto-updates are disabled.

## Build the VS Code extension

```bash
npm run build:vscode-hnnx
```

The VSIX is written under `vscode-extension/`.

## Build Windows and Linux desktop packages

```bash
npm run build:windows-hnnx
npm run build:linux-hnnx
```

The Windows build produces an unsigned x64 NSIS installer. The Linux build
produces an x64 AppImage and Debian package. HNNX uses
`~/.hnnx/venv/Scripts/python.exe` on Windows and
`~/.hnnx/venv/bin/python3` on Linux for its recommended GraphSurgeon
environment.
