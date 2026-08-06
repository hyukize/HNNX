# HNNX

View ONNX models and AIMET quantization encodings without leaving VS Code.

## Usage

Open an `.onnx` file and it will use **HNNX** by default.
You can switch editors at any time with **Open With...** from the editor title
context menu.

Workspace shortcuts are `E` for Edit, `V` for View, `R` for Re-layout and
`I` for Infer Shapes. Open Save As with `Cmd+S` on macOS or `Ctrl+S`
on Windows/Linux. These shortcuts do not intercept typing in form controls.
`RESET` intentionally has no shortcut.

The extension automatically attaches a neighboring encoding file named
`<model>.encodings`, `<model>.onnx.encodings`, or `<model>.encodings.json`.
Use **File > Open...** inside the viewer to select a different encoding file.

Use the `EDIT · BETA` toolbar button to enter **ONNX GraphSurgeon Editor**. Select a
connection line directly to see its exact destination input. Choose `REPLACE`
and then an orange source output, or use `DISCONNECT` to temporarily remove
any non-initializer connection, including the line feeding a graph output.
The keyboard shortcut is `D` on every platform when a connection is selected.
Click an output port and then a replacement orange output to move the source
of an existing connection. For multiple consumers,
choose a branch from the list and hover an entry to emphasize its line.
Shift-click behaves like a normal click. Structural edits are collected until
`REFRESH VIEW` redraws them in the graph view, avoiding a full layout after every
operation. Nodes remain draggable and only their directly connected lines are
redrawn during movement. Graph Input and Output endpoints can be positioned in
the same way; `RESET` restores the automatic layout. Use the searchable
`+ ADD` dialog to add a typed graph input, a graph output sourced from an
existing tensor, or a common ONNX operator with its tensor inputs and relevant
attributes. Additions participate in undo/redo, shape inference, refresh and
`SAVE AS`.
Right-click an operator to search its inputs,
reconnect a disconnected slot, clear an existing connection, or delete an
unused leaf node. Purple markers distinguish optional ports; missing required
inputs use red markers and `REQUIRED · MISSING`. The editor permits this
invalid intermediate state, but blocks `SAVE AS` and `INFER SHAPES` until all
required inputs are reconnected. Right-clicking empty graph space cancels the current connection
selection. The same menu shows required inputs, read-only initializers, node
names and output tensor names. Required inputs can be rewired, and node/output
renames participate in undo and redo. Tensor fields open a searchable full
candidate list; only optional inputs have an ON/OFF checkbox. External
name-based files are not
updated. Undo and redo use the platform-native
shortcuts. `SAVE AS` is available in View and Edit modes and always serializes
the current session state, including edits made before returning to View mode.
Choose a different path for a copy, or choose the source path and confirm overwrite.
The saved model is modified, topologically sorted and validated by NVIDIA
ONNX GraphSurgeon rather than by the Webview.
Connected edit markers are aligned with the actual routed wire attachment
points. `RESET` discards the complete unsaved edit history and restores the
graph to the beginning of the current edit session.
Graph output rewiring preserves the existing external output name through an
`Identity` boundary when needed. Right-click the graph output endpoint to
rename that external interface explicitly or disconnect its current source.
A disconnected graph output remains visible as a red port until reconnected.

`INFER SHAPES` runs the current unsaved edits through a temporary GraphSurgeon
model and strict ONNX shape inference, then updates tensor dimensions in the
viewer without changing the source file. Custom operators without an ONNX
shape function may remain partially inferred.

Install the Python backend in the local or remote extension host:

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

The extension discovers this environment automatically. A different
interpreter can be selected with **HNNX: Python Path** or the
**HNNX: Configure GraphSurgeon Python** command. The command validates
that both `onnx` and `onnx_graphsurgeon` can be imported before saving the
path. If automatic discovery fails while saving, the extension offers to
enter a path or open the relevant Settings page.

The extension is workspace-side compatible, so the same workflow works after
VS Code attaches to a running Kubernetes container. ONNX external data files
are loaded from the model directory only when the viewer requests them.

ONNX tabs are kept open by default, so switching to another file preserves the
loaded graph. Disable **HNNX: Auto Pin Onnx Editors** in Settings
to restore the normal VS Code preview-tab behavior.

## Current MVP limits

- ONNX GraphSurgeon Editor rewires and temporarily disconnects inputs in the
  main graph only. Incomplete required inputs cannot be saved.
- Nested subgraphs and arbitrary initializer-value editing are not supported yet.
- One encoding attachment at a time.
- Remote files are transferred to the Webview as complete byte arrays.
- ONNX external data paths are restricted to the model directory.
- Edited models with external data must be saved beside the original ONNX.
