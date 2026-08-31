# Graph Editing

English | [한국어](ko-kr/graph-editing.md)

HNNX includes a beta ONNX editor backed by NVIDIA ONNX GraphSurgeon. It edits
the ONNX main graph while preserving HNNX's visual inspection workflow.

Always test editing on a copy of an important model.

## GraphSurgeon environment

Editing, Save As validation, and shape inference require Python 3 with `onnx`
and `onnx_graphsurgeon`.

Create the recommended environment automatically:

- **macOS desktop:** **HNNX > GraphSurgeon Settings… > Create Recommended Environment**
- **Windows/Linux desktop:** **View > GraphSurgeon Settings… > Create Recommended Environment**
- **VS Code:** run **HNNX: Create GraphSurgeon Environment**

The recommended path is `~/.hnnx/venv`. VS Code creates it on the active
extension host, including Remote SSH, Dev Container, and Kubernetes hosts.

Manual setup on macOS or Linux:

```bash
python3 -m venv ~/.hnnx/venv
~/.hnnx/venv/bin/python -m pip install onnx onnx_graphsurgeon \
  --extra-index-url https://pypi.ngc.nvidia.com
```

Manual setup on Windows PowerShell:

```powershell
py -m venv $HOME\.hnnx\venv
$HOME\.hnnx\venv\Scripts\python.exe -m pip install onnx onnx_graphsurgeon `
  --extra-index-url https://pypi.ngc.nvidia.com
```

To use an existing interpreter, select it in GraphSurgeon Settings or run
**HNNX: Configure GraphSurgeon Python** in VS Code. HNNX validates that both
required modules can be imported before saving the path.

The automatic command installs packages only inside the dedicated virtual
environment. It does not bypass PEP 668 or modify Homebrew/system Python.

## Enter and leave Edit mode

- Press `E` or select `EDIT · BETA` to enter Edit mode.
- Press `V` to return to View mode.
- Edits remain in the current session after returning to View mode.
- `SAVE AS` always serializes the current session state.

Loaded AIMET encodings are a separate, read-only attachment. Editing topology
does not rewrite tensor names or QParams in the encodings file.

## Connections

Select a connection line to see its exact `tensor → node.input` mapping.

- **REPLACE:** select the action, then choose an orange source output.
- **DISCONNECT:** temporarily clear the selected destination input.
- **New branch:** click or drag an output port to a compatible input.
- **Fan-out:** choose a specific consumer from the branch list; hovering an
  entry emphasizes its line.
- **Cancel:** press `Q` or right-click empty graph space.

Required inputs can be temporarily disconnected, but appear red as
`REQUIRED · MISSING`. Optional inputs use purple markers. Save As and Infer
Shapes remain blocked until every required input and graph output is valid.

## Nodes and graph endpoints

Click a node to manage its inputs, name, and output tensor names. The searchable
`+ ADD` dialog can create typed Graph Inputs, Graph Outputs, and a curated set
of common Opset 17 operators.

`D` deletes the selected editable connection, endpoint, or safe node. Node
deletion remains disabled while the node still feeds another node or graph
output. Initializers are read-only; HNNX does not provide arbitrary weight
editing or training.

Nodes and Graph Input/Output endpoints can be dragged for visual organization.
These positions are view-only; `RESET` restores automatic layout.

## Validation, layout, and saving

- **INFER SHAPES** applies unsaved commands to a temporary model, runs the ONNX
  checker and strict shape inference, and overlays inferred types and shapes.
- **RE-LAYOUT** rebuilds and lays out the complete graph.
- **REFRESH VIEW** appears as a quiet fallback only when an incremental preview
  is pending.
- **SAVE AS** applies GraphSurgeon edits, topologically sorts and validates the
  model, then opens the platform file chooser.
- **RESET** discards the complete unsaved edit session and intentionally has no
  keyboard shortcut.

Models using external tensor data should be saved beside the original ONNX so
their relative `.data` references remain valid.
During shape inference HNNX reads only bounded external ranges required by
small Constant and shape/axes/sizes tensors. Large weights are not loaded, and
the temporary values are not embedded into the saved model.

## Keyboard shortcuts

Shortcuts do not intercept typing in text, search, or editable controls.

| Shortcut | Action |
| --- | --- |
| `E` | Enter Edit Beta |
| `V` | Return to View mode |
| `R` | Re-layout the graph |
| `I` | Infer shapes |
| `Cmd+S` / `Ctrl+S` | Save As |
| `D` | Delete or disconnect the selected editable item |
| `Q` | Cancel the current edit selection |
| `Cmd/Ctrl+Z` | Undo |
| `Cmd/Ctrl+Shift+Z` | Redo |

## Current limits

- The main graph is editable; nested subgraph editing is not supported.
- Arbitrary initializer values and trained weights are not editable.
- Incomplete required inputs cannot be saved or shape-inferred.
- Custom operators without ONNX shape functions may remain partially inferred.
- External AIMET encodings are not automatically updated after topology edits.
- Large models still require a full initial parse, layout, and SVG build.
