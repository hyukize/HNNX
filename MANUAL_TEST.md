# HNNX Manual Test

## Test files

- `test/manual-editor.onnx`
- `test/manual-editor.encodings`

Use a copy of the ONNX file when testing `SAVE AS`.

## 1. macOS installation and GraphSurgeon setup

1. Build and install the latest `dist/HNNX-0.1.17-arm64.dmg` locally.
2. Control-click the copied app, choose **Open**, and confirm the unsigned-app
   prompt. Confirm that macOS does not report the app as damaged.
3. Open **HNNX > GraphSurgeon Settings…** or press `Cmd+,`.
4. Select the Python executable that contains `onnx` and
   `onnx_graphsurgeon`.
5. Confirm that **GraphSurgeon Python is ready** appears.
6. Open the settings again and confirm that the saved interpreter path is
   displayed.

Expected: an invalid interpreter is rejected and cannot be saved.

### Theme selection

1. Open **View > Theme** and select **Dark**, then **Light**.
2. Select **Auto** and change the macOS appearance.
3. Restart HNNX and confirm the selected preference is retained.

Expected: forced themes change immediately; Auto follows macOS without a
reload, and the preference survives restart.

## 2. Open ONNX and AIMET encodings

1. Drag `manual-editor.onnx`, `test/aimet.onnx.data`, and
   `manual-editor.encodings` into the window together.
2. Confirm that the graph opens without a second attachment step.
3. Open Model Properties and confirm that AIMET encodings version `2.0.0`
   is shown.
4. Open Model Statistics with the lower-left info button.

Expected:

- Current graph nodes: `8`
- Nodes with QParam: `4`
- Node precision: A8 `6`, MIXED `2`
- Encodings: A16 `2`, A8 `3`, W8 `1`
- Matched QParams: `6 / 6`
- Inferred activation precision: `5`
- Validation errors and warnings: `0`

## 3. Quantization and precision visualization

Inspect the following branch:

`Abs → Transpose → Split → Concat → Transpose → Mul`

Expected:

- `a` is explicitly A8.
- Precision continues through the encoding-free Transpose, Split and Concat
  branch as inferred `~A8`.
- `Mul` contains the W8 parameter information.
- The transition to explicit `y` A16 is visually distinguishable from inferred
  precision.
- Clicking a quantization badge shows input tensor precision, output QParams
  and parameter details in the right sidebar.
- The encoded Graph Input shows `A8`; clicking it opens `Graph Input QParam`
  details rather than making the input boundary look like an inferred value.

## 4. General viewer UX

1. Resize the right sidebar by dragging its left boundary.
2. Open Find and scroll through its results.
3. Confirm that vertical and horizontal scrollbar thumbs remain visible.
4. Pinch or use the configured wheel modifier to zoom.
5. Zoom out to the overview limit and move across the graph.

Expected: resizing persists while switching sidebar sections, zoom steps are
fine-grained, and scrollbar tracks do not leave opaque empty blocks.

## 5. Enter GraphSurgeon Editor

1. Press `EDIT`.
2. Confirm that `INFER SHAPES`, `RE-LAYOUT`, `SAVE AS`, undo and redo controls
   appear.
3. Confirm that the graph has orange output ports.

Expected: the top status asks for a connection line or orange output.

Output-port gestures:

- Click starts a new outgoing connection while preserving existing branches.
- Shift has no special connection behavior. Select a line and use `REPLACE`
  when an existing connection should be moved.

## 6. Select and replace an existing connection

1. Click the line `alternative → Neg`.
2. Confirm that the selected line turns orange.
3. Confirm that the action bar says `alternative → neg.X`.
4. Press `DISCONNECT` and confirm that `Neg.X` becomes a red missing-required
   port and that Save As is blocked.
5. Undo the disconnect, select the line again, and press `REPLACE`.
6. Select the orange output port for tensor `a`.

Expected:

- The `Neg` node is now connected to `a`.
- The graph is rebuilt without reloading the ONNX file.
- Undo restores `alternative → Neg`.
- Redo restores `a → Neg`.

## 7. Disconnect an optional input

1. Click the line `maximum → Clip`.
2. Confirm that the action bar says `maximum → clip.max`.
3. Confirm that `DISCONNECT` is enabled.
4. Press `DISCONNECT`.

Expected:

- Only the maximum input line disappears.
- The Clip node remains.
- Click `Clip`; `min` and `max` are listed and `max` is OFF.
- Choose `maximum` (or another compatible tensor) for `max`; its line returns.
- Turn `max` OFF again; only that optional line disappears.
- Undo and redo both connection and disconnection.

Click the unused `Clip` leaf node and confirm that `DELETE NODE` is
enabled. Delete it, undo, and confirm that the node returns. On a node whose
output is still consumed, confirm that deletion is disabled with an
explanation.

## 8. Original output-to-input connection workflow

1. Press `Q` to clear any selection.
2. Select an orange output port, or drag it toward another node.
3. Confirm that compatible blue and purple input ports appear.
4. Select the target input, or drop the dragged preview line on it.
5. Try a connection that would consume a node's own output or create a cycle.

Expected: valid residual-style connections are accepted; self-connections,
cycles and incompatible tensor data types are rejected with a status message.

## 9. Graph Output replacement

1. Select the orange output port for tensor `b`.
2. Select the blue port on one of the graph outputs, such as `clipped`.

Expected: that Graph Output now points to `b`; Undo restores the original
output.

## 10. Shape inference over unsaved edits

1. Ensure `Neg` consumes `a`, not `alternative`.
2. Press `INFER SHAPES`.

Expected:

- The status reports `SHAPE INFERENCE PASSED`.
- Tensor `b` changes from the original `[3,5]` branch shape to `[2,4]`.
- Shapes appear on graph connections where Netron has room to display them.
- The source ONNX timestamp does not change.

Then make another graph edit.

Expected: inferred shapes are cleared automatically so stale results are not
displayed. Pressing `INFER SHAPES` computes them again.

## 11. Re-layout and Save As

1. Press `RE-LAYOUT`.
2. Confirm that the current edits remain.
3. Press `SAVE AS` and save as `manual-editor.edited.onnx`.
4. Close the current model and reopen the saved copy.
5. Run `INFER SHAPES` again.

Expected:

- The source model was not overwritten.
- Rewired and disconnected inputs persist in the saved copy.
- The saved model opens normally and passes shape inference.

## 12. VS Code and remote workspace

1. Install `vscode-extension/hnnx-0.1.17.vsix`.
2. Reload VS Code.
3. Run **HNNX: Configure GraphSurgeon Python** from the Command
   Palette.
4. In Remote/Kubernetes, enter the Python path inside the remote environment,
   not the macOS path.
5. Open `manual-editor.onnx`.
6. Switch to another text file and return to the ONNX tab.
7. Repeat connection replacement, optional disconnect, `INFER SHAPES` and
   `SAVE AS`.
8. Set HNNX Theme to Auto, switch the active VS Code color theme, then verify
   explicit Light and Dark overrides through the HNNX View menu.
9. With a neighboring encodings file present, confirm the `ENC` button appears.
10. Toggle `ENC` off and on, modify the file externally, then choose
    **File > Reload Encodings**.
11. Confirm reload uses the existing URI without a picker. Test
    **Detach Encodings** and **Load AIMET Encodings...**.
12. Disable `HNNX: Auto Load Encodings` in Settings and reopen the model.

Expected:

- The ONNX editor remains pinned and does not reload merely because another
  file was selected.
- Shape inference and saving execute through the configured remote Python.
- Missing Python offers **Enter Python Path** and **Open Settings**.
- Auto follows the active VS Code theme immediately; explicit overrides persist.
- Encodings visibility changes without reloading the ONNX model, reload reads
  the exact attached local/remote URI, and automatic loading follows the setting.

## 13. Move endpoints and add graph items

1. Enter edit mode and drag a graph Input endpoint horizontally and vertically.
2. Drag a graph Output endpoint to another position.
3. Confirm their connected lines remain attached, then press `RESET`.
4. Press `+ ADD`, search for `Relu`, select an existing tensor as its input,
   enter a unique output name and add it.
5. Press `+ ADD`, choose **Graph Input**, set its name, data type and shape,
   then add it.
6. Press `+ ADD`, choose **Graph Output**, select the new Relu output tensor,
   enter an external output name and add it.
7. Confirm each addition appears immediately, then undo and redo the additions.
8. Run `INFER SHAPES` and `SAVE AS`; reopen the saved copy.

Expected:

- All three endpoint types can be moved without a full layout pass.
- `RESET` restores the original automatic endpoint and node positions.
- The new Input, Relu and Output appear immediately without pressing
  `REFRESH VIEW`.
- Their names, types, shapes, connections and Relu attributes persist in the
  saved model.
- Undo and redo preserve the expected creation order.

## 14. Pointer cleanup, shortcuts, and text input

1. Drag a node toward the right or bottom edge so the canvas expands, then
   release the pointer.
2. Move the pointer around without holding a mouse or trackpad button.
3. Pan the empty graph background, release it, and move the pointer again.
4. Select a connection and press the physical `D` key while the Korean input
   method is active.
5. Start a source connection and press the physical `Q` key while the Korean
   input method is active.
6. Open `+ ADD`, enter text in the node name and search fields, and use
   Backspace repeatedly.

Expected:

- Cursor movement after either release does not pan, resize or auto-scroll the
  graph.
- Physical `D` disconnects and physical `Q` cancels regardless of the current
  keyboard input language.
- Backspace edits text normally and never triggers graph navigation, reset or
  deletion.

## 15. Windows and Linux desktop packages

### Windows x64

1. Install `dist/HNNX-0.1.17-x64-setup.exe`.
2. Open an ONNX file through HNNX and by double-clicking the associated file.
3. Open **View > Theme**, test Auto/Light/Dark, and restart the app.
4. Open GraphSurgeon Settings and create the recommended environment.
5. Confirm the interpreter is `~/.hnnx/venv/Scripts/python.exe`, then run
   Infer Shapes and Save As.

### Linux x64

1. Mark `dist/HNNX-0.1.17-x64.AppImage` executable and launch it.
2. Install `dist/HNNX-0.1.17-x64.deb` on a Debian/Ubuntu test machine.
3. Confirm both packages use the HNNX icon and `hnnx` executable identity.
4. Test Auto/Light/Dark, GraphSurgeon environment creation, Infer Shapes and
   Save As.

Expected: both platforms open and save ONNX files, retain appearance settings,
and use their platform-correct Python virtual-environment path. Unsigned
Windows builds may show a SmartScreen warning.

## Not implemented yet

- Delete or bypass a multi-node block
- Edit arbitrary attributes on an existing node
