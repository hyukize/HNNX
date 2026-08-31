# HNNX Conservative Regression Test

This matrix is intentionally conservative. A release candidate is considered
healthy only when every automated gate passes and the short manual package
smoke test has no blocker.

## Result levels

- **PASS**: observed result matches the expected result.
- **FAIL**: behavior is incorrect or the test process exits non-zero.
- **BLOCKED**: the required runtime, model, or interactive environment is not
  available. A blocked test is not counted as a pass.
- **N/A**: the platform-specific case does not apply to the current package.

## A. Static and build gates

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| A1 | Full JavaScript ESLint | No error or warning promoted to error | Automated |
| A2 | Python backend compile | `onnx-graphsurgeon.py` compiles | Automated |
| A3 | Web build | `dist/web` builds without missing assets | Automated |
| A4 | VSIX package | VSIX builds and contains the web application and Python backend | Automated |
| A5 | macOS package | arm64 DMG builds and mounts | Automated build, manual mount |

## B. AIMET viewer

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| B1 | Open ONNX, external data and encodings together | ONNX is selected as the model, `.data` remains a sidecar, and encodings attach in one operation | Automated |
| B2 | Quantization badges | Explicit and inferred precision use the correct label and color | Automated |
| B3 | QParam details | Input tensor precision, output QParams, and parameter encodings are separated | Automated |
| B4 | Statistics | A8/A16/mixed and matched/mismatched counts are present | Automated |
| B5 | Resizable sidebar | Sidebar grows leftward and width persists across views | Automated |
| B6 | Find panel | Search field stays inside the sidebar and scroll thumbs remain usable | Automated + manual |
| B7 | Zoom | Control-wheel and Command-wheel zoom in fine steps; overview zoom-out remains usable | Automated + manual |
| B8 | Graph Input QParam | An explicitly encoded model input shows a colored `A*` badge and `Graph Input QParam` details | Automated |
| B9 | Appearance modes | Auto follows the host theme; Light and Dark overrides update every bundled theme rule and persist | Automated + packaged manual smoke |

## C. Graph editor connections

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| C1 | Enter/leave editor | Beta state and editor toolbar are visible; viewer state is preserved | Automated |
| C2 | Select connection | Edge, source, target, and available actions are unambiguous | Automated |
| C3 | Disconnect optional input | Only the selected edge disappears immediately | Automated |
| C4 | Disconnect required input | Edge disappears, missing port is red, save is disabled | Automated |
| C5 | Reconnect missing input | New edge appears immediately and save becomes valid | Automated |
| C6 | Replace connection | Old edge is removed and the new edge is routed without model reload | Automated |
| C7 | Graph output disconnect/reconnect | Output can be temporarily missing but cannot be saved unresolved | Automated |
| C8 | Fan-out and replacement | Output click/drag always creates a branch; selecting a line exposes the explicit Replace action | Automated |
| C9 | Invalid connection | Self-reference and cycles are rejected before connection; incompatible types remain manual | Automated + manual |
| C10 | Right-click empty canvas | Active source/target selection is cancelled | Automated |
| C11 | Q shortcut | Physical `KeyQ` cancels selection, including Korean IME layouts | Automated |
| C12 | D shortcut | Physical `KeyD` disconnects the selected edge, including Korean IME layouts | Automated |
| C12a | Undo/redo shortcut | Physical `KeyZ` performs undo/redo with platform modifier, including Korean IME layouts | Automated |
| C13 | Delete pass-through then reconnect | Delete a Cast, reconnect its consumer, then undo/redo both changes without a graph rebuild or detached-DOM error | Automated |
| C14 | Shortcut focus safety | `D`/`Q` act as commands on the canvas but remain ordinary text inside inputs | Automated |
| C15 | TopK AIMET precision | Input precision propagates only to `values`; integer `indices` remains unlabelled | Automated |
| C16 | Netron-style node-pair bundle | Two or more logical connections sharing one producer and consumer remain one fixed `×N · shape · precision` edge regardless of operator, metadata availability or host; hidden members do not affect layout bounds | Automated + real-model smoke |
| C17 | Bundled edit ports | Bundled producer output and consumer input each show one `×N` port; searchable selectors expose the exact tensor and input slot | Automated |

## D. Editing, history, and layout

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| D1 | Move operator node | Node moves and its directly connected edges stay attached | Automated |
| D2 | Move graph Input/Output | Endpoints move like operator nodes | Automated |
| D3 | Live canvas expansion | Dragging beyond the current bounds expands the canvas before pointer release | Automated |
| D4 | Drag cleanup | After pointer release, ordinary cursor movement does not pan or resize the graph | Automated |
| D5 | Background pan cleanup | After releasing a canvas pan, cursor movement does not continue scrolling | Manual |
| D6 | Reset | All graph edits and manual positions return to the open-time state | Automated |
| D7 | Undo/redo | Buttons and macOS `Cmd+Z`/`Cmd+Shift+Z` restore edits in order | Automated |
| D8 | Text editing | Backspace deletes text in names/search fields and has no graph shortcut | Automated |
| D9 | Re-layout | Graph rebuild retains the current edit model | Automated |
| D10 | Redo branch invalidation | Undo followed by a different edit clears the stale redo branch | Automated |
| D11 | Missing external data inference | Remove referenced external tensor data; inference continues and reports a warning | Automated |
| D12 | Deferred refresh affordance | `REFRESH VIEW` stays visibly subdued and disabled until a deferred view refresh is pending | Automated + manual UI |
| D13 | Re-layout serialization | Rapid repeated re-layout requests finish in order and restore the enabled `RE-LAYOUT` control | Automated |
| D14 | Connection replacement | Select an existing edge, choose `REPLACE`, select an orange output, then Undo; source path and model connection change and restore | Automated |
| D15 | Fan-out source identity | Multiple connections carrying one output tensor leave one shared source anchor and expose one edit port | Automated + real-model visual check |
| D16 | Atomic attachment layout | ONNX + data + encodings leaves node transforms stable, and one Re-layout activation restores a moved node | Automated + real-model visual check |
| D17 | Post-drag toolbar click | A connection or node drag guard suppresses a click only on the pointer-up target or its descendant, within 4 px and 100 ms; Re-layout passes immediately | Automated A/B regression |

## E. Add, rename, delete, validate, and save

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| E1 | Add common operator | New node and edges appear immediately without `REFRESH VIEW` | Automated |
| E2 | Add graph Input | Name, type, and dimensions appear immediately | Automated |
| E3 | Add graph Output | Selected tensor is exposed immediately | Automated |
| E4 | Rename node/tensor/output | All consumer references remain valid | Automated |
| E5 | Delete unused leaf | Node disappears; undo restores it | Automated |
| E6 | Prevent unsafe deletion | Consumed nodes cannot be deleted without an explicit bypass workflow | Manual |
| E7 | Shape inference | Unsaved edits are inferred; stale shapes clear after a new edit | Backend automated + manual UI |
| E8 | Save As | Available in View/Edit; chosen copy or confirmed overwrite reopens and passes ONNX checker | Backend automated + manual UI |
| E9 | External data | External tensor references remain valid after edits | Automated |
| E10 | Endpoint keyboard deletion | Original and newly added graph inputs/outputs can be deleted with `D` and restored with undo | Automated |

## F. VS Code extension and remote workspace

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| F1 | External data resolution | Relative files resolve inside the model directory | Automated |
| F2 | Traversal protection | Absolute and parent paths outside the directory are rejected | Automated |
| F3 | Preview pinning | Active ONNX preview is kept open; inactive/disabled previews are not | Automated |
| F4 | Python validation | Configured interpreter is accepted; invalid path is rejected | Automated with configured Python |
| F5 | Remote edit bridge | Edits execute through the configured Python backend | Automated with configured Python |
| F6 | Remote shape inference | Inference executes through the configured Python backend | Automated with configured Python |
| F7 | Tab retention | Switching to another file and back does not unnecessarily reload the ONNX model | Manual VS Code |
| F8 | Webview scrollbars | Both vertical and horizontal thumbs are visible when content overflows | Manual VS Code |

## G. Upstream Netron compatibility

| ID | Test | Expected | Coverage |
| --- | --- | --- | --- |
| G1 | Validation model corpus | Every validation-tagged model downloads and parses | Automated |
| G2 | Browser suite | Upstream browser tests pass headlessly | Automated |
| G3 | Desktop suite | Upstream Electron tests pass | Automated |
| G4 | Formats unrelated to ONNX | CoreML, PyTorch, TensorFlow, TFLite, GGUF, and other validation samples remain readable | Automated through G1-G3 |

## H. Release package smoke test

1. Install the newly built DMG on an Apple Silicon Mac.
2. Open an ONNX model, attach AIMET encodings, enter Edit Beta, make one edit,
   undo it, and close the app.
3. Install the newly built VSIX in local VS Code and in one remote workspace.
4. Open a model, switch to a text file, return to the model, and run shape
   inference.
5. Confirm that closing all model tabs leaves no sustained CPU usage from the
   extension host or WindowServer attributable to HNNX.
6. Install the Windows x64 NSIS package and verify file open, theme persistence,
   `Scripts/python.exe`, Infer Shapes and Save As on Windows.
7. Run the Linux x64 AppImage and install the deb package; verify desktop
   identity, theme persistence, `bin/python3`, Infer Shapes and Save As.

Expected: no crash, stale loading state, runaway CPU, missing scrollbar, or
unexpected model reload.

Cross-built Windows and Linux package structure is automated. Native Windows
and Linux GUI execution remains a release smoke test on those operating systems.
