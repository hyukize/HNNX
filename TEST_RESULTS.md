# HNNX Regression Results

Run date: 2026-08-06  
Host: Apple Silicon macOS

The test scope is defined in `REGRESSION_TEST.md`.

## Automated results

| Gate | Result | Detail |
| --- | --- | --- |
| JavaScript static analysis | PASS | Changed editor source and all Stage 2/3 browser tests, 0 errors |
| Python static analysis | PASS | `ruff 0.16.1` in the configured HNNX environment; 0 errors |
| GraphSurgeon backend | PASS | Configured HNNX environment, 16 / 16, including missing external tensor data |
| Stage 1 edit matrix | PASS | 100 / 100 multi-edit backend workflows |
| Stage 2 mixed UI | PASS | 100 / 100 ordered Infer/Re-layout/View/Q/drag/history workflows |
| Stage 3 model state machine | PASS | 100 / 100 deterministic round-trip, partial-history, atomic-failure and recovery workflows |
| Stage 3 adversarial UI | PASS | 100 / 100 unique high-speed history and renderer workflows; 98 in the full pass plus 2 setup-timeout cases rerun unchanged and passed |
| AIMET precision policy | PASS | TopK `values` inherits activation bitwidth; `indices` remains unlabelled |
| VS Code extension | PASS | VSIX package built successfully; remote-runtime smoke test remains manual |
| AIMET browser/editor E2E | PASS | 13 / 13, including the HNNX showcase model, workspace shortcuts, fixed Split-Concat bundles, fan-out port alignment and physical D/Q/Z with Korean IME |
| Netron validation models | NOT RUN | Not rerun in this focused editor audit |
| Upstream browser/Electron E2E | NOT RUN | Not rerun in this focused editor audit |
| VSIX packaging | PASS | 183 files; web app and Python backend present |
| Apple Silicon DMG build | PASS | DMG mounted; app executable is arm64; backend is unpacked |

Validation formats included Core ML, ExecuTorch, GGUF, Keras, ncnn, ONNX,
ONNX Runtime, OpenVINO, Pickle, PyTorch, Safetensors, scikit-learn,
TensorFlow, and TensorFlow Lite.

## Regressions found and fixed during this run

1. A root-level stale `view.js` copy was untracked and could mislead manual
   packaging or review; it was removed.
2. Deleting a node removed SVG edges but could leave them in the visual value
   selection list, leading to `replaceChild` on a detached element during a
   later reconnect. Visual edge removal now clears focus and selection refs.
3. Restoring a deleted node assumed its saved next sibling still belonged to
   the same SVG parent. Restoration now safely appends when that sibling has
   changed.
4. Incremental connection undo/redo left a false `REFRESH VIEW` dirty state.
   It now remains fully incremental when no view rebuild is required.
5. The Graph Input edit-menu listener was attached to the generic operator
   node class instead of the graph-input class. Graph Inputs now open their
   own menu, and `D` correctly deletes both original and newly added inputs.
6. The `D` dispatcher checked generic `source` metadata before graph endpoint
   metadata, so a Graph Input could be mistaken for an operator node. Endpoint
   branches now take precedence.

7. Test hosts did not implement the persistent `get`/`set` contract used by
   sidebar width and viewer options.
8. AIMET sidebar collection assumed every framework argument was an array,
   which broke scalar metadata in non-ONNX formats.
9. Empty ONNX optional-input placeholders leaked into normal viewer
   validation. They are now materialized only after Edit Beta is entered.
10. The local browser test server still invoked `python` on macOS. It now uses
   `python3`; package commands choose `python3` outside Windows.
11. ADD E2E advanced before its asynchronous graph refresh completed. The test
   now waits for the completed addition before opening the next dialog.
12. Shape inference stopped at ONNX validation when referenced external tensor
   data was unavailable. It now skips only that data-dependent validation,
   continues structural inference, and returns a visible limitation warning.
13. Repeated Split outputs feeding the same Concat produced dozens of parallel
   visual edges. Same-shape/type semantic peers now collapse into a fixed
   bundle while preserving every underlying ONNX connection.
14. Entering Edit mode rebuilt optional ports and changed the SVG origin, which
   moved the visible graph. Edit entry now preserves the origin, zoom and scroll;
   edge bundles remain collapsed in Edit mode. Hidden bundle members are excluded
   from layout while their logical ONNX connections remain available to the editor.
15. Adding a graph input could leave multiple renderer entries referring to the
    same graph state. Deleting an operator immediately afterward updated the model
    and Undo stack but could leave the live SVG node visible. Visual removal now
    prefers connected renderer entries, detects duplicate live entries, and uses
    a position-preserving redraw after graph-interface edits.
16. Shift-click and Shift-drag attempted to move an existing outgoing branch,
    which duplicated the explicit line-selection `REPLACE` workflow and was easy
    to trigger inconsistently. Shift now has no special connection behavior;
    output click and drag always start a normal connection.
17. Fixed Split-to-Concat bundles blocked pointer dragging from their collapsed
    output port. A disconnected representative bundle path can now be restored by
    dragging the same collapsed Split output port back to the Concat input port.
18. Renamed the deferred renderer action from `UPDATE GRAPH` to `REFRESH VIEW`
    across the toolbar, status messages, tooltips, documentation and packages.
19. `SAVE AS` was hidden outside Edit mode. It is now always visible in View and
    Edit modes, opens with the current ONNX path, supports a chosen copy path or
    confirmed overwrite, and remains accessible in a wrapped narrow toolbar.
    View/Edit now controls only the editing UI: saving always serializes the
    current session state, and incomplete edited graphs remain blocked in either mode.

## Package artifacts

| Artifact | Size | SHA-256 |
| --- | --- | --- |
| `dist/HNNX-0.1.0-arm64.dmg` | 127 MB | `2b2da8d48ca20a6b1f73b6ab604340090098ba0d95a93eab57206cd8c79c2898` |
| `vscode-extension/hnnx-0.1.0.vsix` | 3.6 MB | `c9f8774c94e68f0dbb98972fb357f3958cc81bb8828df0948feebcff76380250` |

## Manual checks still required

- Install the DMG and exercise one complete edit/save/reopen flow.
- Install the VSIX in a real local and Kubernetes/remote VS Code window.
- Confirm vertical and horizontal webview scroll thumbs with a large model.
- Close all model tabs and observe the extension host and WindowServer CPU for
  several minutes.
- Exercise invalid cycle/type connections and unsafe node deletion messages.

These are not counted as automated passes.

## Environment note

The generic `npm run build` also attempts Windows, Linux DEB, and Linux RPM
packages. macOS, Windows, and DEB stages completed, but the RPM stage was
blocked because `rpmbuild` is not installed on this Mac. The dedicated
Apple Silicon `build:mac-hnnx` target completed successfully.
