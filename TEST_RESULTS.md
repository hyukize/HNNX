# HNNX Regression Results

Run date: 2026-08-10
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
| VS Code extension | PASS | 9 unit tests passed, including auto-load control, exact-URI reload and document-scoped detach; 3 configured-Python tests skipped |
| AIMET browser/editor E2E | PASS | 15 / 15, including runtime Light/Dark overrides, post-connection first-click Re-layout, HNNX showcase model, fixed Split-Concat bundles and fan-out alignment |
| Netron validation models | NOT RUN | Not rerun in this focused editor audit |
| Upstream browser/Electron E2E | NOT RUN | Not rerun in this focused editor audit |
| VSIX packaging | PASS | 184 files; web app and Python backend present |
| Apple Silicon DMG build | PASS | DMG mounted; app executable is arm64; backend is unpacked |
| Windows x64 NSIS build | PASS | Cross-build completed; PE32+ HNNX executable and unpacked GraphSurgeon backend present |
| Linux x64 AppImage/deb build | PASS | Cross-build completed; x86-64 ELF executable, HNNX desktop identity and unpacked backend present |

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
20. A simultaneous ONNX, external `.data`, and AIMET encodings drop treated the
    `.data` sidecar as a second model and skipped the attachment path. File
    classification now selects the explicit `.onnx` as the model, preserves
    sidecar access, and recognizes both `.encodings` and `.encodings.json`.
21. An explicit QParam attached to a Graph Input was visible only on its edge.
    Graph Input endpoints now use the same `Q:A8`/`Q:A16` badge policy as other
    tensor producers and open a dedicated `Graph Input QParam` detail section.
22. `REFRESH VIEW` occupied a central toolbar position and became the only
    orange control when pending. It now stays as a quiet, disabled secondary
    control beside `RE-LAYOUT`, becoming available only for a deferred redraw;
    `RE-LAYOUT` remains the primary full redraw action.
23. Active Edit mode painted the entire `VIEW` button orange even though only
    the Beta marker was intended to carry the accent. `VIEW` is now neutral
    with an orange outline and Beta marker in both light and dark themes.
24. Full graph renders could overlap, leaving `RE-LAYOUT` apparently ignored
    on large graphs. Re-layout, deferred refresh, Reset, and inferred-shape
    application now share a serialized render queue with immediate `LAYOUT…`
    feedback. Rapid repeated layout requests and the explicit connection
    `REPLACE` workflow are covered by browser regression tests.
25. Netron's native fan-out routing could draw one output tensor from several
    attachment points on the producer node, making paths look laterally shifted
    from their edit port. Fan-out paths now share one producer anchor and one
    output port while retaining their independently computed downstream routes.
    This was verified with both the synthetic fan-out fixture and the real
    `decode_layer_05` ONNX/data/encodings set.
26. AIMET attachment refresh previously animated every node against already-final
    edge paths. On large graphs this extended the mismatch window and could race
    with the first Re-layout interaction. Attachment refresh is now atomic, and
    regression coverage verifies stable node transforms plus a one-activation
    Re-layout after manual movement.
27. Connection and node drag cleanup used a document-wide one-shot click
    suppressor. When the expected synthetic trailing click was absent, the
    listener consumed the next unrelated toolbar click, so Re-layout appeared
    to require two presses. Suppression is now limited to the pointer-up target
    or its descendant, within 4 pixels and 100 milliseconds of pointer-up.
    Clicks on another target or position—including Re-layout—pass immediately.
    The old implementation fails the deterministic regression while the scoped
    implementation passes it.
28. Appearance depended only on `prefers-color-scheme`, so the VS Code Webview
    could disagree with a manually selected VS Code theme and neither package
    offered an override. HNNX now provides persistent Auto, Light and Dark
    modes. Auto follows macOS in the desktop app and the active VS Code theme
    in the extension; runtime host-theme changes update without reloading.
29. Desktop packaging was HNNX-specific only on macOS, and GraphSurgeon's
    recommended environment assumed the Unix `bin/python3` layout. Dedicated
    Windows x64 NSIS and Linux x64 AppImage/deb targets now preserve HNNX
    branding. Windows uses `Scripts/python.exe`; Linux and macOS use
    `bin/python3`. Non-macOS desktop menus also expose the persistent theme
    selector.
30. The VSIX contained the HNNX PNG but did not declare it in the extension
    manifest, so VS Code showed the default extension placeholder. The
    manifest now declares `media/icon.png`; the packaged VSIX manifest and
    256×256 image entry were both inspected.
31. VS Code's native drag-and-drop handling prevented an encodings-only drop
    from reliably reaching the Webview. HNNX now offers optional neighboring
    file auto-load, a compact `ENC` visibility toggle, and File-menu Load,
    exact-URI Reload, and Detach actions. These operations replace only the
    AIMET attachment and do not reload the ONNX model.
32. The encodings lifecycle update was initially packaged only as a VSIX.
    Release 0.1.13 synchronizes the same source and version across the macOS
    arm64 DMG, Windows x64 installer, Linux x64 AppImage/deb, and VSIX.

## Package artifacts

| Artifact | Size | SHA-256 |
| --- | --- | --- |
| `dist/HNNX-0.1.13-arm64.dmg` | 127,152,789 bytes | `97e5078f7f139f145b5aebb7e22dce9fb8e4715bf3dceacbec5cf8a866bcc5fe` |
| `dist/HNNX-0.1.13-x64-setup.exe` | 91,802,962 bytes | `cf4d1b1742133db1e15e236830841d56ef181d47bf1e7815f1e37139dbd616ba` |
| `dist/HNNX-0.1.13-x64.AppImage` | 131,362,586 bytes | `7027df5ecf9c9ee8b693daad95d2d4f12acf4fcb5a148929e0ff4e36c5422521` |
| `dist/HNNX-0.1.13-x64.deb` | 103,187,984 bytes | `7f10ba5658664249b4520db2e06182cc793e3f7dd9d08fa7c767555a7a5a00f8` |
| `vscode-extension/hnnx-0.1.13.vsix` | 3,623,704 bytes | `8ce4b5eb614384b7693c096227dc4ffa1e5a5f6ec4fe5c81ec3c24e6b16b2e74` |

## Manual checks still required

- Install the DMG and exercise one complete edit/save/reopen flow.
- Install the VSIX in a real local and Kubernetes/remote VS Code window.
- Run the NSIS installer on native Windows x64 and both Linux packages on a
  native x64 Debian/Ubuntu desktop.
- Confirm vertical and horizontal webview scroll thumbs with a large model.
- Close all model tabs and observe the extension host and WindowServer CPU for
  several minutes.
- Exercise invalid cycle/type connections and unsafe node deletion messages.

These are not counted as automated passes.

## Environment note

Dedicated HNNX build targets now produce macOS arm64 DMG, Windows x64 NSIS,
Linux x64 AppImage/deb, and VSIX packages. Windows and Linux artifacts were
cross-built on Apple Silicon macOS and structurally inspected; native GUI
smoke tests remain manual.
