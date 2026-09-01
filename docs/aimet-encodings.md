# AIMET Encodings

English | [한국어](ko-kr/aimet-encodings.md)

HNNX can display an ONNX model together with AIMET encodings formats 0.6.1,
1.x, and 2.x.

## Load encodings

In the desktop app, drag the ONNX model and encodings file together. An
external `.data` sidecar can be included in the same drop.

In VS Code, HNNX automatically searches beside the model for:

```text
model.encodings
model.onnx.encodings
model.encodings.json
```

Control this behavior with `hnnx.autoLoadEncodings`.

Use the File menu to manage the attachment without reopening the model:

- **Load AIMET Encodings…** selects a file.
- **Reload Encodings** rereads the same local or remote URI.
- **Detach Encodings** removes only the attachment.
- The `ENC` toolbar button hides or shows the visualization while keeping the
  attachment loaded.

## Read the graph

Node and endpoint badges represent explicit encoding information:

- `A8`, `A16`: explicit activation or Graph Input QParam
- `W4`, `W8`, `W4/W8`: parameter bit width
- `A8→A16`: input precision changes to an explicit output precision
- `A8/A16→A16`: mixed input precision produces an explicit A16 output

Propagated precision is shown on graph edges as a muted label such as `~A8`.
This distinguishes a precision-preserving path from a node that owns an
explicit QParam.

Click a badge, tensor, node, or encoded Graph Input to inspect scale,
offset/zero point, quantization range, axis, block size, granularity, and
symmetry. HNNX preserves explicit min/max values from legacy encodings. For
integer AIMET 1.x and 2.x encodings, it derives the representable range from
bit width, scale, and offset/zero point when the file does not store min/max.
The detail view labels the range source as `explicit` or `derived`.

Derived ranges are limited to unambiguous integer per-tensor and per-channel
encodings. HNNX does not invent ranges for propagated `~A8` precision,
floating-point encodings, incompatible scale/zero-point arrays, or LPBQ scale
representations.

## Precision propagation

When there is one unambiguous encoded precision, HNNX can trace it through
encoding-free operations including:

```text
Identity, Cast, Transpose, Reshape, Flatten, Squeeze, Unsqueeze,
DepthToSpace, SpaceToDepth, Split, Concat
```

Cast is treated as precision-preserving for AIMET visualization. Concat does
not infer one precision when its activation inputs conflict. For TopK, only
the `values` output inherits activation precision; `indices` remains an ONNX
integer tensor.

This propagation infers display precision only. It does not invent scale,
zero-point, or other QParam values.

## Statistics and mismatches

The model information view summarizes:

- Nodes and nodes with explicit QParams
- Activation and parameter bit-width distribution
- Mixed-precision nodes and transitions
- Matched and unmatched encoding entries
- Inferred activation precision
- Validation warnings and errors
- KV-cache precision classification

An unmatched entry does not always mean the ONNX is invalid. It can result
from export-time names, removed tensors, a stale encodings file, or intentional
topology edits. Compare the tensor name and graph location before deciding how
to handle it.

## Editing behavior

AIMET encodings are read-only and independent from GraphSurgeon editing.
Renaming, deleting, or rewiring ONNX tensors does not modify the external
encodings file. Detach encodings before a complex edit when the old badges
would be misleading, then load an updated encoding export afterward.

See [Graph Editing](graph-editing.md) for the ONNX editing workflow.
