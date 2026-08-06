import importlib.util
import itertools
import json
import os
import tempfile
import unittest

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(ROOT, "source", "onnx-graphsurgeon.py")
SPEC = importlib.util.spec_from_file_location("netron_editor_matrix", BACKEND_PATH)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


def _tensor(name, data_type, dimensions):
    return helper.make_tensor_value_info(name, data_type, dimensions)


def _initializer(name, values, dtype=numpy.int64):
    return numpy_helper.from_array(numpy.asarray(values, dtype=dtype), name)


def _source_model(directory):
    path = os.path.join(directory, "workflow.onnx")
    nodes = [
        helper.make_node("Identity", ["x"], ["base"], name="source"),
        helper.make_node(
            "Clip", ["base", "scalar", "scalar"], ["clipped"], name="clip"
        ),
        helper.make_node("Relu", ["unused"], ["dead"], name="dead_branch"),
    ]
    inputs = [
        _tensor("x", TensorProto.FLOAT, [2, 3]),
        _tensor("y", TensorProto.FLOAT, [2, 3]),
        _tensor("matrix", TensorProto.FLOAT, [3, 4]),
        _tensor("condition", TensorProto.BOOL, [2, 3]),
        _tensor("indices", TensorProto.INT64, [2, 3]),
        _tensor("small", TensorProto.FLOAT, [1, 3]),
        _tensor("rank3", TensorProto.FLOAT, [1, 2, 3]),
        _tensor("unused", TensorProto.FLOAT, [2, 3]),
    ]
    initializers = [
        _initializer("axes0", [0]),
        _initializer("axes1", [1]),
        _initializer("ends1", [1]),
        _initializer("gather_index", [1]),
        _initializer("k2", [2]),
        _initializer("repeats", [1, 2]),
        _initializer("shape23", [2, 3]),
        _initializer("split_sizes", [1, 1]),
        _initializer("starts0", [0]),
        _initializer("scale", [1.0, 1.0, 1.0], numpy.float32),
        _initializer("bias", [0.0, 0.0, 0.0], numpy.float32),
        _initializer("scalar", [1.0], numpy.float32),
    ]
    # GraphSurgeon only exposes initializers that participate in the imported
    # graph. Keep every fixture initializer reachable until the edit sequence
    # attaches it to the operator under test. Cleanup removes unused anchors.
    for initializer in initializers:
        if initializer.name != "scalar":
            nodes.append(
                helper.make_node(
                    "Identity",
                    [initializer.name],
                    [f"anchor_{initializer.name}"],
                    name=f"anchor_{initializer.name}",
                )
            )
    graph = helper.make_graph(
        nodes,
        "editor-workflow",
        inputs,
        [_tensor("clipped", TensorProto.FLOAT, [2, 3])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def _add_node(op, name, inputs, outputs, attributes=None):
    return {
        "kind": "add-node",
        "op": op,
        "name": name,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "attributes": dict(attributes or {}),
    }


def _interleave(core, blocks, seed):
    buckets = [[] for _ in range(len(core) + 1)]
    for index, block in enumerate(blocks):
        slot = (seed * (index + 3) + index * index + 1) % len(buckets)
        buckets[slot].extend(block)
    edits = []
    for index, edit in enumerate(core):
        edits.extend(buckets[index])
        edits.append(edit)
    edits.extend(buckets[-1])
    return edits


def _common_blocks(seed):
    optional_index = 1 + (seed % 2)
    replacement = ["x", "y", "base"][seed % 3]
    return [
        [
            {
                "kind": "add-input",
                "name": f"extra_{seed}",
                "dataType": "float32",
                "dimensions": [2, 3],
            }
        ],
        [{"kind": "rename-node", "nodeIndex": 0, "name": f"source_{seed}"}],
        [
            {
                "kind": "node-input",
                "nodeIndex": 1,
                "inputIndex": 0,
                "value": replacement,
            }
        ],
        [
            {
                "kind": "disconnect-input",
                "nodeIndex": 1,
                "inputIndex": optional_index,
            },
            {
                "kind": "node-input",
                "nodeIndex": 1,
                "inputIndex": optional_index,
                "value": "scalar",
            },
        ],
        [{"kind": "delete-node", "nodeIndex": 2}],
        [
            {
                "kind": "rename-graph-output",
                "outputIndex": 0,
                "name": f"original_output_{seed}",
            }
        ],
    ]


UNARY = [
    ("Identity", {}),
    ("Relu", {}),
    ("Sigmoid", {}),
    ("Tanh", {}),
    ("Erf", {}),
    ("Neg", {}),
    ("Exp", {}),
    ("Log", {}),
    ("Sqrt", {}),
    ("Reciprocal", {}),
    ("Softmax", {"axis": -1}),
]
BINARY = ["Add", "Sub", "Mul", "Div", "Max", "Min"]
TAILS = [
    ("Identity", {}),
    ("Transpose", {"perm": [1, 0]}),
    ("Flatten", {"axis": 1}),
    ("ReduceMean", {"axes": [-1], "keepdims": 1}),
    ("Cast", {"to": 10}),
]


def _chain_workflow(seed, unary_ops):
    current = "x"
    core = []
    for depth, (op, attributes) in enumerate(unary_ops):
        output = f"flow_{seed}_{depth}"
        core.append(
            _add_node(
                op, f"flow_{seed}_{op.lower()}_{depth}", [current], [output], attributes
            )
        )
        current = output
    binary = BINARY[seed % len(BINARY)]
    branch = f"flow_{seed}_branch"
    core.append(
        _add_node(binary, f"flow_{seed}_{binary.lower()}", [current, "y"], [branch])
    )
    tail, attributes = TAILS[(seed * 3) % len(TAILS)]
    final = f"flow_{seed}_final"
    core.append(
        _add_node(tail, f"flow_{seed}_{tail.lower()}", [branch], [final], attributes)
    )
    core.append(
        {"kind": "add-output", "name": f"workflow_output_{seed}", "value": final}
    )
    return _interleave(core, _common_blocks(seed), seed)


SPECIAL_OPERATORS = [
    ("TopK", ["x", "k2"], ["values", "special"], {"axis": -1}),
    ("Split", ["x", "split_sizes"], ["special", "split_right"], {"axis": 0}),
    ("Squeeze", ["rank3", "axes0"], ["special"], {}),
    ("Unsqueeze", ["x", "axes0"], ["special"], {}),
    ("Expand", ["small", "shape23"], ["special"], {}),
    ("Slice", ["x", "starts0", "ends1", "axes0"], ["special"], {}),
    ("Gather", ["x", "gather_index"], ["special"], {"axis": 1}),
    ("GatherElements", ["x", "indices"], ["special"], {"axis": 1}),
    ("Tile", ["x", "repeats"], ["special"], {}),
    ("Reshape", ["x", "shape23"], ["special"], {}),
    ("Shape", ["x"], ["special"], {}),
    ("ArgMax", ["x"], ["special"], {"axis": 1, "keepdims": 1}),
    ("ArgMin", ["x"], ["special"], {"axis": 1, "keepdims": 0}),
    ("MatMul", ["x", "matrix"], ["special"], {}),
    ("Where", ["condition", "x", "y"], ["special"], {}),
    ("Equal", ["x", "y"], ["special"], {}),
    (
        "LayerNormalization",
        ["x", "scale", "bias"],
        ["special"],
        {"axis": -1, "epsilon": 0.00001},
    ),
    ("Concat", ["x", "y"], ["special"], {"axis": 0}),
    ("ReduceSum", ["x", "axes1"], ["special"], {"keepdims": 1}),
    ("Cast", ["x"], ["special"], {"to": 10}),
]


def _special_workflow(seed, specification):
    op, inputs, outputs, attributes = specification
    prefix = f"special_{seed}"
    renamed_outputs = [f"{prefix}_{name}" for name in outputs]
    exposed = renamed_outputs[-1] if op == "TopK" else renamed_outputs[0]
    core = [
        _add_node("Relu", f"{prefix}_relu", ["x"], [f"{prefix}_relu_out"]),
        {
            "kind": "add-output",
            "name": f"{prefix}_branch_output",
            "value": f"{prefix}_relu_out",
        },
        _add_node(op, f"{prefix}_{op.lower()}", inputs, renamed_outputs, attributes),
        _add_node("Identity", f"{prefix}_identity", [exposed], [f"{prefix}_final"]),
        {"kind": "add-output", "name": f"{prefix}_output", "value": f"{prefix}_final"},
    ]
    return _interleave(core, _common_blocks(seed), seed)


INVALID_FINALS = [
    ({"kind": "unknown"}, "Unsupported"),
    (
        {"kind": "add-input", "name": "x", "dataType": "float32", "dimensions": [1]},
        "already in use",
    ),
    (
        {"kind": "add-input", "name": "bad", "dataType": "mystery", "dimensions": [1]},
        "Unsupported",
    ),
    (
        {"kind": "add-node", "op": "Relu", "inputs": ["missing"], "outputs": ["bad"]},
        "was not found",
    ),
    (
        {"kind": "add-node", "op": "Relu", "inputs": ["x"], "outputs": ["x"]},
        "already in use",
    ),
    ({"kind": "add-output", "name": "bad", "value": "missing"}, "source tensor"),
    (
        {"kind": "node-input", "nodeIndex": 99, "inputIndex": 0, "value": "x"},
        "node index",
    ),
    (
        {"kind": "disconnect-input", "nodeIndex": 0, "inputIndex": 0},
        "Cannot disconnect required",
    ),
    (
        {"kind": "rename-output", "nodeIndex": 0, "outputIndex": 0, "name": "x"},
        "already in use",
    ),
    ({"kind": "delete-output", "outputIndex": 99}, "graph output index"),
]


def _assert_valid_workflow(test, edits):
    with tempfile.TemporaryDirectory() as directory:
        source = _source_model(directory)
        target = os.path.join(directory, "edited.onnx")
        BACKEND.apply_edits(source, target, edits)
        model = onnx.load(target)
        onnx.checker.check_model(model)
        result = BACKEND.infer_shapes(source, edits)
        test.assertNotIn("error", result, result.get("error"))
        test.assertGreaterEqual(len(edits), 6)


def _valid_case(edits):
    def run(test):
        _assert_valid_workflow(test, edits)

    return run


def _invalid_workflow(seed, invalid):
    unary_ops = [UNARY[seed % len(UNARY)], UNARY[(seed + 3) % len(UNARY)]]
    edits = _chain_workflow(seed, unary_ops)
    insertion = 2 + (seed % (len(edits) - 2))
    edits.insert(insertion, invalid)
    return edits


def _invalid_case(edits, message):
    def run(test):
        with tempfile.TemporaryDirectory() as directory:
            source = _source_model(directory)
            with test.assertRaises((TypeError, ValueError)) as context:
                BACKEND._apply_graph_edits(source, edits)
            test.assertIn(message, str(context.exception))
            test.assertGreaterEqual(len(edits), 7)

    return run


class EditorWorkflowUseCaseTest(unittest.TestCase):
    pass


CASES = []
WORKFLOW_FINGERPRINTS = set()
for seed, unary_ops in enumerate(
    itertools.islice(itertools.permutations(UNARY, 3), 70), start=1
):
    edits = _chain_workflow(seed, unary_ops)
    CASES.append((f"mixed_chain_{seed:02d}", _valid_case(edits)))
    WORKFLOW_FINGERPRINTS.add(json.dumps(edits, sort_keys=True))
for offset, specification in enumerate(SPECIAL_OPERATORS, start=71):
    edits = _special_workflow(offset, specification)
    CASES.append(
        (
            f"structured_{offset:02d}_{specification[0].lower()}",
            _valid_case(edits),
        )
    )
    WORKFLOW_FINGERPRINTS.add(json.dumps(edits, sort_keys=True))
for offset, (invalid, message) in enumerate(INVALID_FINALS, start=91):
    edits = _invalid_workflow(offset, invalid)
    CASES.append((f"rejected_sequence_{offset:02d}", _invalid_case(edits, message)))
    WORKFLOW_FINGERPRINTS.add(json.dumps(edits, sort_keys=True))

if len(CASES) != 100:
    raise RuntimeError(f"Expected exactly 100 workflow use cases, found {len(CASES)}.")

if len(WORKFLOW_FINGERPRINTS) != 100:
    raise RuntimeError("All 100 workflow edit sequences must be structurally unique.")

for number, (name, function) in enumerate(CASES, start=1):
    setattr(EditorWorkflowUseCaseTest, f"test_{number:03d}_{name}", function)


if __name__ == "__main__":
    unittest.main(verbosity=2)
