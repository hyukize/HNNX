import hashlib
import importlib.util
import json
import os
import random
import tempfile
import unittest

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(ROOT, "source", "onnx-graphsurgeon.py")
SPEC = importlib.util.spec_from_file_location("netron_editor_stage3", BACKEND_PATH)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


def _tensor(name, data_type=TensorProto.FLOAT, dimensions=None):
    return helper.make_tensor_value_info(name, data_type, dimensions or [2, 3])


def _source_model(directory, seed):
    path = os.path.join(directory, f"stage3-{seed}.onnx")
    limit = numpy_helper.from_array(numpy.asarray([1.0], dtype=numpy.float32), "limit")
    nodes = [
        helper.make_node("Identity", ["x"], ["a"], name="source"),
        helper.make_node("Relu", ["a"], ["r"], name="relu"),
        helper.make_node("Add", ["r", "y"], ["sum"], name="add"),
        helper.make_node("Clip", ["sum", "limit", "limit"], ["clipped"], name="clip"),
        helper.make_node("Identity", ["unused"], ["dead"], name="dead"),
    ]
    graph = helper.make_graph(
        nodes,
        "stage3-state-machine",
        [_tensor("x"), _tensor("y"), _tensor("alt"), _tensor("unused")],
        [_tensor("clipped")],
        [limit],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def _add_node(op, name, inputs, outputs, attributes=None):
    return {
        "kind": "add-node",
        "op": op,
        "name": name,
        "inputs": inputs,
        "outputs": outputs,
        "attributes": attributes or {},
    }


def _workflow(seed):
    rng = random.Random(seed)
    blocks = [
        [
            {
                "kind": "add-input",
                "name": f"extra_{seed}",
                "dataType": "float32",
                "dimensions": [2, 3],
            }
        ],
        [
            {"kind": "rename-node", "nodeIndex": 0, "name": f"source_{seed}_a"},
            {"kind": "rename-node", "nodeIndex": 0, "name": f"source_{seed}_b"},
        ],
        [
            {
                "kind": "node-input",
                "nodeIndex": 2,
                "inputIndex": 0,
                "value": rng.choice(["a", "r", "x", "alt"]),
            },
            {
                "kind": "node-input",
                "nodeIndex": 2,
                "inputIndex": 0,
                "value": rng.choice(["a", "r", "x", "alt"]),
            },
        ],
        [
            {"kind": "disconnect-input", "nodeIndex": 3, "inputIndex": 1},
            {
                "kind": "node-input",
                "nodeIndex": 3,
                "inputIndex": 1,
                "value": "limit",
            },
            {"kind": "disconnect-input", "nodeIndex": 3, "inputIndex": 2},
            {
                "kind": "node-input",
                "nodeIndex": 3,
                "inputIndex": 2,
                "value": "limit",
            },
        ],
        [
            {
                "kind": "graph-output",
                "outputIndex": 0,
                "value": rng.choice(["a", "r", "sum"]),
                "name": f"stage3_probe_{seed}",
            },
            {
                "kind": "graph-output",
                "outputIndex": 0,
                "value": "clipped",
                "name": f"stage3_probe_{seed}",
            },
            {
                "kind": "rename-graph-output",
                "outputIndex": 0,
                "name": f"stage3_original_{seed}",
            },
        ],
        [{"kind": "delete-node", "nodeIndex": 4}],
    ]
    rng.shuffle(blocks)
    edits = [edit for block in blocks for edit in block]
    current = "x"
    operators = rng.sample(["Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Exp", "Sqrt"], 4)
    for index, op in enumerate(operators):
        output = f"generated_{seed}_{index}"
        edits.append(
            _add_node(op, f"generated_{seed}_{op.lower()}_{index}", [current], [output])
        )
        current = output
    merged = f"generated_{seed}_merged"
    edits.append(
        _add_node(
            rng.choice(["Add", "Sub", "Mul", "Div"]),
            f"generated_{seed}_binary",
            [current, f"extra_{seed}"],
            [merged],
        )
    )
    final = f"generated_{seed}_final"
    edits.append(
        _add_node(
            "Transpose",
            f"generated_{seed}_transpose",
            [merged],
            [final],
            {"perm": [1, 0]},
        )
    )
    edits.append(
        {"kind": "add-output", "name": f"stage3_output_{seed}", "value": final}
    )
    return edits


def _attribute_signature(attribute):
    return (attribute.name, attribute.type, attribute.SerializeToString().hex())


def _model_signature(model):
    graph = model.graph
    return {
        "nodes": [
            (
                node.domain,
                node.op_type,
                node.name,
                tuple(node.input),
                tuple(node.output),
                tuple(sorted(_attribute_signature(value) for value in node.attribute)),
            )
            for node in graph.node
        ],
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "initializers": [
            (
                value.name,
                value.data_type,
                tuple(value.dims),
                hashlib.sha256(value.raw_data).hexdigest(),
            )
            for value in graph.initializer
        ],
    }


def _tensor_signature(result):
    return sorted(
        (
            tensor["name"],
            tensor["dataType"],
            tuple(tensor["dimensions"]),
        )
        for tensor in result.get("tensors", [])
    )


INVALID_EDITS = [
    ({"kind": "unknown-stage3"}, "Unsupported"),
    (
        {"kind": "node-input", "nodeIndex": 99, "inputIndex": 0, "value": "x"},
        "node index",
    ),
    (
        {"kind": "node-input", "nodeIndex": 1, "inputIndex": 0, "value": "missing"},
        "not found",
    ),
    ({"kind": "disconnect-input", "nodeIndex": 1, "inputIndex": 0}, "required input"),
    ({"kind": "delete-node", "nodeIndex": -1}, "node index"),
    ({"kind": "delete-input", "inputIndex": 88}, "input index"),
    ({"kind": "delete-output", "outputIndex": 88}, "output index"),
    ({"kind": "rename-node", "nodeIndex": 0, "name": ""}, "non-empty"),
    (
        {"kind": "rename-output", "nodeIndex": 0, "outputIndex": 0, "name": "x"},
        "already in use",
    ),
    ({"kind": "add-output", "name": "bad", "value": "missing"}, "source tensor"),
]


class EditorStage3StateMachineTest(unittest.TestCase):
    pass


def _valid_case(seed):
    def run(test):
        edits = _workflow(seed)
        test.assertGreaterEqual(len(edits), 20)
        with tempfile.TemporaryDirectory() as directory:
            source = _source_model(directory, seed)
            first = os.path.join(directory, "first.onnx")
            second = os.path.join(directory, "second.onnx")
            BACKEND.apply_edits(source, first, edits)
            BACKEND.apply_edits(source, second, edits)
            first_model = onnx.load(first)
            second_model = onnx.load(second)
            onnx.checker.check_model(first_model)
            onnx.checker.check_model(second_model)
            test.assertEqual(
                _model_signature(first_model), _model_signature(second_model)
            )

            inferred_from_edits = BACKEND.infer_shapes(source, edits)
            inferred_after_reload = BACKEND.infer_shapes(first, [])
            test.assertNotIn("error", inferred_from_edits)
            test.assertNotIn("error", inferred_after_reload)
            test.assertEqual(
                _tensor_signature(inferred_from_edits),
                _tensor_signature(inferred_after_reload),
            )

            undo_depth = 1 + seed % 7
            prefix = edits[:-undo_depth]
            prefix_target = os.path.join(directory, "prefix.onnx")
            BACKEND.apply_edits(source, prefix_target, prefix)
            onnx.checker.check_model(prefix_target)
            replay = os.path.join(directory, "replay.onnx")
            BACKEND.apply_edits(source, replay, edits)
            test.assertEqual(
                _model_signature(first_model), _model_signature(onnx.load(replay))
            )

    return run


def _recovery_case(seed):
    def run(test):
        edits = _workflow(seed)
        invalid, message = INVALID_EDITS[seed % len(INVALID_EDITS)]
        insertion = 1 + seed % (len(edits) - 1)
        broken = edits[:insertion] + [invalid] + edits[insertion:]
        with tempfile.TemporaryDirectory() as directory:
            source = _source_model(directory, seed)
            target = os.path.join(directory, "atomic.onnx")
            sentinel = f"stage3-sentinel-{seed}".encode()
            with open(target, "wb") as stream:
                stream.write(sentinel)
            with test.assertRaisesRegex((TypeError, ValueError), message):
                BACKEND.apply_edits(source, target, broken)
            with open(target, "rb") as stream:
                test.assertEqual(stream.read(), sentinel)

            recovered = os.path.join(directory, "recovered.onnx")
            BACKEND.apply_edits(source, recovered, edits)
            onnx.checker.check_model(recovered)

    return run


WORKFLOW_FINGERPRINTS = set()
for case_number in range(1, 81):
    edits = _workflow(case_number)
    WORKFLOW_FINGERPRINTS.add(json.dumps(edits, sort_keys=True))
    setattr(
        EditorStage3StateMachineTest,
        f"test_{case_number:03d}_deterministic_roundtrip",
        _valid_case(case_number),
    )
for case_number in range(81, 101):
    edits = _workflow(case_number)
    invalid, _ = INVALID_EDITS[case_number % len(INVALID_EDITS)]
    WORKFLOW_FINGERPRINTS.add(
        json.dumps({"edits": edits, "invalid": invalid}, sort_keys=True)
    )
    setattr(
        EditorStage3StateMachineTest,
        f"test_{case_number:03d}_atomic_failure_recovery",
        _recovery_case(case_number),
    )

if len(WORKFLOW_FINGERPRINTS) != 100:
    raise RuntimeError("Stage 3 requires 100 structurally unique state-machine cases.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
