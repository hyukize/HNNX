import importlib.util
import os
import tempfile
import unittest

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(ROOT, "source", "onnx-graphsurgeon.py")
SPEC = importlib.util.spec_from_file_location("netron_onnx_graphsurgeon", BACKEND_PATH)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


class OnnxGraphSurgeonBackendTest(unittest.TestCase):
    def test_adds_graph_input_node_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "additions.onnx")
            target = os.path.join(directory, "additions.output.onnx")
            node = helper.make_node("Identity", ["x"], ["y"], name="identity")
            graph = helper.make_graph(
                [node],
                "additions",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 13)],
                ),
                source,
            )

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "add-input",
                        "name": "extra",
                        "dataType": "float32",
                        "dimensions": [1, 4],
                    },
                    {
                        "kind": "add-node",
                        "op": "Add",
                        "name": "add_created",
                        "inputs": ["x", "extra"],
                        "outputs": ["sum"],
                        "attributes": {},
                    },
                    {
                        "kind": "add-output",
                        "name": "sum_output",
                        "value": "sum",
                    },
                ],
            )

            output = onnx.load(target)
            self.assertIn("extra", [value.name for value in output.graph.input])
            nodes = {node.name: node for node in output.graph.node}
            self.assertEqual(nodes["add_created"].op_type, "Add")
            self.assertEqual(list(nodes["add_created"].input), ["x", "extra"])
            self.assertEqual(list(nodes["add_created"].output), ["sum"])
            self.assertIn("sum_output", [value.name for value in output.graph.output])
            boundary = next(
                node
                for node in output.graph.node
                if node.name.startswith("NetronGraphOutput_")
            )
            self.assertEqual(list(boundary.input), ["sum"])
            self.assertEqual(list(boundary.output), ["sum_output"])
            onnx.checker.check_model(output)

    def test_rewires_node_input_and_graph_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "editable.onnx")
            target = os.path.join(directory, "editable.output.onnx")
            nodes = [
                helper.make_node("Abs", ["x"], ["a"], name="abs"),
                helper.make_node("Neg", ["alt"], ["b"], name="neg"),
                helper.make_node("Add", ["a", "b"], ["y"], name="add"),
            ]
            graph = helper.make_graph(
                nodes,
                "editable",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("alt", TensorProto.FLOAT, [1]),
                ],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            onnx.save(helper.make_model(graph), source)

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "node-input",
                        "nodeIndex": 1,
                        "inputIndex": 0,
                        "value": "a",
                    },
                    {"kind": "graph-output", "outputIndex": 0, "value": "b"},
                ],
            )

            output = onnx.load(target)
            nodes_by_name = {node.name: node for node in output.graph.node}
            self.assertEqual(list(nodes_by_name["neg"].input), ["a"])
            self.assertEqual(output.graph.output[0].name, "y")
            identities = [
                node for node in output.graph.node if node.op_type == "Identity"
            ]
            self.assertEqual(len(identities), 1)
            self.assertEqual(list(identities[0].input), ["b"])
            self.assertEqual(list(identities[0].output), ["y"])
            onnx.checker.check_model(output)

    def test_preserves_external_initializer_reference(self):
        with tempfile.TemporaryDirectory() as source_directory:
            source = os.path.join(source_directory, "external.onnx")
            target = os.path.join(source_directory, "external.edited.onnx")
            weight = numpy_helper.from_array(
                numpy.asarray([2.0], dtype=numpy.float32), "weight"
            )
            node = helper.make_node("Add", ["x", "weight"], ["y"], name="add")
            graph = helper.make_graph(
                [node],
                "external",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                [weight],
            )
            model = helper.make_model(graph)
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="external.data",
                size_threshold=0,
            )

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "node-input",
                        "nodeIndex": 0,
                        "inputIndex": 0,
                        "value": "x",
                    },
                ],
            )

            output = onnx.load(target, load_external_data=False)
            entries = {
                entry.key: entry.value
                for entry in output.graph.initializer[0].external_data
            }
            referenced = os.path.normpath(
                os.path.join(source_directory, entries["location"])
            )
            self.assertEqual(
                referenced, os.path.join(source_directory, "external.data")
            )
            onnx.checker.check_model(target)
            inferred = BACKEND.infer_shapes(source, [])
            self.assertTrue(
                any(tensor["name"] == "y" for tensor in inferred["tensors"])
            )

    def test_infers_without_missing_external_tensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "missing-external.onnx")
            data = os.path.join(directory, "missing-external.data")
            weight = numpy_helper.from_array(
                numpy.asarray([2.0], dtype=numpy.float32), "weight"
            )
            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "weight"], ["y"], name="add")],
                "missing-external",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
                [weight],
            )
            onnx.save_model(
                helper.make_model(graph),
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=os.path.basename(data),
                size_threshold=0,
            )
            os.remove(data)

            result = BACKEND.infer_shapes(source, [])

            self.assertNotIn("error", result)
            tensors = {tensor["name"]: tensor for tensor in result["tensors"]}
            self.assertEqual(tensors["y"]["dimensions"], [1])
            self.assertEqual(result["warnings"][0]["code"], "missing-external-data")
            self.assertEqual(
                result["warnings"][0]["files"], ["missing-external.data"]
            )

    def test_disconnects_only_optional_input_without_shifting_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "optional.onnx")
            target = os.path.join(directory, "optional.output.onnx")
            node = helper.make_node(
                "Clip", ["x", "minimum", "maximum"], ["y"], name="clip"
            )
            graph = helper.make_graph(
                [node],
                "optional",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("minimum", TensorProto.FLOAT, []),
                    helper.make_tensor_value_info("maximum", TensorProto.FLOAT, []),
                ],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 13)],
            )
            onnx.save(model, source)

            BACKEND.apply_edits(
                source,
                target,
                [
                    {"kind": "disconnect-input", "nodeIndex": 0, "inputIndex": 1},
                ],
            )

            output = onnx.load(target)
            self.assertEqual(list(output.graph.node[0].input), ["x", "", "maximum"])
            onnx.checker.check_model(output)

            with self.assertRaisesRegex(ValueError, "required input"):
                BACKEND.apply_edits(
                    source,
                    target,
                    [
                        {"kind": "disconnect-input", "nodeIndex": 0, "inputIndex": 0},
                    ],
                )

    def test_reconnects_an_omitted_optional_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "optional.onnx")
            target = os.path.join(directory, "optional.output.onnx")
            nodes = [
                helper.make_node("Abs", ["x"], ["limit"], name="abs"),
                helper.make_node("Clip", ["x"], ["y"], name="clip"),
            ]
            graph = helper.make_graph(
                nodes,
                "optional",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 13)],
                ),
                source,
            )

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "node-input",
                        "nodeIndex": 1,
                        "inputIndex": 2,
                        "value": "limit",
                    },
                ],
            )

            output = onnx.load(target)
            nodes_by_name = {node.name: node for node in output.graph.node}
            self.assertEqual(list(nodes_by_name["clip"].input), ["x", "", "limit"])
            onnx.checker.check_model(output)

    def test_deletes_an_unused_leaf_node(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "leaf.onnx")
            target = os.path.join(directory, "leaf.output.onnx")
            nodes = [
                helper.make_node("Abs", ["x"], ["y"], name="kept"),
                helper.make_node("Neg", ["x"], ["unused"], name="leaf"),
            ]
            graph = helper.make_graph(
                nodes,
                "leaf",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            onnx.save(helper.make_model(graph), source)

            BACKEND.apply_edits(
                source,
                target,
                [
                    {"kind": "delete-node", "nodeIndex": 1},
                ],
            )

            output = onnx.load(target)
            self.assertEqual([node.name for node in output.graph.node], ["kept"])
            onnx.checker.check_model(output)

    def test_renames_node_and_output_tensor_references(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "rename.onnx")
            target = os.path.join(directory, "rename.output.onnx")
            nodes = [
                helper.make_node("Abs", ["x"], ["intermediate"], name="old_node"),
                helper.make_node("Neg", ["intermediate"], ["y"], name="consumer"),
            ]
            graph = helper.make_graph(
                nodes,
                "rename",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            onnx.save(helper.make_model(graph), source)

            BACKEND.apply_edits(
                source,
                target,
                [
                    {"kind": "rename-node", "nodeIndex": 0, "name": "renamed_node"},
                    {
                        "kind": "rename-output",
                        "nodeIndex": 0,
                        "outputIndex": 0,
                        "name": "renamed_tensor",
                    },
                    {
                        "kind": "rename-graph-output",
                        "outputIndex": 0,
                        "name": "final_output",
                    },
                ],
            )

            output = onnx.load(target)
            self.assertEqual(output.graph.node[0].name, "renamed_node")
            self.assertEqual(list(output.graph.node[0].output), ["renamed_tensor"])
            self.assertEqual(list(output.graph.node[1].input), ["renamed_tensor"])
            self.assertEqual(list(output.graph.node[1].output), ["final_output"])
            self.assertEqual(output.graph.output[0].name, "final_output")
            onnx.checker.check_model(output)

    def test_infers_shapes_after_applying_unsaved_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "infer.onnx")
            nodes = [
                helper.make_node("Abs", ["x"], ["a"], name="abs"),
                helper.make_node("Neg", ["alternative"], ["b"], name="neg"),
            ]
            graph = helper.make_graph(
                nodes,
                "infer",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                    helper.make_tensor_value_info(
                        "alternative", TensorProto.FLOAT, [4, 5]
                    ),
                ],
                [helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, None])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 13)],
                ),
                source,
            )

            result = BACKEND.infer_shapes(
                source,
                [
                    {
                        "kind": "node-input",
                        "nodeIndex": 1,
                        "inputIndex": 0,
                        "value": "a",
                    },
                ],
            )

            tensors = {tensor["name"]: tensor for tensor in result["tensors"]}
            self.assertEqual(tensors["a"]["dimensions"], [2, 3])
            self.assertEqual(tensors["b"]["dimensions"], [2, 3])
            self.assertEqual(tensors["b"]["dataType"], "float32")

    def test_reports_shape_inference_failure_with_node_and_tensor_details(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "shape-mismatch.onnx")
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "Add", ["left", "right"], ["output"], name="bad_add"
                    )
                ],
                "shape-mismatch",
                [
                    helper.make_tensor_value_info("left", TensorProto.FLOAT, [2, 3]),
                    helper.make_tensor_value_info("right", TensorProto.FLOAT, [2, 4]),
                ],
                [
                    helper.make_tensor_value_info(
                        "output", TensorProto.FLOAT, [None, None]
                    )
                ],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 13)],
                ),
                source,
            )

            result = BACKEND.infer_shapes(source, [])

            diagnostic = result["error"]
            self.assertEqual(diagnostic["node"]["name"], "bad_add")
            self.assertEqual(diagnostic["node"]["opType"], "Add")
            self.assertIn("cannot be broadcast", diagnostic["summary"])
            self.assertEqual(
                [value["dimensions"] for value in diagnostic["inputs"]],
                [[2, 3], [2, 4]],
            )
            self.assertIn("ShapeInferenceError", diagnostic["message"])

    def test_adds_opset17_split_with_multiple_outputs_and_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "split.onnx")
            target = os.path.join(directory, "split.output.onnx")
            graph = helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"], name="identity")],
                "split",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "add-node",
                        "op": "Split",
                        "name": "split_created",
                        "inputs": ["x", "split_created_split_sizes"],
                        "outputs": ["split_left", "split_right"],
                        "attributes": {"axis": 1},
                        "initializers": [
                            {
                                "name": "split_created_split_sizes",
                                "dataType": "int64",
                                "dimensions": [2],
                                "values": [2, 4],
                            }
                        ],
                    }
                ],
            )

            output = onnx.load(target)
            split = next(
                node for node in output.graph.node if node.name == "split_created"
            )
            self.assertEqual(split.op_type, "Split")
            self.assertEqual(
                list(split.input), ["x", "split_created_split_sizes"]
            )
            self.assertEqual(list(split.output), ["split_left", "split_right"])
            sizes = next(
                value
                for value in output.graph.initializer
                if value.name == "split_created_split_sizes"
            )
            self.assertEqual(numpy_helper.to_array(sizes).tolist(), [2, 4])
            onnx.checker.check_model(output)

            inferred = BACKEND.infer_shapes(
                source,
                [
                    {
                        "kind": "add-node",
                        "op": "Split",
                        "name": "split_created",
                        "inputs": ["x", "split_created_split_sizes"],
                        "outputs": ["split_left", "split_right"],
                        "attributes": {"axis": 1},
                        "initializers": [
                            {
                                "name": "split_created_split_sizes",
                                "dataType": "int64",
                                "dimensions": [2],
                                "values": [2, 4],
                            }
                        ],
                    }
                ],
            )
            tensors = {tensor["name"]: tensor for tensor in inferred["tensors"]}
            self.assertEqual(tensors["split_left"]["dimensions"], [1, 2])
            self.assertEqual(tensors["split_right"]["dimensions"], [1, 4])

    def test_adds_opset17_equal_split_without_a_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "equal-split.onnx")
            target = os.path.join(directory, "equal-split.output.onnx")
            graph = helper.make_graph(
                [],
                "equal-split",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])],
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )

            BACKEND.apply_edits(
                source,
                target,
                [
                    {
                        "kind": "add-node",
                        "op": "Split",
                        "name": "equal_split",
                        "inputs": ["x"],
                        "outputs": ["left", "right"],
                        "attributes": {"axis": 1},
                    }
                ],
            )

            output = onnx.load(target)
            split = next(
                node for node in output.graph.node if node.name == "equal_split"
            )
            self.assertEqual(list(split.input), ["x"])
            self.assertEqual(list(split.output), ["left", "right"])
            self.assertEqual(len(output.graph.initializer), 0)
            onnx.checker.check_model(output)

    def test_adds_opset17_llm_shape_routing_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "routing.onnx")
            target = os.path.join(directory, "routing.output.onnx")
            graph = helper.make_graph(
                [],
                "routing",
                [
                    helper.make_tensor_value_info(
                        "x", TensorProto.FLOAT, [1, 2, 1, 4]
                    ),
                    helper.make_tensor_value_info(
                        "small", TensorProto.FLOAT, [1, 1, 4]
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "x", TensorProto.FLOAT, [1, 2, 1, 4]
                    )
                ],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )

            def int64_initializer(name, values):
                return {
                    "name": name,
                    "dataType": "int64",
                    "dimensions": [len(values)],
                    "values": values,
                }

            edits = [
                {
                    "kind": "add-node",
                    "op": "Squeeze",
                    "name": "squeeze",
                    "inputs": ["x", "squeeze_axes"],
                    "outputs": ["squeezed"],
                    "attributes": {},
                    "initializers": [int64_initializer("squeeze_axes", [2])],
                },
                {
                    "kind": "add-node",
                    "op": "Unsqueeze",
                    "name": "unsqueeze",
                    "inputs": ["x", "unsqueeze_axes"],
                    "outputs": ["unsqueezed"],
                    "attributes": {},
                    "initializers": [int64_initializer("unsqueeze_axes", [-1])],
                },
                {
                    "kind": "add-node",
                    "op": "Expand",
                    "name": "expand",
                    "inputs": ["small", "expand_shape"],
                    "outputs": ["expanded"],
                    "attributes": {},
                    "initializers": [
                        int64_initializer("expand_shape", [1, 2, 1, 4])
                    ],
                },
                {
                    "kind": "add-node",
                    "op": "Slice",
                    "name": "slice",
                    "inputs": ["x", "slice_starts", "slice_ends", "slice_axes"],
                    "outputs": ["sliced"],
                    "attributes": {},
                    "initializers": [
                        int64_initializer("slice_starts", [0]),
                        int64_initializer("slice_ends", [1]),
                        int64_initializer("slice_axes", [1]),
                    ],
                },
            ]
            BACKEND.apply_edits(source, target, edits)
            onnx.checker.check_model(target)

            inferred = BACKEND.infer_shapes(source, edits)
            tensors = {tensor["name"]: tensor for tensor in inferred["tensors"]}
            self.assertEqual(tensors["squeezed"]["dimensions"], [1, 2, 4])
            self.assertEqual(tensors["unsqueezed"]["dimensions"], [1, 2, 1, 4, 1])
            self.assertEqual(tensors["expanded"]["dimensions"], [1, 2, 1, 4])
            self.assertEqual(tensors["sliced"]["dimensions"], [1, 1, 1, 4])

    def test_adds_opset17_reduce_sum_with_axes_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "reduce-sum.onnx")
            target = os.path.join(directory, "reduce-sum.output.onnx")
            graph = helper.make_graph(
                [],
                "reduce-sum",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )
            edit = {
                "kind": "add-node",
                "op": "ReduceSum",
                "name": "reduce_last_axis",
                "inputs": ["x", "reduce_axes"],
                "outputs": ["reduced"],
                "attributes": {"keepdims": 1},
                "initializers": [
                    {
                        "name": "reduce_axes",
                        "dataType": "int64",
                        "dimensions": [1],
                        "values": [-1],
                    }
                ],
            }
            BACKEND.apply_edits(source, target, [edit])
            output = onnx.load(target)
            node = next(
                node for node in output.graph.node if node.name == "reduce_last_axis"
            )
            self.assertEqual(list(node.input), ["x", "reduce_axes"])
            self.assertEqual(node.attribute[0].name, "keepdims")
            self.assertNotIn("axes", [attribute.name for attribute in node.attribute])
            onnx.checker.check_model(output)

            inferred = BACKEND.infer_shapes(source, [edit])
            tensors = {tensor["name"]: tensor for tensor in inferred["tensors"]}
            self.assertEqual(tensors["reduced"]["dimensions"], [2, 1])

    def test_adds_opset17_topk_with_typed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "topk.onnx")
            target = os.path.join(directory, "topk.output.onnx")
            graph = helper.make_graph(
                [],
                "topk",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )
            edit = {
                "kind": "add-node",
                "op": "TopK",
                "name": "topk_created",
                "inputs": ["x", "topk_created_k"],
                "outputs": ["top_values", "top_indices"],
                "attributes": {"axis": -1, "largest": 1, "sorted": 1},
                "initializers": [
                    {
                        "name": "topk_created_k",
                        "dataType": "int64",
                        "dimensions": [1],
                        "values": [3],
                    }
                ],
            }
            BACKEND.apply_edits(source, target, [edit])
            output = onnx.load(target)
            onnx.checker.check_model(output)
            topk = next(
                node for node in output.graph.node if node.name == "topk_created"
            )
            self.assertEqual(list(topk.input), ["x", "topk_created_k"])
            self.assertEqual(list(topk.output), ["top_values", "top_indices"])
            k = next(
                value
                for value in output.graph.initializer
                if value.name == "topk_created_k"
            )
            self.assertEqual(numpy_helper.to_array(k).tolist(), [3])

            inferred = BACKEND.infer_shapes(source, [edit])
            tensors = {tensor["name"]: tensor for tensor in inferred["tensors"]}
            self.assertEqual(tensors["top_values"]["dimensions"], [1, 3])
            self.assertEqual(tensors["top_values"]["dataType"], "float32")
            self.assertEqual(tensors["top_indices"]["dimensions"], [1, 3])
            self.assertEqual(tensors["top_indices"]["dataType"], "int64")

    def test_deleting_unrelated_cast_preserves_topk_output_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "topk-delete-cast.onnx")
            k = helper.make_tensor("k", TensorProto.INT64, [1], [2])
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "Cast", ["x"], ["unused_cast"], name="unrelated_cast", to=10
                    ),
                    helper.make_node(
                        "TopK", ["x", "k"], ["values", "indices"], name="topk"
                    ),
                    helper.make_node(
                        "Identity", ["indices"], ["result"], name="use_indices"
                    ),
                ],
                "topk-delete-cast",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("result", TensorProto.INT64, [1, 2])],
                initializer=[k],
            )
            onnx.save(
                helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 17)],
                ),
                source,
            )

            result = BACKEND.infer_shapes(
                source, [{"kind": "delete-node", "nodeIndex": 0}]
            )
            self.assertNotIn("error", result)
            _, edited = BACKEND._apply_graph_edits(
                source, [{"kind": "delete-node", "nodeIndex": 0}]
            )
            topk = next(node for node in edited.graph.node if node.name == "topk")
            self.assertEqual(list(topk.output), ["values", "indices"])
            onnx.checker.check_model(edited)


if __name__ == "__main__":
    unittest.main()
