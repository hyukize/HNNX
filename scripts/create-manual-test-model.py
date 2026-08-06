#!/usr/bin/env python3

import os

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "test", "manual-editor.onnx")


def main():
    weight = numpy_helper.from_array(
        numpy.asarray([0.5], dtype=numpy.float32), "weight"
    )
    nodes = [
        helper.make_node("Abs", ["x"], ["a"], name="abs"),
        helper.make_node(
            "Transpose", ["a"], ["transposed"], name="transpose_in", perm=[1, 0]
        ),
        helper.make_node(
            "Split",
            ["transposed"],
            ["split_0", "split_1"],
            name="split",
            axis=0,
            split=[2, 2],
        ),
        helper.make_node(
            "Concat", ["split_0", "split_1"], ["merged"], name="concat", axis=0
        ),
        helper.make_node(
            "Transpose", ["merged"], ["pre_output"], name="transpose_out", perm=[1, 0]
        ),
        helper.make_node("Mul", ["pre_output", "weight"], ["y"], name="mul"),
        helper.make_node("Neg", ["alternative"], ["b"], name="neg"),
        helper.make_node("Clip", ["a", "", "maximum"], ["clipped"], name="clip"),
    ]
    graph = helper.make_graph(
        nodes,
        "hnnx-manual-editor",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("alternative", TensorProto.FLOAT, [3, 5]),
            helper.make_tensor_value_info("maximum", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("clipped", TensorProto.FLOAT, [2, 4]),
        ],
        [weight],
    )
    model = helper.make_model(
        graph,
        producer_name="HNNX manual test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
