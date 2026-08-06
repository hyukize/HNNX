#!/usr/bin/env python3

"""Create a compact Opset 17 model that exercises HNNX's main workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(7)
    initializers = [
        numpy_helper.from_array(
            rng.normal(0, 0.12, (4, 4, 1, 1)).astype(np.float32), "conv.weight"
        ),
        numpy_helper.from_array(np.zeros((4,), dtype=np.float32), "conv.bias"),
        numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), "split.sizes"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "clip.min"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip.max"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "topk.k"),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["input", "conv.weight", "conv.bias"], ["conv_out"], name="conv_1x1"
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu"),
        helper.make_node(
            "Split",
            ["relu_out", "split.sizes"],
            ["part_0", "part_1", "part_2", "part_3"],
            name="split_channels",
            axis=1,
        ),
        helper.make_node(
            "Concat",
            ["part_0", "part_1", "part_2", "part_3"],
            ["merged"],
            name="concat_channels",
            axis=1,
        ),
        helper.make_node("Mul", ["merged", "scale"], ["scaled"], name="scale_half"),
        helper.make_node("Add", ["scaled", "input"], ["residual"], name="residual_add"),
        helper.make_node(
            "Clip", ["residual", "clip.min", "clip.max"], ["clipped"], name="clip"
        ),
        helper.make_node(
            "Cast",
            ["clipped"],
            ["cast_out"],
            name="removable_cast",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "TopK",
            ["cast_out", "topk.k"],
            ["top_values", "top_indices"],
            name="topk",
            axis=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "hnnx_showcase",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
        [
            helper.make_tensor_value_info(
                "top_values", TensorProto.FLOAT, [1, 2, 8, 8]
            ),
            helper.make_tensor_value_info(
                "top_indices", TensorProto.INT64, [1, 2, 8, 8]
            ),
        ],
        initializer=initializers,
        doc_string=(
            "HNNX showcase: quantization overlays, Split/Concat bundles, "
            "residual fan-out, editable Cast removal, TopK type handling, "
            "graph outputs, and Opset 17 shape inference."
        ),
    )
    model = helper.make_model(
        graph,
        producer_name="HNNX",
        producer_version="0.1.7",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    checker.check_model(model)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    checker.check_model(inferred)
    return inferred


def build_encodings() -> dict[str, object]:
    activation = {
        "input": "uint8",
        "conv_out": "int16",
        "relu_out": "uint8",
        "part_0": "uint8",
        "part_1": "uint8",
        "part_2": "uint8",
        "part_3": "uint8",
        "merged": "uint8",
        "residual": "int16",
        "clipped": "int16",
        "cast_out": "int16",
        "top_values": "int16",
    }
    return {
        "version": "2.0.0",
        "activation_encodings": [
            {
                "name": name,
                "output_dtype": data_type,
                "y_scale": 0.03125 if "8" in data_type else 0.0009765625,
                "y_zero_point": 7 if data_type == "uint8" else 0,
            }
            for name, data_type in activation.items()
        ],
        "param_encodings": [
            {
                "name": "conv.weight",
                "output_dtype": "int8",
                "y_scale": [0.0038, 0.0041, 0.0036, 0.0040],
                "y_zero_point": [0, 0, 0, 0],
                "axis": 0,
            }
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "examples"
    output.mkdir(exist_ok=True)
    model_path = output / "hnnx-showcase.onnx"
    encodings_path = output / "hnnx-showcase.encodings"
    onnx.save_model(build_model(), model_path)
    encodings_path.write_text(
        json.dumps(build_encodings(), indent=2) + "\n", encoding="utf-8"
    )
    print(model_path)
    print(encodings_path)


if __name__ == "__main__":
    main()
