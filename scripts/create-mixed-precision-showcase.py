#!/usr/bin/env python3

"""Create a screenshot-friendly Opset 17 mixed-precision HNNX showcase."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(18)
    initializers = [
        numpy_helper.from_array(
            rng.normal(0, 0.18, (8, 8)).astype(np.float32), "low.weight"
        ),
        numpy_helper.from_array(
            rng.normal(0, 0.12, (8, 8)).astype(np.float32), "mid.weight"
        ),
        numpy_helper.from_array(
            rng.normal(0, 0.08, (8, 8)).astype(np.float32), "high.weight"
        ),
        numpy_helper.from_array(
            np.ones((8,), dtype=np.int64), "split.sizes"
        ),
    ]
    split_outputs = [f"mid_part_{index}" for index in range(8)]
    nodes = [
        helper.make_node(
            "MatMul", ["input", "low.weight"], ["low_proj"], name="project_a4"
        ),
        helper.make_node("Relu", ["low_proj"], ["low_relu"], name="activation_a4"),
        helper.make_node(
            "MatMul", ["input", "mid.weight"], ["mid_proj"], name="project_a8"
        ),
        helper.make_node(
            "Split",
            ["mid_proj", "split.sizes"],
            split_outputs,
            name="split_a8_eight_way",
            axis=1,
        ),
        helper.make_node(
            "Concat",
            split_outputs,
            ["mid_roundtrip"],
            name="concat_a8_eight_way",
            axis=1,
        ),
        helper.make_node(
            "Sigmoid", ["mid_roundtrip"], ["mid_gate"], name="activation_a8"
        ),
        helper.make_node(
            "MatMul", ["input", "high.weight"], ["high_proj"], name="project_a16"
        ),
        helper.make_node(
            "Tanh", ["high_proj"], ["high_gate"], name="activation_a16"
        ),
        helper.make_node(
            "Add", ["low_relu", "mid_gate"], ["low_mid_mix"], name="mix_a4_a8"
        ),
        helper.make_node(
            "Add",
            ["mid_gate", "high_gate"],
            ["mid_high_mix"],
            name="mix_a8_a16",
        ),
        helper.make_node(
            "Concat",
            ["low_mid_mix", "mid_high_mix"],
            ["mixed_concat"],
            name="concat_mixed_precision",
            axis=1,
        ),
        helper.make_node(
            "Softmax", ["mixed_concat"], ["probabilities"], name="output_a16", axis=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "hnnx_mixed_precision_showcase",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [1, 16]
            )
        ],
        initializer=initializers,
        doc_string=(
            "Screenshot-ready HNNX graph with explicit A4, A8 and A16 paths, "
            "mixed-precision transitions, W4/W8 parameters and an eight-way "
            "Split-to-Concat edge bundle."
        ),
    )
    model = helper.make_model(
        graph,
        producer_name="HNNX",
        producer_version="0.1.18",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    checker.check_model(model)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    checker.check_model(inferred)
    return inferred


def _activation(name: str, data_type: str, scale: float) -> dict[str, object]:
    return {
        "name": name,
        "output_dtype": data_type,
        "y_scale": scale,
        "y_zero_point": 0,
    }


def _parameter(name: str, data_type: str, scale: float) -> dict[str, object]:
    return {
        "name": name,
        "output_dtype": data_type,
        "y_scale": [scale] * 8,
        "y_zero_point": [0] * 8,
        "axis": 1,
    }


def build_encodings() -> dict[str, object]:
    return {
        "version": "2.0.0",
        "activation_encodings": [
            _activation("input", "int16", 0.0009765625),
            _activation("low_proj", "int4", 0.125),
            _activation("low_relu", "int4", 0.125),
            _activation("mid_proj", "uint8", 0.03125),
            _activation("mid_gate", "uint8", 0.00390625),
            _activation("high_proj", "int16", 0.0009765625),
            _activation("high_gate", "int16", 0.0009765625),
            _activation("low_mid_mix", "uint8", 0.015625),
            _activation("mid_high_mix", "int16", 0.0009765625),
            _activation("mixed_concat", "int16", 0.0009765625),
            _activation("probabilities", "int16", 0.000030517578125),
        ],
        "param_encodings": [
            _parameter("low.weight", "int4", 0.015625),
            _parameter("mid.weight", "int8", 0.00390625),
            _parameter("high.weight", "int8", 0.001953125),
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "examples"
    output.mkdir(exist_ok=True)
    model_path = output / "hnnx-mixed-precision.onnx"
    encodings_path = output / "hnnx-mixed-precision.encodings"
    onnx.save_model(build_model(), model_path)
    encodings_path.write_text(
        json.dumps(build_encodings(), indent=2) + "\n", encoding="utf-8"
    )
    print(model_path)
    print(encodings_path)


if __name__ == "__main__":
    main()
