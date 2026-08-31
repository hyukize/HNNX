#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    entry
    for entry in sys.path
    if os.path.abspath(entry or os.getcwd()) != SCRIPT_DIRECTORY
]


def _dependencies():
    try:
        import onnx
        import onnx_graphsurgeon as graphsurgeon
    except ImportError as error:
        raise RuntimeError(
            f"NVIDIA ONNX GraphSurgeon is required for {sys.executable} ({error}). "
            "Install it in a virtual environment with: "
            "python3 -m pip install onnx onnx_graphsurgeon "
            "--extra-index-url https://pypi.ngc.nvidia.com"
        ) from error
    return onnx, graphsurgeon


_MAX_INFERENCE_EXTERNAL_TENSOR_BYTES = 1024 * 1024
_MAX_INFERENCE_EXTERNAL_TOTAL_BYTES = 32 * 1024 * 1024


_VALUE_DEPENDENT_INPUTS = {
    "ConstantOfShape": {0},
    "Expand": {1},
    "Pad": {1},
    "ReduceL1": {1},
    "ReduceL2": {1},
    "ReduceLogSum": {1},
    "ReduceLogSumExp": {1},
    "ReduceMax": {1},
    "ReduceMean": {1},
    "ReduceMin": {1},
    "ReduceProd": {1},
    "ReduceSum": {1},
    "ReduceSumSquare": {1},
    "Reshape": {1},
    "Resize": {1, 2, 3},
    "Slice": {1, 2, 3, 4},
    "Split": {1},
    "Squeeze": {1},
    "Tile": {1},
    "TopK": {1},
    "Unsqueeze": {1},
}


def _attribute_tensors(onnx, attribute):
    if attribute.type == onnx.AttributeProto.TENSOR:
        return [attribute.t]
    if attribute.type == onnx.AttributeProto.TENSORS:
        return list(attribute.tensors)
    return []


def _external_tensor_entries(onnx, model):
    entries = []

    def visit(graph, scope):
        for initializer in graph.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                entries.append(
                    {
                        "tensor": initializer,
                        "kind": "initializer",
                        "name": initializer.name,
                        "node": None,
                        "label": initializer.name or f"{scope}/initializer",
                    }
                )
        for node_index, node in enumerate(graph.node):
            node_label = node.name or f"{node.op_type}[{node_index}]"
            for attribute in node.attribute:
                for tensor_index, tensor in enumerate(
                    _attribute_tensors(onnx, attribute)
                ):
                    if tensor.data_location != onnx.TensorProto.EXTERNAL:
                        continue
                    suffix = f"[{tensor_index}]" if tensor_index else ""
                    entries.append(
                        {
                            "tensor": tensor,
                            "kind": "attribute",
                            "name": node.output[0] if node.output else "",
                            "node": node,
                            "label": (
                                f"{scope}/{node_label}:{attribute.name}{suffix}"
                            ),
                        }
                    )
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g, f"{scope}/{node_label}:{attribute.name}")
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, nested in enumerate(attribute.graphs):
                        visit(
                            nested,
                            f"{scope}/{node_label}:{attribute.name}[{graph_index}]",
                        )

    visit(model.graph, "graph")
    return entries


def _external_metadata(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


def _shape_external_candidate(onnx, entry, value_inputs):
    if entry["kind"] == "attribute":
        return entry["node"] is not None and entry["node"].op_type == "Constant"
    integer_types = {
        onnx.TensorProto.BOOL,
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
    }
    tensor = entry["tensor"]
    return entry["name"] in value_inputs or tensor.data_type in integer_types


def _value_dependent_inputs(model):
    names = set()
    for node in model.graph.node:
        positions = _VALUE_DEPENDENT_INPUTS.get(node.op_type, set())
        for input_index in positions:
            if input_index < len(node.input) and node.input[input_index]:
                names.add(node.input[input_index])
    return names


def _requires_data_propagation(model):
    initializers = {initializer.name for initializer in model.graph.initializer}
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    for name in _value_dependent_inputs(model):
        if name in initializers:
            continue
        producer = producers.get(name)
        if producer is not None and producer.op_type != "Constant":
            return True
    return False


def _hydrate_inference_external_data(onnx, model, input_path, warnings):
    value_inputs = _value_dependent_inputs(model)
    candidates = []
    skipped = []
    total = 0
    for entry in _external_tensor_entries(onnx, model):
        if not _shape_external_candidate(onnx, entry, value_inputs):
            continue
        metadata = _external_metadata(entry["tensor"])
        location = metadata.get("location")
        try:
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", ""))
        except (TypeError, ValueError):
            skipped.append(entry["label"])
            continue
        if (
            not location
            or offset < 0
            or length < 0
            or length > _MAX_INFERENCE_EXTERNAL_TENSOR_BYTES
            or total + length > _MAX_INFERENCE_EXTERNAL_TOTAL_BYTES
        ):
            skipped.append(entry["label"])
            continue
        path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(input_path)), location)
        )
        if not os.path.isfile(path):
            continue
        entry.update({"path": path, "offset": offset, "length": length})
        candidates.append(entry)
        total += length

    files = {}
    for entry in candidates:
        files.setdefault(entry["path"], []).append(entry)
    loaded = 0
    loaded_bytes = 0
    for path, entries in files.items():
        with open(path, "rb") as stream:
            for entry in sorted(entries, key=lambda value: value["offset"]):
                stream.seek(entry["offset"])
                data = stream.read(entry["length"])
                if len(data) != entry["length"]:
                    skipped.append(entry["label"])
                    continue
                tensor = entry["tensor"]
                tensor.raw_data = data
                tensor.data_location = onnx.TensorProto.DEFAULT
                tensor.ClearField("external_data")
                loaded += 1
                loaded_bytes += len(data)
    if skipped:
        warnings.append(
            {
                "code": "external-shape-data-skipped",
                "message": (
                    "Some external constants were not loaded because their "
                    "metadata, size, or data range was unsafe for partial loading."
                ),
                "tensors": sorted(set(skipped)),
            }
        )
    return {"tensors": loaded, "bytes": loaded_bytes}


def _prepare_shape_inference(onnx, model, input_path, warnings):
    prepared = onnx.ModelProto()
    prepared.CopyFrom(model)
    _hydrate_inference_external_data(onnx, prepared, input_path, warnings)
    return prepared, _requires_data_propagation(prepared)


def _copy_inferred_types(target, inferred):
    target.graph.ClearField("value_info")
    target.graph.value_info.extend(inferred.graph.value_info)
    inferred_inputs = {value.name: value for value in inferred.graph.input}
    inferred_outputs = {value.name: value for value in inferred.graph.output}
    for value in target.graph.input:
        if value.name in inferred_inputs:
            value.type.CopyFrom(inferred_inputs[value.name].type)
    for value in target.graph.output:
        if value.name in inferred_outputs:
            value.type.CopyFrom(inferred_outputs[value.name].type)


def _external_initializers(model):
    onnx, _ = _dependencies()
    return {
        initializer.name: initializer
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
    }


def _missing_external_data(onnx, model, input_path):
    directory = os.path.dirname(os.path.abspath(input_path))
    missing = set()
    for external in _external_tensor_entries(onnx, model):
        tensor = external["tensor"]
        locations = [
            entry.value
            for entry in tensor.external_data
            if entry.key == "location" and entry.value
        ]
        for location in locations:
            path = os.path.normpath(os.path.join(directory, location))
            if not os.path.isfile(path):
                missing.add(location)
    return sorted(missing)


def _restore_external_data(model, originals, input_path, output_path):
    input_directory = os.path.dirname(os.path.abspath(input_path))
    output_directory = os.path.dirname(os.path.abspath(output_path))
    if originals and input_directory != output_directory:
        raise ValueError(
            "Models with external tensor data must be saved beside "
            "the original ONNX file."
        )
    for initializer in model.graph.initializer:
        original = originals.get(initializer.name)
        if original is None:
            continue
        initializer.CopyFrom(original)
        for entry in initializer.external_data:
            if entry.key == "location":
                source = os.path.normpath(os.path.join(input_directory, entry.value))
                entry.value = os.path.relpath(source, output_directory)


def _optional_input(onnx, model, node, input_index):
    domain = node.domain or ""
    versions = {opset.domain or "": opset.version for opset in model.opset_import}
    version = versions.get(domain)
    if version is None:
        raise ValueError(
            f"Cannot disconnect input {input_index} on '{node.op}': "
            f"the model has no opset for domain '{domain or 'ai.onnx'}'."
        )
    try:
        schema = onnx.defs.get_schema(node.op, version, domain)
    except Exception as error:
        raise ValueError(
            f"Cannot disconnect input {input_index} on '{node.op}': "
            "its ONNX schema is unavailable."
        ) from error
    if input_index < len(schema.inputs):
        formal = schema.inputs[input_index]
    elif (
        schema.inputs
        and schema.inputs[-1].option
        == onnx.defs.OpSchema.FormalParameterOption.Variadic
    ):
        formal = schema.inputs[-1]
    else:
        return False
    return formal.option == onnx.defs.OpSchema.FormalParameterOption.Optional


def _numpy_dtype(onnx, name):
    aliases = {
        "boolean": "BOOL",
        "bool": "BOOL",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "float16": "FLOAT16",
        "bfloat16": "BFLOAT16",
    }
    key = aliases.get(str(name).lower(), str(name).upper())
    value = getattr(onnx.TensorProto, key, None)
    if value is None:
        raise ValueError(f"Unsupported ONNX tensor data type '{name}'.")
    return onnx.helper.tensor_dtype_to_np_dtype(value)


def _add_graph_output(graphsurgeon, graph, tensors, name, value):
    if value.name == name:
        graph.outputs.append(value)
        return
    if name in tensors:
        raise ValueError(f"Tensor name '{name}' is already in use.")
    output = graphsurgeon.Variable(
        name=name,
        dtype=value.dtype,
        shape=value.shape,
    )
    node_names = {node.name for node in graph.nodes if node.name}
    node_name = f"NetronGraphOutput_{len(graph.outputs)}"
    suffix = 2
    while node_name in node_names:
        node_name = f"NetronGraphOutput_{len(graph.outputs)}_{suffix}"
        suffix += 1
    graph.nodes.append(
        graphsurgeon.Node(
            op="Identity",
            name=node_name,
            inputs=[value],
            outputs=[output],
        )
    )
    graph.outputs.append(output)
    tensors[name] = output


def _apply_graph_edits(input_path, edits):
    onnx, graphsurgeon = _dependencies()
    if not isinstance(edits, list):
        raise ValueError("ONNX graph edits must be a list.")

    model = onnx.load(input_path, load_external_data=False)
    graph = graphsurgeon.import_onnx(model)
    tensors = graph.tensors(check_duplicates=True)
    original_nodes = list(graph.nodes)
    deleted = False

    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("Invalid ONNX graph edit.")
        kind = edit.get("kind", "node-input")

        if kind == "add-input":
            name = edit.get("name")
            data_type = edit.get("dataType")
            dimensions = edit.get("dimensions")
            if not isinstance(name, str) or not name:
                raise ValueError("A graph input name is required.")
            if name in tensors:
                raise ValueError(f"Tensor name '{name}' is already in use.")
            if not isinstance(dimensions, list):
                raise ValueError(f"Graph input '{name}' is missing its shape.")
            value = graphsurgeon.Variable(
                name=name,
                dtype=_numpy_dtype(onnx, data_type),
                shape=dimensions,
            )
            graph.inputs.append(value)
            tensors[name] = value
        elif kind == "add-node":
            op = edit.get("op")
            domain = edit.get("domain", "")
            name = edit.get("name", "")
            input_names = edit.get("inputs")
            output_names = edit.get("outputs")
            attributes = edit.get("attributes", {})
            initializers = edit.get("initializers", [])
            if not isinstance(op, str) or not op:
                raise ValueError("An ONNX operator type is required.")
            if not isinstance(input_names, list) or not input_names:
                raise ValueError(f"Node '{name or op}' requires at least one input.")
            if not isinstance(output_names, list) or not output_names:
                raise ValueError(f"Node '{name or op}' requires at least one output.")
            if not isinstance(attributes, dict):
                raise ValueError(f"Node '{name or op}' has invalid attributes.")
            if not isinstance(initializers, list):
                raise ValueError(f"Node '{name or op}' has invalid initializers.")
            if initializers:
                import numpy

                for initializer in initializers:
                    if not isinstance(initializer, dict):
                        raise ValueError(
                            f"Node '{name or op}' has an invalid initializer."
                        )
                    initializer_name = initializer.get("name")
                    data_type = initializer.get("dataType")
                    dimensions = initializer.get("dimensions")
                    values = initializer.get("values")
                    if not isinstance(initializer_name, str) or not initializer_name:
                        raise ValueError(
                            f"Node '{name or op}' has an unnamed initializer."
                        )
                    if initializer_name in tensors:
                        raise ValueError(
                            f"Tensor name '{initializer_name}' is already in use."
                        )
                    if (
                        not isinstance(dimensions, list)
                        or not all(
                            isinstance(dimension, int) and dimension >= 0
                            for dimension in dimensions
                        )
                        or not isinstance(values, list)
                    ):
                        raise ValueError(
                            f"Initializer '{initializer_name}' has invalid "
                            "dimensions or values."
                        )
                    try:
                        array = numpy.asarray(
                            values, dtype=_numpy_dtype(onnx, data_type)
                        ).reshape(dimensions)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Initializer '{initializer_name}' values do not match "
                            f"shape {dimensions}."
                        ) from error
                    tensors[initializer_name] = graphsurgeon.Constant(
                        name=initializer_name,
                        values=array,
                    )
            inputs = []
            for input_name in input_names:
                if input_name == "":
                    inputs.append(graphsurgeon.Variable.empty())
                    continue
                if input_name not in tensors:
                    raise ValueError(
                        f"Tensor '{input_name}' was not found for node '{name or op}'."
                    )
                inputs.append(tensors[input_name])
            outputs = []
            output_dtype = inputs[0].dtype if inputs else None
            if op == "Shape":
                output_dtype = _numpy_dtype(onnx, "int64")
            elif op == "Where" and len(inputs) >= 2:
                # Where's condition is boolean, but the output type follows X/Y.
                output_dtype = inputs[1].dtype
            elif op in {"Equal", "Greater", "Less", "Not"}:
                output_dtype = _numpy_dtype(onnx, "boolean")
            elif op in {"ArgMax", "ArgMin"}:
                output_dtype = _numpy_dtype(onnx, "int64")
            elif op == "Cast":
                try:
                    output_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                        int(attributes.get("to"))
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Cast node '{name or op}' requires a valid 'to' attribute."
                    ) from None
            for output_index, output_name in enumerate(output_names):
                if not isinstance(output_name, str) or not output_name:
                    raise ValueError(f"Node '{name or op}' has an invalid output name.")
                if output_name in tensors:
                    raise ValueError(f"Tensor name '{output_name}' is already in use.")
                output = graphsurgeon.Variable(
                    name=output_name,
                    # Most ONNX operators preserve their first input data type.
                    # Shape, comparisons, and Cast are handled explicitly above.
                    dtype=(
                        _numpy_dtype(onnx, "int64")
                        if op == "TopK" and output_index == 1
                        else output_dtype
                    ),
                    # New nodes can change rank or dimensions. Let ONNX shape
                    # inference determine output shapes instead of seeding an
                    # incorrect copy of the first input shape.
                    shape=None,
                )
                outputs.append(output)
                tensors[output_name] = output
            graph.nodes.append(
                graphsurgeon.Node(
                    op=op,
                    name=name,
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attributes,
                    domain=domain or None,
                )
            )
        elif kind == "add-output":
            name = edit.get("name")
            value_name = edit.get("value")
            if not isinstance(name, str) or not name:
                raise ValueError("A graph output name is required.")
            if not isinstance(value_name, str) or value_name not in tensors:
                raise ValueError(
                    f"Graph output source tensor '{value_name}' was not found."
                )
            if any(output.name == name for output in graph.outputs):
                raise ValueError(f"Graph output name '{name}' is already in use.")
            _add_graph_output(graphsurgeon, graph, tensors, name, tensors[value_name])
        elif kind == "node-input":
            value_name = edit.get("value")
            if not isinstance(value_name, str) or not value_name:
                raise ValueError("An ONNX graph edit is missing its tensor name.")
            if value_name not in tensors:
                raise ValueError(
                    f"Tensor '{value_name}' was not found in the ONNX graph."
                )
            value = tensors[value_name]
            node_index = edit.get("nodeIndex")
            input_index = edit.get("inputIndex")
            if (
                not isinstance(node_index, int)
                or node_index < 0
                or node_index >= len(original_nodes)
            ):
                raise ValueError(f"ONNX node index '{node_index}' was not found.")
            node = original_nodes[node_index]
            if not isinstance(input_index, int) or input_index < 0:
                raise ValueError(
                    f"Input index '{input_index}' was not found on node {node_index}."
                )
            while len(node.inputs) <= input_index:
                node.inputs.append(graphsurgeon.Variable.empty())
            node.inputs[input_index] = value
        elif kind == "disconnect-input":
            node_index = edit.get("nodeIndex")
            input_index = edit.get("inputIndex")
            if (
                not isinstance(node_index, int)
                or node_index < 0
                or node_index >= len(original_nodes)
            ):
                raise ValueError(f"ONNX node index '{node_index}' was not found.")
            node = original_nodes[node_index]
            if (
                not isinstance(input_index, int)
                or input_index < 0
                or input_index >= len(node.inputs)
            ):
                raise ValueError(
                    f"Input index '{input_index}' was not found on node {node_index}."
                )
            if not _optional_input(onnx, model, node, input_index):
                raise ValueError(
                    f"Cannot disconnect required input {input_index} on "
                    f"node '{node.name or node.op}'. Replace the connection instead."
                )
            node.inputs[input_index] = graphsurgeon.Variable.empty()
        elif kind == "delete-node":
            node_index = edit.get("nodeIndex")
            if (
                not isinstance(node_index, int)
                or node_index < 0
                or node_index >= len(original_nodes)
            ):
                raise ValueError(f"ONNX node index '{node_index}' was not found.")
            node = original_nodes[node_index]
            if node in graph.nodes:
                graph.nodes.remove(node)
                deleted = True
        elif kind == "delete-input":
            input_index = edit.get("inputIndex")
            if (
                not isinstance(input_index, int)
                or input_index < 0
                or input_index >= len(graph.inputs)
            ):
                raise ValueError(
                    f"ONNX graph input index '{input_index}' was not found."
                )
            graph.inputs.pop(input_index)
            deleted = True
        elif kind == "delete-output":
            output_index = edit.get("outputIndex")
            if (
                not isinstance(output_index, int)
                or output_index < 0
                or output_index >= len(graph.outputs)
            ):
                raise ValueError(
                    f"ONNX graph output index '{output_index}' was not found."
                )
            graph.outputs.pop(output_index)
            deleted = True
        elif kind == "rename-node":
            node_index = edit.get("nodeIndex")
            name = edit.get("name")
            if (
                not isinstance(node_index, int)
                or node_index < 0
                or node_index >= len(original_nodes)
            ):
                raise ValueError(f"ONNX node index '{node_index}' was not found.")
            if not isinstance(name, str) or not name:
                raise ValueError("An ONNX node name must be a non-empty string.")
            original_nodes[node_index].name = name
        elif kind == "rename-output":
            node_index = edit.get("nodeIndex")
            output_index = edit.get("outputIndex")
            name = edit.get("name")
            if (
                not isinstance(node_index, int)
                or node_index < 0
                or node_index >= len(original_nodes)
            ):
                raise ValueError(f"ONNX node index '{node_index}' was not found.")
            node = original_nodes[node_index]
            if (
                not isinstance(output_index, int)
                or output_index < 0
                or output_index >= len(node.outputs)
            ):
                raise ValueError(
                    f"Output index '{output_index}' was not found on node {node_index}."
                )
            if not isinstance(name, str) or not name:
                raise ValueError("An ONNX tensor name must be a non-empty string.")
            value = node.outputs[output_index]
            existing = tensors.get(name)
            if existing is not None and existing is not value:
                raise ValueError(f"Tensor name '{name}' is already in use.")
            previous = value.name
            value.name = name
            if previous in tensors and tensors[previous] is value:
                del tensors[previous]
            tensors[name] = value
        elif kind == "rename-graph-output":
            output_index = edit.get("outputIndex")
            name = edit.get("name")
            if (
                not isinstance(output_index, int)
                or output_index < 0
                or output_index >= len(graph.outputs)
            ):
                raise ValueError(
                    f"ONNX graph output index '{output_index}' was not found."
                )
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "An ONNX graph output name must be a non-empty string."
                )
            value = graph.outputs[output_index]
            existing = tensors.get(name)
            if existing is not None and existing is not value:
                raise ValueError(f"Tensor name '{name}' is already in use.")
            previous = value.name
            value.name = name
            if previous in tensors and tensors[previous] is value:
                del tensors[previous]
            tensors[name] = value
        elif kind == "graph-output":
            value_name = edit.get("value")
            if not isinstance(value_name, str) or not value_name:
                raise ValueError("An ONNX graph edit is missing its tensor name.")
            if value_name not in tensors:
                raise ValueError(
                    f"Tensor '{value_name}' was not found in the ONNX graph."
                )
            value = tensors[value_name]
            output_index = edit.get("outputIndex")
            if (
                not isinstance(output_index, int)
                or output_index < 0
                or output_index >= len(graph.outputs)
            ):
                raise ValueError(
                    f"ONNX graph output index '{output_index}' was not found."
                )
            previous = graph.outputs[output_index]
            output_name = edit.get("name", previous.name)
            if not isinstance(output_name, str) or not output_name:
                raise ValueError(
                    "An ONNX graph output name must be a non-empty string."
                )
            if value.name == output_name:
                if value.dtype is None:
                    value.dtype = previous.dtype
                if value.shape is None:
                    value.shape = previous.shape
                graph.outputs[output_index] = value
            else:
                existing = tensors.get(output_name)
                if existing is not None:
                    suffix = 1
                    detached_name = f"{output_name}__netron_detached"
                    while detached_name in tensors:
                        suffix += 1
                        detached_name = f"{output_name}__netron_detached_{suffix}"
                    del tensors[output_name]
                    existing.name = detached_name
                    tensors[detached_name] = existing
                output = graphsurgeon.Variable(
                    name=output_name,
                    dtype=value.dtype if value.dtype is not None else previous.dtype,
                    shape=value.shape if value.shape is not None else previous.shape,
                )
                node_names = {node.name for node in graph.nodes if node.name}
                node_name = f"NetronGraphOutput_{output_index}"
                suffix = 1
                while node_name in node_names:
                    suffix += 1
                    node_name = f"NetronGraphOutput_{output_index}_{suffix}"
                graph.nodes.append(
                    graphsurgeon.Node(
                        op="Identity",
                        name=node_name,
                        inputs=[value],
                        outputs=[output],
                    )
                )
                graph.outputs[output_index] = output
                tensors[output_name] = output
        else:
            raise ValueError(f"Unsupported ONNX GraphSurgeon edit '{kind}'.")

    if deleted:
        # Preserve fixed output positions on multi-output operators. Removing
        # an unused TopK Values output, for example, shifts Indices from output
        # 1 to output 0 and produces an invalid ONNX node even when the deleted
        # node is on an unrelated branch.
        graph.cleanup(remove_unused_node_outputs=False)
    graph.toposort()
    return onnx, graphsurgeon.export_onnx(graph)


def apply_edits(input_path, output_path, edits):
    if not edits:
        raise ValueError("No ONNX graph edits were provided.")
    model = _dependencies()[0].load(input_path, load_external_data=False)
    external = _external_initializers(model)
    onnx, output = _apply_graph_edits(input_path, edits)
    # Newly added intermediate values deliberately start without a shape. Run
    # ONNX inference before saving so graph outputs receive a valid type shape.
    # Infer on a copy that can hydrate small external shape constants, then copy
    # only type metadata back so Save As preserves every external data reference.
    prepared, data_prop = _prepare_shape_inference(onnx, output, input_path, [])
    inferred = onnx.shape_inference.infer_shapes(
        prepared,
        check_type=True,
        strict_mode=True,
        data_prop=data_prop,
    )
    _copy_inferred_types(output, inferred)
    _restore_external_data(output, external, input_path, output_path)
    onnx.save_model(output, output_path)
    onnx.checker.check_model(output_path)


def _tensor_type(onnx, value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type or tensor_type.elem_type == 0:
        return None
    data_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower()
    data_type = {
        "bool": "boolean",
        "float": "float32",
        "double": "float64",
        "complex64": "complex<float32>",
        "complex128": "complex<float64>",
    }.get(data_type, data_type)
    dimensions = []
    if tensor_type.HasField("shape"):
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(dimension.dim_value)
            elif dimension.HasField("dim_param"):
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
    return {
        "name": value_info.name,
        "dataType": data_type,
        "dimensions": dimensions,
    }


def _tensor_types(onnx, model):
    tensors = {}
    values = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    for value_info in values:
        value = _tensor_type(onnx, value_info)
        if value and value["name"]:
            tensors[value["name"]] = value
    for initializer in model.graph.initializer:
        if not initializer.name:
            continue
        data_type = onnx.TensorProto.DataType.Name(initializer.data_type).lower()
        data_type = {
            "bool": "boolean",
            "float": "float32",
            "double": "float64",
        }.get(data_type, data_type)
        tensors[initializer.name] = {
            "name": initializer.name,
            "dataType": data_type,
            "dimensions": list(initializer.dims),
        }
    return tensors


def _node_attribute(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute.i
    return default


def _shape_text(tensor):
    if not tensor:
        return "unknown"
    data_type = tensor.get("dataType") or "unknown"
    dimensions = tensor.get("dimensions")
    if not isinstance(dimensions, list):
        return data_type
    shape = ", ".join("?" if value is None else str(value) for value in dimensions)
    return f"{data_type}[{shape}]"


def _shape_mismatch(node, inputs):
    known = [value for value in inputs if isinstance(value.get("dimensions"), list)]
    data_types = {value.get("dataType") for value in inputs if value.get("dataType")}
    same_type_operators = {
        "Add",
        "Concat",
        "Div",
        "Equal",
        "Greater",
        "Less",
        "MatMul",
        "Mul",
        "Pow",
        "Sub",
    }
    if node.op_type in same_type_operators and len(data_types) > 1:
        values = ", ".join(
            f"{value['name']}={value.get('dataType', 'unknown')}" for value in inputs
        )
        return f"Input data types do not match: {values}."
    if node.op_type in {"Add", "Sub", "Mul", "Div", "Pow"} and len(known) >= 2:
        left = known[0]["dimensions"]
        for right_value in known[1:]:
            right = right_value["dimensions"]
            size = max(len(left), len(right))
            padded_left = [1] * (size - len(left)) + left
            padded_right = [1] * (size - len(right)) + right
            for index, (left_dimension, right_dimension) in enumerate(
                zip(padded_left, padded_right, strict=True)
            ):
                if (
                    left_dimension is not None
                    and right_dimension is not None
                    and left_dimension != right_dimension
                    and left_dimension != 1
                    and right_dimension != 1
                ):
                    axis = index - size
                    return (
                        f"Inputs cannot be broadcast at axis {axis}: "
                        f"{known[0]['name']}={_shape_text(known[0])}, "
                        f"{right_value['name']}={_shape_text(right_value)} "
                        f"({left_dimension} vs {right_dimension})."
                    )
    if node.op_type == "MatMul" and len(known) >= 2:
        left = known[0]["dimensions"]
        right = known[1]["dimensions"]
        if len(left) >= 1 and len(right) >= 2:
            left_dimension = left[-1]
            right_dimension = right[-2]
            if (
                left_dimension is not None
                and right_dimension is not None
                and left_dimension != right_dimension
            ):
                return (
                    "MatMul reduction dimensions do not match: "
                    f"{known[0]['name']}[-1]={left_dimension}, "
                    f"{known[1]['name']}[-2]={right_dimension}."
                )
    if node.op_type == "Concat" and len(known) >= 2:
        ranks = {len(value["dimensions"]) for value in known}
        if len(ranks) > 1:
            values = ", ".join(
                f"{value['name']}=rank {len(value['dimensions'])}" for value in known
            )
            return f"Concat input ranks do not match: {values}."
        rank = next(iter(ranks))
        axis = _node_attribute(node, "axis", 0)
        axis = axis if axis >= 0 else rank + axis
        reference = known[0]["dimensions"]
        for value in known[1:]:
            for index, (expected, actual) in enumerate(
                zip(reference, value["dimensions"], strict=True)
            ):
                if (
                    index != axis
                    and expected is not None
                    and actual is not None
                    and expected != actual
                ):
                    return (
                        f"Concat dimensions differ outside axis {axis}: "
                        f"{known[0]['name']}[{index}]={expected}, "
                        f"{value['name']}[{index}]={actual}."
                    )
    return None


def _shape_inference_error(onnx, model, error):
    message = str(error).strip() or error.__class__.__name__
    tensor_names = set(re.findall(r"['\"]([^'\"]+)['\"]", message))
    patterns = [
        r"node name:\s*([^)]+)",
        r"Node\s*\(([^)]+)\)",
        r"node\s+['\"]([^'\"]+)['\"]",
    ]
    node_name = None
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            node_name = match.group(1).strip()
            break
    node = None
    node_index = None
    for index, candidate in enumerate(model.graph.node):
        if node_name and candidate.name == node_name:
            node = candidate
            node_index = index
            break
    if node is None:
        for index, candidate in enumerate(model.graph.node):
            names = set(candidate.input) | set(candidate.output)
            if tensor_names & names:
                node = candidate
                node_index = index
                break
    if node is None:
        lowered = message.lower()
        matches = [
            (index, candidate)
            for index, candidate in enumerate(model.graph.node)
            if candidate.name and candidate.name.lower() in lowered
        ]
        if len(matches) == 1:
            node_index, node = matches[0]
    tensors = _tensor_types(onnx, model)
    inputs = []
    outputs = []
    if node is not None:
        inputs = [tensors.get(name, {"name": name}) for name in node.input if name]
        outputs = [tensors.get(name, {"name": name}) for name in node.output if name]
    summary = _shape_mismatch(node, inputs) if node is not None else None
    if not summary:
        summary = re.sub(
            r"\[(?:ShapeInferenceError|TypeInferenceError)\]\s*", "", message
        )
        summary = re.sub(r"^Inference error\(s\):\s*", "", summary).strip()
    return {
        "message": message,
        "summary": summary,
        "node": {
            "index": node_index,
            "name": node.name if node is not None else node_name,
            "opType": node.op_type if node is not None else None,
        }
        if node is not None or node_name
        else None,
        "inputs": inputs,
        "outputs": outputs,
    }


def infer_shapes(input_path, edits):
    onnx, model = _apply_graph_edits(input_path, edits)
    warnings = []
    prepared = model
    data_prop = False
    try:
        source = onnx.load(input_path, load_external_data=False)
        has_external_data = bool(_external_tensor_entries(onnx, source))
        if has_external_data:
            missing = _missing_external_data(onnx, source, input_path)
            if missing:
                warnings.append(
                    {
                        "code": "missing-external-data",
                        "message": (
                            "External tensor data is missing; value-dependent "
                            "shapes may remain unknown."
                        ),
                        "files": missing,
                    }
                )
            else:
                # Use the path so ONNX can validate external tensor locations.
                onnx.checker.check_model(input_path)
        prepared, data_prop = _prepare_shape_inference(
            onnx, model, input_path, warnings
        )
        inferred = onnx.shape_inference.infer_shapes(
            prepared,
            check_type=True,
            strict_mode=True,
            data_prop=data_prop,
        )
        # Newly added values intentionally have no seeded shape. Validate only
        # after inference has populated graph-output type shapes, matching the
        # save path in apply_edits().
        # Checking a ModelProto with external references resolves their paths
        # against the process working directory. Those models were already
        # checked above using input_path, which preserves the correct base path.
        if not has_external_data:
            onnx.checker.check_model(inferred)
    except Exception as error:
        try:
            partial = onnx.shape_inference.infer_shapes(
                prepared,
                check_type=False,
                strict_mode=False,
                data_prop=data_prop,
            )
            model = partial
        except Exception:
            pass
        return {
            "error": _shape_inference_error(onnx, model, error),
            "warnings": warnings,
        }
    tensors = {}
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    )
    for value_info in values:
        value = _tensor_type(onnx, value_info)
        if value and value["name"]:
            tensors[value["name"]] = value
    return {
        "tensors": list(tensors.values()),
        "nodes": len(inferred.graph.node),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply HNNX edits with NVIDIA ONNX GraphSurgeon."
    )
    parser.add_argument("--mode", choices=["edit", "infer"], default="edit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--edits", required=True)
    args = parser.parse_args()
    with open(args.edits, encoding="utf-8") as file:
        edits = json.load(file)
    if args.mode == "infer":
        print(json.dumps(infer_shapes(args.input, edits)))
    else:
        if not args.output:
            raise ValueError("--output is required in edit mode.")
        apply_edits(args.input, args.output, edits)
        print(json.dumps({"output": os.path.abspath(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ONNX GraphSurgeon: {error}", file=sys.stderr)
        sys.exit(1)
