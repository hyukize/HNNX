import { EncodingFile, Utility } from '../source/aimet.js';
import assert from 'assert/strict';

const value = (name, dimensions, initializer = false) => ({
    name,
    type: { shape: { dimensions } },
    initializer: initializer ? {} : null
});

const activation = value('conv.output', [1, 16, 8, 8]);
const weight = value('conv.weight', [16, 3, 3, 3], true);
const model = {
    modules: [{
        inputs: [{ name: 'input', value: [value('input', [1, 3, 8, 8])] }],
        outputs: [{ name: 'output', value: [activation] }],
        nodes: [{
            inputs: [{ name: 'input', value: [weight] }],
            outputs: [{ name: 'output', value: [activation] }]
        }]
    }],
    functions: []
};

const version2 = new EncodingFile();
assert.equal(version2.open({
    version: '2.0.0',
    activation_encodings: [{
        name: 'conv.output',
        output_dtype: 'uint8',
        y_scale: 0.125,
        y_zero_point: 17
    }],
    param_encodings: [{
        name: 'conv.weight',
        output_dtype: 'int4',
        y_scale: Array.from({ length: 16 }, (_, index) => 0.01 + index * 0.001),
        axis: 0
    }]
}), true);
version2.bind(model);
assert.deepEqual(version2.summary, {
    total: 2,
    activations: 1,
    parameters: 1,
    matched: 2,
    unmatched: 0,
    modelValues: 3,
    unencoded: 1,
    inferred: 0
});
assert.equal(version2.value(activation).label, 'A8');
assert.equal(version2.tensor(null, weight).label, 'W4');
assert.equal(version2.issues.length, 0);
assert.deepEqual(version2.profile, {
    encodings: [['A8', 1], ['W4', 1]],
    cache: []
});
const grouped = version2.node(model.modules[0].nodes[0]);
assert.deepEqual(grouped.inputs, []);
assert.deepEqual(grouped.parameters.map((entry) => entry.label), ['W4']);
assert.deepEqual(grouped.outputs.map((entry) => entry.label), ['A8']);
assert.deepEqual(EncodingFile.labels(grouped), ['P:W4', 'O:A8']);
assert.deepEqual(EncodingFile.description(grouped), ['Parameter: W4', 'Output: A8']);
assert.equal(EncodingFile.precision(grouped), 'a8');
assert.equal(EncodingFile.transition(grouped), '');
assert.deepEqual(EncodingFile.nodeBadge(grouped, grouped), {
    labels: ['Q:A8', 'W4'],
    descriptions: ['Output QParams: A8', 'Parameter QParams: W4']
});
const mixedSignature = {
    inputs: [{ label: 'A16' }, { label: 'A8' }, { label: 'A8' }],
    parameters: [],
    outputs: [{ label: 'A16' }]
};
assert.equal(EncodingFile.transition(mixedSignature), 'A8/A16 \u2192 A16');
assert.deepEqual(EncodingFile.nodeBadge(mixedSignature, mixedSignature), {
    labels: ['Q:A8/A16\u2192A16'],
    descriptions: ['Quantization signature: A8/A16 \u2192 A16 (explicit output QParam)']
});
assert.deepEqual(EncodingFile.labels({
    inputs: [{ label: 'A16' }, { label: 'A8' }, { label: 'A8' }],
    parameters: [{ label: 'W8' }, { label: 'W8' }],
    outputs: [{ label: 'A16' }, { label: 'A16' }]
}), ['I:A8/A16', 'P:W8', 'O:A16']);

const mixed = new EncodingFile();
assert.equal(mixed.open({
    version: '1.0.0',
    activation_encodings: [
        { name: 'past_key_values.attention.2.key', dtype: 'INT', bw: 8, scale: [0.1], offset: [0], enc_type: 'PER_TENSOR' },
        { name: 'single.2.key', dtype: 'INT', bw: 8, scale: [0.1], offset: [0], enc_type: 'PER_TENSOR' },
        { name: 'single.2.value', dtype: 'INT', bw: 8, scale: [0.1], offset: [0], enc_type: 'PER_TENSOR' },
        { name: 'single.3.key', dtype: 'INT', bw: 16, scale: [0.1], offset: [0], enc_type: 'PER_TENSOR' }
    ],
    param_encodings: [
        { name: 'conv.weight', dtype: 'INT', bw: 8, scale: [0.1], offset: [0], enc_type: 'PER_TENSOR' }
    ]
}), true);
assert.deepEqual(mixed.profile, {
    encodings: [['A16', 1], ['A8', 3], ['W8', 1]],
    cache: [['A8', 3]]
});

const legacy = new EncodingFile();
assert.equal(legacy.open({
    version: '0.6.1',
    activation_encodings: {
        missing: [{ dtype: 'int', bitwidth: 16, is_symmetric: 'False', scale: 0.25, offset: -3, min: -1, max: 1 }]
    },
    param_encodings: {
        'conv.weight': [
            { dtype: 'int', bitwidth: 8, is_symmetric: 'True', scale: 0.1, offset: 0, min: -1, max: 1 },
            { dtype: 'int', bitwidth: 8, is_symmetric: 'True', scale: 0.2, offset: 0, min: -2, max: 2 }
        ]
    }
}), true);
legacy.bind(model);
assert.equal(legacy.get('conv.weight').granularity, 'per-channel');
assert.equal(legacy.get('conv.weight').symmetric, true);
assert.equal(legacy.summary.unmatched, 1);
assert.match(legacy.issues[0].message, /missing/);

const passthroughInput = value('passthrough.input', [1, 2, 3]);
const transposeOutput = value('transpose.output', [1, 3, 2]);
const reshapeOutput = value('reshape.output', [1, 6]);
const splitOutput0 = value('split.output.0', [1, 3]);
const splitOutput1 = value('split.output.1', [1, 3]);
const concatOutput = value('concat.output', [1, 6]);
const castOutput = value('cast.output', [1, 6]);
const reluOutput = value('relu.output', [1, 6]);
const shape = value('reshape.shape', [2], true);
const passthroughModel = {
    modules: [{
        inputs: [{ name: 'input', value: [passthroughInput] }],
        outputs: [{ name: 'output', value: [reluOutput] }],
        nodes: [
            {
                type: { name: 'Transpose' },
                inputs: [{ name: 'data', value: [passthroughInput] }],
                outputs: [{ name: 'output', value: [transposeOutput] }]
            },
            {
                type: { name: 'Reshape' },
                inputs: [{ name: 'data', value: [transposeOutput] }, { name: 'shape', value: [shape] }],
                outputs: [{ name: 'output', value: [reshapeOutput] }]
            },
            {
                type: { name: 'Split' },
                inputs: [{ name: 'input', value: [reshapeOutput] }],
                outputs: [{ name: 'outputs', value: [splitOutput0, splitOutput1] }]
            },
            {
                type: { name: 'Concat' },
                inputs: [{ name: 'inputs', value: [splitOutput0, splitOutput1] }],
                outputs: [{ name: 'output', value: [concatOutput] }]
            },
            {
                type: { name: 'Cast' },
                inputs: [{ name: 'input', value: [concatOutput] }],
                outputs: [{ name: 'output', value: [castOutput] }]
            },
            {
                type: { name: 'Relu' },
                inputs: [{ name: 'input', value: [castOutput] }],
                outputs: [{ name: 'output', value: [reluOutput] }]
            }
        ]
    }],
    functions: []
};
const inferred = new EncodingFile();
assert.equal(inferred.open({
    version: '2.0.0',
    activation_encodings: [{
        name: 'passthrough.input',
        output_dtype: 'uint8',
        y_scale: 0.125,
        y_zero_point: 17
    }],
    param_encodings: []
}), true);
inferred.bind(passthroughModel);
assert.equal(inferred.value(transposeOutput), null);
assert.equal(inferred.precision(transposeOutput).label, 'A8');
assert.equal(inferred.precision(transposeOutput).inferred, true);
assert.equal(inferred.precision(reshapeOutput).label, 'A8');
assert.equal(inferred.precision(splitOutput0).label, 'A8');
assert.equal(inferred.precision(splitOutput1).label, 'A8');
assert.equal(inferred.precision(concatOutput).label, 'A8');
assert.equal(inferred.precision(castOutput).label, 'A8');
assert.equal(inferred.precision(reluOutput), null);
assert.equal(inferred.summary.inferred, 6);
const transposeGroup = inferred.node(passthroughModel.modules[0].nodes[0]);
assert.deepEqual(EncodingFile.labels(transposeGroup), ['I:A8', 'O:A8']);
assert.deepEqual(EncodingFile.description(transposeGroup), ['Input: A8', 'Output: A8 (inferred)']);
const explicitTransposeGroup = inferred.node(passthroughModel.modules[0].nodes[0], false);
assert.deepEqual(EncodingFile.labels(explicitTransposeGroup), ['I:A8']);
assert.deepEqual(EncodingFile.nodeBadge(explicitTransposeGroup, transposeGroup), {
    labels: [],
    descriptions: []
});
const reluGroup = inferred.node(passthroughModel.modules[0].nodes[5]);
assert.deepEqual(EncodingFile.labels(reluGroup), ['I:A8']);

const topKInput = value('topk.input', [1, 32]);
const topKCount = value('topk.k', [1], true);
const topKValues = value('topk.values', [1, 4]);
const topKIndices = value('topk.indices', [1, 4]);
const topKModel = {
    modules: [{
        inputs: [{ name: 'input', value: [topKInput] }],
        outputs: [
            { name: 'values', value: [topKValues] },
            { name: 'indices', value: [topKIndices] }
        ],
        nodes: [{
            type: { name: 'TopK' },
            inputs: [
                { name: 'X', value: [topKInput] },
                { name: 'K', value: [topKCount] }
            ],
            outputs: [
                { name: 'Values', value: [topKValues] },
                { name: 'Indices', value: [topKIndices] }
            ]
        }]
    }],
    functions: []
};
const topK = new EncodingFile();
assert.equal(topK.open({
    version: '2.0.0',
    activation_encodings: [{
        name: 'topk.input',
        output_dtype: 'uint8',
        y_scale: 0.125,
        y_zero_point: 17
    }],
    param_encodings: []
}), true);
topK.bind(topKModel);
assert.equal(topK.precision(topKValues).label, 'A8');
assert.equal(topK.precision(topKValues).inferred, true);
assert.equal(topK.precision(topKIndices), null);
assert.deepEqual(EncodingFile.labels(topK.node(topKModel.modules[0].nodes[0])), ['I:A8', 'O:A8']);

const concatInput8 = value('concat.input.8', [1, 3]);
const concatInput16 = value('concat.input.16', [1, 3]);
const mixedConcatOutput = value('concat.mixed.output', [1, 6]);
const mixedConcatModel = {
    modules: [{
        inputs: [{ name: 'inputs', value: [concatInput8, concatInput16] }],
        outputs: [{ name: 'output', value: [mixedConcatOutput] }],
        nodes: [{
            type: { name: 'Concat' },
            inputs: [{ name: 'inputs', value: [concatInput8, concatInput16] }],
            outputs: [{ name: 'output', value: [mixedConcatOutput] }]
        }]
    }],
    functions: []
};
const mixedConcat = new EncodingFile();
assert.equal(mixedConcat.open({
    version: '2.0.0',
    activation_encodings: [
        { name: 'concat.input.8', output_dtype: 'uint8', y_scale: 0.1 },
        { name: 'concat.input.16', output_dtype: 'int16', y_scale: 0.01 }
    ],
    param_encodings: []
}), true);
mixedConcat.bind(mixedConcatModel);
assert.equal(mixedConcat.precision(mixedConcatOutput), null);

assert.equal(Utility.count([[1, 2], [3, 4]]), 4);
assert.equal(Utility.format([1, 2, 3, 4, 5]), '[1, 2, 3, 4, …] (5)');
