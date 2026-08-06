'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const commands = [];
const copies = [];
const messages = [];
let autoPinOnnxEditors = true;
let saveDialogResult = null;
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') {
        return {
            window: {
                createOutputChannel: () => ({ appendLine() {} }),
                showSaveDialog: async () => saveDialogResult,
                showInformationMessage: (message) => messages.push(message)
            },
            workspace: {
                getConfiguration: () => ({
                    get: (name, fallback) => name === 'autoPinOnnxEditors' ? autoPinOnnxEditors : fallback
                }),
                fs: {
                    copy: async (source, target, options) => copies.push({ source, target, options })
                }
            },
            commands: {
                executeCommand: async (command) => commands.push(command)
            },
            Uri: {
                joinPath: (uri, ...parts) => ({
                    ...uri,
                    path: require('node:path').posix.join(uri.path, ...parts)
                })
            }
        };
    }
    return originalLoad.call(this, request, parent, isMain);
};

const {
    OnnxEditorProvider,
    applyOnnxGraphSurgeonEdits,
    inferOnnxGraphSurgeonShapes,
    validatePython
} = require('../extension.js');
Module._load = originalLoad;

const provider = new OnnxEditorProvider({});
const model = { scheme: 'vscode-remote', authority: 'container', path: '/models/demo.onnx' };

test('resolves external ONNX data inside the model directory', () => {
    const uri = provider.resolveSibling(model, 'weights/demo.data');
    assert.equal(uri.path, '/models/weights/demo.data');
    assert.equal(uri.scheme, 'vscode-remote');
});

test('rejects external data paths outside the model directory', () => {
    assert.throws(() => provider.resolveSibling(model, '../secret'), /escapes the model directory/);
    assert.throws(() => provider.resolveSibling(model, '/etc/passwd'), /escapes the model directory/);
});

test('keeps an active ONNX preview editor open by default', async () => {
    commands.length = 0;
    autoPinOnnxEditors = true;
    await provider.keepEditorOpen(model, { active: true });
    assert.deepEqual(commands, ['workbench.action.keepEditor']);
});

test('does not keep an inactive editor or disabled preview editor open', async () => {
    commands.length = 0;
    autoPinOnnxEditors = true;
    await provider.keepEditorOpen(model, { active: false });
    autoPinOnnxEditors = false;
    await provider.keepEditorOpen(model, { active: true });
    assert.deepEqual(commands, []);
});

test('Save As copies an unedited ONNX to the chosen URI and permits overwrite', async () => {
    const posted = [];
    const webview = { postMessage: async (message) => posted.push(message) };
    copies.length = 0;
    messages.length = 0;
    saveDialogResult = { ...model, path: '/models/demo-copy.onnx' };
    await provider.saveOnnxAs(model, webview, null);
    assert.equal(copies.length, 1);
    assert.equal(copies[0].target.path, '/models/demo-copy.onnx');
    assert.deepEqual(copies[0].options, { overwrite: true });
    assert.equal(posted.at(-1).path, '/models/demo-copy.onnx');

    copies.length = 0;
    saveDialogResult = { ...model };
    await provider.saveOnnxAs(model, webview, null);
    assert.equal(copies.length, 0);
    assert.equal(posted.at(-1).path, '/models/demo.onnx');
});

test('runs the NVIDIA ONNX GraphSurgeon backend when a test interpreter is configured', {
    skip: !process.env.HNNX_TEST_PYTHON
}, async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hnnx-provider-'));
    const output = path.join(directory, 'edited.onnx');
    try {
        await applyOnnxGraphSurgeonEdits(
            path.resolve(__dirname, '../../source/onnx-graphsurgeon.py'),
            path.resolve(__dirname, '../../test/aimet.onnx'),
            output,
            [{ kind: 'node-input', nodeIndex: 0, inputIndex: 0, value: 'x' }],
            process.env.HNNX_TEST_PYTHON
        );
        assert.equal(fs.existsSync(output), true);
    } finally {
        fs.rmSync(directory, { recursive: true, force: true });
    }
});

test('validates a configured GraphSurgeon interpreter before saving it', {
    skip: !process.env.HNNX_TEST_PYTHON
}, async () => {
    assert.equal(await validatePython(process.env.HNNX_TEST_PYTHON), true);
    assert.equal(await validatePython('/path/that/does/not/exist/python'), false);
});

test('runs shape inference through the configured remote-side Python', {
    skip: !process.env.HNNX_TEST_PYTHON
}, async () => {
    const result = await inferOnnxGraphSurgeonShapes(
        path.resolve(__dirname, '../../source/onnx-graphsurgeon.py'),
        path.resolve(__dirname, '../../test/aimet.onnx'),
        [],
        process.env.HNNX_TEST_PYTHON
    );
    assert.equal(result.nodes, 1);
    assert.equal(result.tensors.some((tensor) => tensor.name === 'y'), true);
});
