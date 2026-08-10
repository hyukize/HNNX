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
const reads = [];
const stats = [];
const workspaceFiles = new Map();
let autoPinOnnxEditors = true;
let autoLoadEncodings = true;
let colorTheme = 'auto';
let activeColorThemeKind = 2;
let saveDialogResult = null;
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') {
        return {
            window: {
                get activeColorTheme() {
                    return { kind: activeColorThemeKind };
                },
                createOutputChannel: () => ({ appendLine() {} }),
                withProgress: async (options, task) => await task(),
                showSaveDialog: async () => saveDialogResult,
                showInformationMessage: (message) => messages.push(message)
            },
            workspace: {
                getConfiguration: () => ({
                    get: (name, fallback) => {
                        if (name === 'autoPinOnnxEditors') {
                            return autoPinOnnxEditors;
                        }
                        if (name === 'autoLoadEncodings') {
                            return autoLoadEncodings;
                        }
                        if (name === 'colorTheme') {
                            return colorTheme;
                        }
                        return fallback;
                    },
                    update: async (name, value) => {
                        if (name === 'colorTheme') {
                            colorTheme = value;
                        }
                    }
                }),
                fs: {
                    copy: async (source, target, options) => copies.push({ source, target, options }),
                    readFile: async (uri) => {
                        reads.push(uri.path);
                        if (!workspaceFiles.has(uri.path)) {
                            throw new Error(`File not found: ${uri.path}`);
                        }
                        return workspaceFiles.get(uri.path);
                    },
                    stat: async (uri) => {
                        stats.push(uri.path);
                        if (!workspaceFiles.has(uri.path)) {
                            throw new Error(`File not found: ${uri.path}`);
                        }
                        return { type: 1 };
                    }
                }
            },
            commands: {
                executeCommand: async (command) => commands.push(command)
            },
            ColorThemeKind: { Light: 1, Dark: 2, HighContrast: 3, HighContrastLight: 4 },
            FileType: { File: 1 },
            ConfigurationTarget: { Global: 1 },
            ProgressLocation: { Notification: 15 },
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

test('resolves Auto from VS Code and persists explicit HNNX themes', async () => {
    const posted = [];
    const webview = { postMessage: async (message) => posted.push(message) };
    colorTheme = 'auto';
    activeColorThemeKind = 2;
    await provider.sendTheme(model, webview);
    assert.deepEqual(posted.at(-1), { type: 'theme', preference: 'auto', effective: 'dark' });

    await provider.setTheme(model, webview, 'light');
    assert.equal(colorTheme, 'light');
    assert.deepEqual(posted.at(-1), { type: 'theme', preference: 'light', effective: 'light' });

    activeColorThemeKind = 1;
    await provider.setTheme(model, webview, 'auto');
    assert.deepEqual(posted.at(-1), { type: 'theme', preference: 'auto', effective: 'light' });
    await assert.rejects(provider.setTheme(model, webview, 'sepia'), /Unsupported HNNX color theme/);
});

test('auto-detects conventional encodings and reloads the exact attached URI', async () => {
    const encodings = { ...model, path: '/models/demo.encodings' };
    const bytes = Uint8Array.from([123, 125]);
    workspaceFiles.set(encodings.path, bytes);
    assert.equal((await provider.findEncodings(model)).path, encodings.path);

    const posted = [];
    const document = { uri: model, encodingUri: encodings };
    reads.length = 0;
    await provider.reloadEncodings(document, { postMessage: async (message) => posted.push(message) });
    assert.deepEqual(reads, [encodings.path]);
    assert.equal(posted.at(-1).type, 'attach');
    assert.equal(posted.at(-1).encodings.path, encodings.path);
    assert.equal(posted.at(-1).encodings.name, 'demo.encodings');
});

test('auto-load setting controls neighboring encodings during model open', async () => {
    const encodings = { ...model, path: '/models/demo.encodings' };
    workspaceFiles.set(model.path, Uint8Array.from([8, 1]));
    workspaceFiles.set(encodings.path, Uint8Array.from([123, 125]));
    const posted = [];
    const document = { uri: model, encodingUri: null };
    const webview = { postMessage: async (message) => posted.push(message) };

    autoLoadEncodings = false;
    stats.length = 0;
    await provider.open(document, webview);
    assert.equal(document.encodingUri, null);
    assert.equal(posted.at(-1).encodings, null);
    assert.deepEqual(stats, []);

    autoLoadEncodings = true;
    await provider.open(document, webview);
    assert.equal(document.encodingUri.path, encodings.path);
    assert.equal(posted.at(-1).encodings.path, encodings.path);
});

test('detaching clears the document URI and tells only the current webview', async () => {
    const document = { uri: model, encodingUri: { ...model, path: '/models/demo.encodings' } };
    const posted = [];
    await provider.onMessage(document, { postMessage: async (message) => posted.push(message) }, {
        type: 'detachEncodings'
    });
    assert.equal(document.encodingUri, null);
    assert.deepEqual(posted, [{ type: 'detachEncodings' }]);
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
