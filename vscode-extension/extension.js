'use strict';

const childProcess = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('path');
const vscode = require('vscode');
const manifest = require('./package.json');

const viewType = 'hnnx.onnxViewer';

class OnnxDocument {

    constructor(uri) {
        this.uri = uri;
        this.encodingUri = null;
    }

    dispose() {
    }
}

class OnnxEditorProvider {

    constructor(context) {
        this.context = context;
        this.output = vscode.window.createOutputChannel('HNNX');
    }

    openCustomDocument(uri) {
        return new OnnxDocument(uri);
    }

    async resolveCustomEditor(document, panel) {
        await this.keepEditorOpen(document.uri, panel);
        panel.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.context.extensionUri, 'media')
            ]
        };
        panel.webview.html = this.html(panel.webview);
        const subscriptions = [];
        subscriptions.push(panel.webview.onDidReceiveMessage(async (message) => {
            try {
                await this.onMessage(document, panel.webview, message);
            } catch (error) {
                const text = error instanceof Error ? error.message : String(error);
                this.output.appendLine(text);
                vscode.window.showErrorMessage(`HNNX: ${text}`);
            }
        }));
        subscriptions.push(vscode.window.onDidChangeActiveColorTheme(async () => {
            await this.sendTheme(document.uri, panel.webview);
        }));
        subscriptions.push(vscode.workspace.onDidChangeConfiguration(async (event) => {
            if (event.affectsConfiguration('hnnx.colorTheme', document.uri)) {
                await this.sendTheme(document.uri, panel.webview);
            }
        }));
        panel.onDidDispose(() => subscriptions.forEach((item) => item.dispose()));
    }

    async keepEditorOpen(uri, panel) {
        const configuration = vscode.workspace.getConfiguration('hnnx', uri);
        if (panel.active && configuration.get('autoPinOnnxEditors', true)) {
            await vscode.commands.executeCommand('workbench.action.keepEditor');
        }
    }

    async onMessage(document, webview, message) {
        switch (message.type) {
            case 'ready':
                await this.sendTheme(document.uri, webview);
                await this.open(document, webview);
                break;
            case 'fetch':
                await this.fetch(document.uri, webview, message);
                break;
            case 'pickEncodings':
                await this.pickEncodings(document, webview);
                break;
            case 'reloadEncodings':
                await this.reloadEncodings(document, webview);
                break;
            case 'detachEncodings':
                document.encodingUri = null;
                await webview.postMessage({ type: 'detachEncodings' });
                break;
            case 'saveOnnxAs':
                await this.saveOnnxAs(document.uri, webview, message.edits);
                break;
            case 'inferOnnxShapes':
                await this.inferOnnxShapes(document.uri, webview, message.edits);
                break;
            case 'setTheme':
                await this.setTheme(document.uri, webview, message.value);
                break;
            case 'openExternal':
                if (typeof message.url === 'string') {
                    await vscode.env.openExternal(vscode.Uri.parse(message.url));
                }
                break;
            case 'log':
                this.output.appendLine(`${message.fatal ? 'ERROR' : 'INFO'} ${message.message || ''}`);
                if (message.stack) {
                    this.output.appendLine(message.stack);
                }
                break;
            default:
                break;
        }
    }

    async setTheme(uri, webview, value) {
        const values = new Set(['auto', 'light', 'dark']);
        if (!values.has(value)) {
            throw new Error(`Unsupported HNNX color theme: ${value}`);
        }
        const configuration = vscode.workspace.getConfiguration('hnnx', uri);
        await configuration.update('colorTheme', value, vscode.ConfigurationTarget.Global);
        await this.sendTheme(uri, webview);
    }

    async sendTheme(uri, webview) {
        const configuration = vscode.workspace.getConfiguration('hnnx', uri);
        const configured = configuration.get('colorTheme', 'auto');
        const preference = ['auto', 'light', 'dark'].includes(configured) ? configured : 'auto';
        const kind = vscode.window.activeColorTheme.kind;
        const hostDark = kind === vscode.ColorThemeKind.Dark || kind === vscode.ColorThemeKind.HighContrast;
        let effective = preference;
        if (preference === 'auto') {
            effective = hostDark ? 'dark' : 'light';
        }
        await webview.postMessage({ type: 'theme', preference, effective });
    }

    async open(document, webview) {
        const uri = document.uri;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Opening ${path.posix.basename(uri.path)} in HNNX`,
            cancellable: false
        }, async () => {
            const model = await vscode.workspace.fs.readFile(uri);
            const configuration = vscode.workspace.getConfiguration('hnnx', uri);
            const encodingUri = configuration.get('autoLoadEncodings', true) ? await this.findEncodings(uri) : null;
            document.encodingUri = encodingUri;
            const encodings = encodingUri ? await vscode.workspace.fs.readFile(encodingUri) : null;
            await webview.postMessage({
                type: 'open',
                model: {
                    name: path.posix.basename(uri.path),
                    data: this.arrayBuffer(model)
                },
                encodings: encodingUri ? {
                    name: path.posix.basename(encodingUri.path),
                    path: encodingUri.path,
                    data: this.arrayBuffer(encodings)
                } : null
            });
        });
    }

    async fetch(modelUri, webview, message) {
        try {
            const uri = this.resolveSibling(modelUri, message.file);
            const data = await vscode.workspace.fs.readFile(uri);
            await webview.postMessage({ type: 'fetchResult', id: message.id, data: this.arrayBuffer(data) });
        } catch (error) {
            await webview.postMessage({
                type: 'fetchResult',
                id: message.id,
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }

    async pickEncodings(document, webview) {
        const modelUri = document.uri;
        const selection = await vscode.window.showOpenDialog({
            title: 'Select AIMET encodings',
            defaultUri: vscode.Uri.joinPath(modelUri, '..'),
            canSelectFiles: true,
            canSelectFolders: false,
            canSelectMany: false,
            filters: {
                'AIMET encodings': ['encodings', 'json']
            }
        });
        if (selection && selection.length === 1) {
            document.encodingUri = selection[0];
            await this.attachEncodings(document.encodingUri, webview);
        }
    }

    async reloadEncodings(document, webview) {
        if (!document.encodingUri) {
            document.encodingUri = await this.findEncodings(document.uri);
        }
        if (!document.encodingUri) {
            const action = await vscode.window.showInformationMessage(
                'HNNX: No encodings file is attached or auto-detected.',
                'Load Encodings…'
            );
            if (action === 'Load Encodings…') {
                await this.pickEncodings(document, webview);
            }
            return;
        }
        await this.attachEncodings(document.encodingUri, webview);
    }

    async attachEncodings(uri, webview) {
        const data = await vscode.workspace.fs.readFile(uri);
        await webview.postMessage({
            type: 'attach',
            encodings: {
                name: path.posix.basename(uri.path),
                path: uri.path,
                data: this.arrayBuffer(data)
            }
        });
    }

    async saveOnnxAs(modelUri, webview, edits) {
        try {
            const target = await vscode.window.showSaveDialog({
                title: 'Save ONNX As',
                defaultUri: modelUri,
                filters: {
                    'ONNX model': ['onnx']
                }
            });
            if (!target) {
                await webview.postMessage({ type: 'graphEditSaveResult' });
                return;
            }
            if (Array.isArray(edits)) {
                const pythonPath = await this.graphSurgeonPython(modelUri);
                if (!pythonPath) {
                    await webview.postMessage({ type: 'graphEditSaveResult' });
                    return;
                }
                const supported = new Set(['file', 'vscode-remote']);
                if (!supported.has(modelUri.scheme) || !supported.has(target.scheme)) {
                    throw new Error('ONNX GraphSurgeon editing requires a local or VS Code Remote filesystem.');
                }
                const script = path.join(this.context.extensionPath, 'media', 'onnx-graphsurgeon.py');
                await applyOnnxGraphSurgeonEdits(
                    script,
                    modelUri.fsPath || modelUri.path,
                    target.fsPath || target.path,
                    edits,
                    pythonPath
                );
            } else if (target.scheme !== modelUri.scheme ||
                target.authority !== modelUri.authority || target.path !== modelUri.path) {
                await vscode.workspace.fs.copy(modelUri, target, { overwrite: true });
            }
            await webview.postMessage({ type: 'graphEditSaveResult', path: target.path });
            vscode.window.showInformationMessage(`Saved ONNX model to ${target.path}`);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            await webview.postMessage({ type: 'graphEditSaveResult', error: message });
            throw error;
        }
    }

    async inferOnnxShapes(modelUri, webview, edits) {
        try {
            const pythonPath = await this.graphSurgeonPython(modelUri);
            if (!pythonPath) {
                await webview.postMessage({ type: 'graphEditShapeInferenceResult' });
                return;
            }
            const script = path.join(this.context.extensionPath, 'media', 'onnx-graphsurgeon.py');
            const result = await inferOnnxGraphSurgeonShapes(
                script,
                modelUri.fsPath || modelUri.path,
                edits,
                pythonPath
            );
            await webview.postMessage({
                type: 'graphEditShapeInferenceResult',
                ...result
            });
        } catch (error) {
            await webview.postMessage({
                type: 'graphEditShapeInferenceResult',
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }

    async graphSurgeonPython(modelUri) {
        try {
            const configuration = vscode.workspace.getConfiguration('hnnx', modelUri);
            return await findPython(configuration.get('pythonPath', ''));
        } catch (error) {
            if (!error || error.code !== 'GRAPH_SURGEON_PYTHON_NOT_FOUND') {
                throw error;
            }
            const action = await vscode.window.showErrorMessage(
                'HNNX could not find a Python environment containing ONNX and NVIDIA ONNX GraphSurgeon.',
                'Create Recommended Environment',
                'Enter Python Path',
                'Open Settings'
            );
            if (action === 'Create Recommended Environment') {
                return await this.createGraphSurgeonEnvironment(modelUri);
            } else if (action === 'Enter Python Path') {
                return await this.configureGraphSurgeonPython(modelUri);
            }
            if (action === 'Open Settings') {
                await vscode.commands.executeCommand(
                    'workbench.action.openSettings',
                    'hnnx.pythonPath'
                );
            }
            return '';
        }
    }

    async configureGraphSurgeonPython(resource) {
        const configuration = vscode.workspace.getConfiguration('hnnx', resource);
        const current = configuration.get('pythonPath', '');
        const candidate = await vscode.window.showInputBox({
            title: 'HNNX: GraphSurgeon Python',
            prompt: "Enter the Python interpreter path containing 'onnx' and 'onnx_graphsurgeon'.",
            value: current,
            placeHolder: '~/.hnnx/venv/bin/python3',
            ignoreFocusOut: true
        });
        if (!candidate) {
            return '';
        }
        if (!await validatePython(candidate)) {
            const action = await vscode.window.showErrorMessage(
                `The selected interpreter cannot import ONNX GraphSurgeon: ${candidate}`,
                'Open Settings'
            );
            if (action === 'Open Settings') {
                await vscode.commands.executeCommand(
                    'workbench.action.openSettings',
                    'hnnx.pythonPath'
                );
            }
            return '';
        }
        const python = normalizePython(candidate);
        await configuration.update('pythonPath', python, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`HNNX GraphSurgeon Python: ${python}`);
        return python;
    }

    async createGraphSurgeonEnvironment(resource) {
        const directory = recommendedEnvironmentDirectory();
        const action = await vscode.window.showInformationMessage(
            `Create the recommended ONNX GraphSurgeon environment at ${directory}?`,
            {
                modal: true,
                detail: "HNNX will install 'onnx' and 'onnx_graphsurgeon'. This may take a few minutes and requires internet access."
            },
            'Create and Install'
        );
        if (action !== 'Create and Install') {
            return '';
        }
        try {
            const python = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'HNNX: Creating ONNX GraphSurgeon environment',
                cancellable: false
            }, async (progress) => await createGraphSurgeonEnvironment({
                directory,
                report: (message) => {
                    progress.report({ message });
                    this.output.appendLine(message);
                }
            }));
            const configuration = vscode.workspace.getConfiguration('hnnx', resource);
            await configuration.update('pythonPath', python, vscode.ConfigurationTarget.Global);
            vscode.window.showInformationMessage(`HNNX GraphSurgeon Python is ready: ${python}`);
            return python;
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.output.appendLine(`GraphSurgeon environment setup failed: ${message}`);
            const retry = await vscode.window.showErrorMessage(
                `HNNX could not create the GraphSurgeon environment: ${message}`,
                'Enter Python Path',
                'Show Output'
            );
            if (retry === 'Enter Python Path') {
                return await this.configureGraphSurgeonPython(resource);
            }
            if (retry === 'Show Output') {
                this.output.show(true);
            }
            return '';
        }
    }

    async findEncodings(modelUri) {
        const filename = path.posix.basename(modelUri.path);
        const stem = filename.replace(/\.onnx$/i, '');
        const directory = vscode.Uri.joinPath(modelUri, '..');
        const candidates = [
            `${stem}.encodings`,
            `${filename}.encodings`,
            `${stem}.encodings.json`
        ];
        for (const candidate of candidates) {
            const uri = vscode.Uri.joinPath(directory, candidate);
            try {
                const stat = await vscode.workspace.fs.stat(uri);
                if ((stat.type & vscode.FileType.File) !== 0) {
                    return uri;
                }
            } catch {
                // Try the next conventional filename.
            }
        }
        return null;
    }

    resolveSibling(modelUri, requested) {
        if (typeof requested !== 'string' || requested.length === 0 || requested.includes('\0')) {
            throw new Error('Invalid external data filename.');
        }
        const normalized = path.posix.normalize(requested.replaceAll('\\', '/'));
        if (normalized === '..' || normalized.startsWith('../') || normalized.startsWith('/')) {
            throw new Error(`External data path escapes the model directory: ${requested}`);
        }
        return vscode.Uri.joinPath(modelUri, '..', ...normalized.split('/'));
    }

    arrayBuffer(data) {
        return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    }

    html(webview) {
        const mediaUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media'));
        const source = webview.cspSource;
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${source} data: blob:; style-src ${source} 'unsafe-inline'; font-src ${source}; script-src ${source}; worker-src ${source} blob:;">
<meta name="version" content="${manifest.version}">
<meta name="date" content="">
<meta name="type" content="VS Code">
<base href="${mediaUri}/">
<title>HNNX</title>
<link rel="stylesheet" type="text/css" href="grapher.css">
${this.styles()}
<script type="module" src="vscode-entry.js"></script>
</head>
<body class="welcome spinner">
${this.body()}
</body>
</html>`;
    }

    body() {
        const html = this.template();
        const match = html.match(/<body[^>]*>([\s\S]*)<\/body>/i);
        if (!match) {
            throw new Error('Bundled Netron HTML is invalid.');
        }
        return match[1].replace(/<input[^>]+id="open-file-dialog"[^>]*>/i, '');
    }

    styles() {
        const matches = this.template().match(/<style[^>]*>[\s\S]*?<\/style>/gi);
        return matches ? matches.join('\n') : '';
    }

    template() {
        if (!this._template) {
            this._template = require('fs').readFileSync(path.join(this.context.extensionPath, 'media', 'index.html'), 'utf8');
        }
        return this._template;
    }
}

const execute = (command, args) => new Promise((resolve, reject) => {
    const child = childProcess.spawn(command, args, {
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (data) => {
        stdout += data.toString();
    });
    child.stderr.on('data', (data) => {
        stderr += data.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
        if (code === 0) {
            resolve(stdout);
        } else {
            reject(new Error(stderr.trim() || `NVIDIA ONNX GraphSurgeon exited with code ${code}.`));
        }
    });
});

const normalizePython = (candidate) => candidate && candidate.startsWith('~/') ?
    path.join(os.homedir(), candidate.slice(2)) : candidate;

const recommendedEnvironmentDirectory = () => path.join(os.homedir(), '.hnnx', 'venv');

const recommendedEnvironmentPython = (directory = recommendedEnvironmentDirectory(), platform = process.platform) =>
    platform === 'win32' ? path.join(directory, 'Scripts', 'python.exe') : path.join(directory, 'bin', 'python3');

const findPythonForVenv = async (runner = execute, platform = process.platform) => {
    const candidates = platform === 'win32' ? [
        { command: 'py', args: ['-3'] },
        { command: 'python3', args: [] },
        { command: 'python', args: [] }
    ] : [
        { command: 'python3', args: [] },
        { command: 'python', args: [] }
    ];
    for (const candidate of candidates) {
        try {
            await runner(candidate.command, [...candidate.args, '-c', 'import sys']);
            return candidate;
        } catch {
            // Try the next Python launcher available on the extension host.
        }
    }
    throw new Error('Python 3 was not found on the VS Code extension host. Install Python 3 or enter an existing interpreter path.');
};

const createGraphSurgeonEnvironment = async (options = {}) => {
    const runner = options.runner || execute;
    const directory = options.directory || recommendedEnvironmentDirectory();
    const interpreter = recommendedEnvironmentPython(directory, options.platform || process.platform);
    const report = options.report || (() => {});
    report('Finding Python 3…');
    const bootstrap = options.bootstrap || await findPythonForVenv(runner, options.platform || process.platform);
    report(`Creating ${directory}…`);
    await runner(bootstrap.command, [...bootstrap.args, '-m', 'venv', directory]);
    report('Upgrading pip…');
    await runner(interpreter, ['-m', 'pip', 'install', '--upgrade', 'pip']);
    report('Installing ONNX and NVIDIA ONNX GraphSurgeon…');
    await runner(interpreter, [
        '-m', 'pip', 'install', 'onnx', 'onnx_graphsurgeon',
        '--extra-index-url', 'https://pypi.ngc.nvidia.com'
    ]);
    const validate = options.validate || validatePython;
    if (!await validate(interpreter)) {
        throw new Error('The environment was created, but ONNX GraphSurgeon could not be imported.');
    }
    report('Environment ready.');
    return interpreter;
};

const pythonCandidates = (configured) => {
    const candidates = [];
    if (configured) {
        candidates.push(configured);
    }
    if (process.env.HNNX_PYTHON) {
        candidates.push(process.env.HNNX_PYTHON);
    }
    if (process.platform === 'win32') {
        candidates.push(path.join(os.homedir(), '.hnnx', 'venv', 'Scripts', 'python.exe'));
        candidates.push('python.exe', 'python', 'py.exe', 'py', 'python3.exe', 'python3');
    } else {
        candidates.push(path.join(os.homedir(), '.hnnx', 'venv', 'bin', 'python3'));
        candidates.push('/opt/homebrew/bin/python3', '/usr/local/bin/python3', 'python3', 'python');
    }
    return Array.from(new Set(candidates.map((candidate) => normalizePython(candidate))));
};

const validatePython = async (candidate) => {
    if (!candidate || typeof candidate !== 'string') {
        return false;
    }
    try {
        await execute(normalizePython(candidate), ['-c', 'import onnx, onnx_graphsurgeon']);
        return true;
    } catch {
        return false;
    }
};

const findPython = async (configured) => {
    for (const candidate of pythonCandidates(configured)) {
        if (await validatePython(candidate)) {
            return candidate;
        }
    }
    const error = new Error(
        'A Python environment with NVIDIA ONNX GraphSurgeon was not found. ' +
        'Configure hnnx.pythonPath.'
    );
    error.code = 'GRAPH_SURGEON_PYTHON_NOT_FOUND';
    throw error;
};

const applyOnnxGraphSurgeonEdits = async (script, input, output, edits, configuredPython = '') => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hnnx-'));
    const editsPath = path.join(directory, 'edits.json');
    try {
        await fs.promises.writeFile(editsPath, JSON.stringify(edits), 'utf-8');
        const python = await findPython(configuredPython);
        await execute(python, [
            script,
            '--mode', 'edit',
            '--input', input,
            '--output', output,
            '--edits', editsPath
        ]);
    } finally {
        await fs.promises.rm(directory, { recursive: true, force: true });
    }
};

const inferOnnxGraphSurgeonShapes = async (script, input, edits, configuredPython = '') => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hnnx-'));
    const editsPath = path.join(directory, 'edits.json');
    try {
        await fs.promises.writeFile(editsPath, JSON.stringify(edits), 'utf-8');
        const python = await findPython(configuredPython);
        const output = await execute(python, [
            script,
            '--mode', 'infer',
            '--input', input,
            '--edits', editsPath
        ]);
        return JSON.parse(output);
    } finally {
        await fs.promises.rm(directory, { recursive: true, force: true });
    }
};

function activate(context) {
    const provider = new OnnxEditorProvider(context);
    context.subscriptions.push(provider.output);
    context.subscriptions.push(vscode.commands.registerCommand(
        'hnnx.configureGraphSurgeonPython',
        async () => await provider.configureGraphSurgeonPython(
            vscode.window.activeTextEditor ? vscode.window.activeTextEditor.document.uri : undefined
        )
    ));
    context.subscriptions.push(vscode.commands.registerCommand(
        'hnnx.createGraphSurgeonEnvironment',
        async () => await provider.createGraphSurgeonEnvironment(
            vscode.window.activeTextEditor ? vscode.window.activeTextEditor.document.uri : undefined
        )
    ));
    context.subscriptions.push(vscode.window.registerCustomEditorProvider(viewType, provider, {
        webviewOptions: {
            retainContextWhenHidden: true
        },
        supportsMultipleEditorsPerDocument: false
    }));
}

function deactivate() {
}

module.exports = {
    activate,
    deactivate,
    OnnxEditorProvider,
    applyOnnxGraphSurgeonEdits,
    inferOnnxGraphSurgeonShapes,
    findPython,
    validatePython,
    findPythonForVenv,
    createGraphSurgeonEnvironment,
    recommendedEnvironmentPython
};
