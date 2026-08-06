
import * as aimet from './aimet.js';
import * as base from './base.js';
import * as grapher from './grapher.js';

const view = {};
const markdown = {};
const png = {};
const metadata = {};
const metrics = {};

view.GraphEditCatalog = [
    { kind: 'input', name: 'Graph Input', category: 'Interface',
        description: 'Add a typed external input to the ONNX graph.' },
    { kind: 'output', name: 'Graph Output', category: 'Interface',
        description: 'Expose an existing tensor as a graph output.' },
    { kind: 'node', name: 'Identity', category: 'Core', inputs: ['input'], outputs: ['output'] },
    { kind: 'node', name: 'Relu', category: 'Activation', inputs: ['X'], outputs: ['Y'] },
    { kind: 'node', name: 'Sigmoid', category: 'Activation', inputs: ['X'], outputs: ['Y'] },
    { kind: 'node', name: 'Tanh', category: 'Activation', inputs: ['input'], outputs: ['output'] },
    { kind: 'node', name: 'Erf', category: 'Activation', inputs: ['input'], outputs: ['output'] },
    { kind: 'node', name: 'Softmax', category: 'Activation', inputs: ['input'], outputs: ['output'],
        attributes: [{ name: 'axis', type: 'integer', value: -1 }] },
    { kind: 'node', name: 'LayerNormalization', category: 'Activation', inputs: ['X', 'Scale', 'B'], optionalInputs: [2],
        outputs: ['Y'], shapeTransform: true,
        attributes: [{ name: 'axis', type: 'integer', value: -1 }, { name: 'epsilon', type: 'float', value: 0.00001 }] },
    { kind: 'node', name: 'Add', category: 'Math', inputs: ['A', 'B'], outputs: ['C'] },
    { kind: 'node', name: 'Sub', category: 'Math', inputs: ['A', 'B'], outputs: ['C'] },
    { kind: 'node', name: 'Mul', category: 'Math', inputs: ['A', 'B'], outputs: ['C'] },
    { kind: 'node', name: 'Div', category: 'Math', inputs: ['A', 'B'], outputs: ['C'] },
    { kind: 'node', name: 'Pow', category: 'Math', inputs: ['X', 'Y'], outputs: ['Z'] },
    { kind: 'node', name: 'Max', category: 'Math', inputs: ['data_0'], variadic: true, outputs: ['max'] },
    { kind: 'node', name: 'Min', category: 'Math', inputs: ['data_0'], variadic: true, outputs: ['min'] },
    { kind: 'node', name: 'Neg', category: 'Math', inputs: ['X'], outputs: ['Y'] },
    { kind: 'node', name: 'Exp', category: 'Math', inputs: ['input'], outputs: ['output'] },
    { kind: 'node', name: 'Log', category: 'Math', inputs: ['input'], outputs: ['output'] },
    { kind: 'node', name: 'Sqrt', category: 'Math', inputs: ['X'], outputs: ['Y'] },
    { kind: 'node', name: 'Reciprocal', category: 'Math', inputs: ['X'], outputs: ['Y'] },
    { kind: 'node', name: 'MatMul', category: 'Math', inputs: ['A', 'B'], outputs: ['Y'], shapeTransform: true },
    { kind: 'node', name: 'Equal', category: 'Logic', inputs: ['A', 'B'], outputs: ['C'], typeTransform: true },
    { kind: 'node', name: 'Greater', category: 'Logic', inputs: ['A', 'B'], outputs: ['C'], typeTransform: true },
    { kind: 'node', name: 'Less', category: 'Logic', inputs: ['A', 'B'], outputs: ['C'], typeTransform: true },
    { kind: 'node', name: 'Not', category: 'Logic', inputs: ['X'], outputs: ['Y'], typeTransform: true },
    { kind: 'node', name: 'Where', category: 'Logic', inputs: ['condition', 'X', 'Y'], outputs: ['output'] },
    { kind: 'node', name: 'Clip', category: 'Activation', inputs: ['input', 'min', 'max'], optionalInputs: [1, 2], outputs: ['output'] },
    { kind: 'node', name: 'Concat', category: 'Tensor', inputs: ['inputs'], variadic: true, outputs: ['concat_result'],
        attributes: [{ name: 'axis', type: 'integer', value: 0 }], shapeTransform: true },
    { kind: 'node', name: 'Split', category: 'Tensor', inputs: ['input', 'split'], optionalInputs: [1],
        outputs: ['outputs'], variadicOutputs: true, split: true, shapeTransform: true,
        description: 'Split a tensor into multiple outputs using ONNX opset 13–17 semantics.',
        attributes: [{ name: 'axis', type: 'integer', value: 0 }] },
    { kind: 'node', name: 'Reshape', category: 'Tensor', inputs: ['data', 'shape'], outputs: ['reshaped'], shapeTransform: true },
    { kind: 'node', name: 'Transpose', category: 'Tensor', inputs: ['data'], outputs: ['transposed'],
        attributes: [{ name: 'perm', type: 'integers', value: '' }], shapeTransform: true },
    { kind: 'node', name: 'Flatten', category: 'Tensor', inputs: ['input'], outputs: ['output'],
        attributes: [{ name: 'axis', type: 'integer', value: 1 }], shapeTransform: true },
    { kind: 'node', name: 'Squeeze', category: 'Tensor', inputs: ['data', 'axes'], optionalInputs: [1], axesInput: 'optional',
        outputs: ['squeezed'], shapeTransform: true,
        description: 'Remove singleton dimensions. Optional axes are encoded as an int64 ONNX constant.' },
    { kind: 'node', name: 'Unsqueeze', category: 'Tensor', inputs: ['data', 'axes'], axesInput: 'required',
        outputs: ['expanded'], shapeTransform: true,
        description: 'Insert singleton dimensions using an int64 axes constant.' },
    { kind: 'node', name: 'Expand', category: 'Tensor', inputs: ['input', 'shape'], expand: true,
        outputs: ['output'], shapeTransform: true,
        description: 'Broadcast a tensor to a target shape encoded as an int64 ONNX constant.' },
    { kind: 'node', name: 'Slice', category: 'Tensor', inputs: ['data', 'starts', 'ends', 'axes', 'steps'], optionalInputs: [3, 4], slice: true,
        outputs: ['output'], shapeTransform: true,
        description: 'Slice with int64 starts, ends, and optional axes and steps constants.' },
    { kind: 'node', name: 'Shape', category: 'Tensor', inputs: ['data'], outputs: ['shape'], shapeTransform: true },
    { kind: 'node', name: 'Gather', category: 'Tensor', inputs: ['data', 'indices'], outputs: ['output'],
        attributes: [{ name: 'axis', type: 'integer', value: 0 }], shapeTransform: true },
    { kind: 'node', name: 'GatherElements', category: 'Tensor', inputs: ['data', 'indices'], outputs: ['output'],
        attributes: [{ name: 'axis', type: 'integer', value: 0 }], shapeTransform: true },
    { kind: 'node', name: 'Tile', category: 'Tensor', inputs: ['input', 'repeats'], outputs: ['output'], shapeTransform: true },
    { kind: 'node', name: 'TopK', category: 'Reduction', inputs: ['X', 'K'], outputs: ['Values', 'Indices'],
        topk: true, shapeTransform: true, outputDataTypes: [null, 'int64'],
        description: 'Return the top K values and their int64 indices using ONNX opset 17 semantics.',
        attributes: [
            { name: 'axis', type: 'integer', value: -1 },
            { name: 'largest', type: 'integer', value: 1 },
            { name: 'sorted', type: 'integer', value: 1 }
        ] },
    { kind: 'node', name: 'ReduceMean', category: 'Reduction', inputs: ['data'], outputs: ['reduced'], shapeTransform: true,
        attributes: [{ name: 'axes', type: 'integers', value: '-1' }, { name: 'keepdims', type: 'integer', value: 1 }] },
    { kind: 'node', name: 'ReduceSum', category: 'Reduction', inputs: ['data', 'axes'], optionalInputs: [1],
        axesInput: 'optionalDefaultLast', outputs: ['reduced'], shapeTransform: true,
        attributes: [{ name: 'keepdims', type: 'integer', value: 1 }] },
    { kind: 'node', name: 'ReduceMax', category: 'Reduction', inputs: ['data'], outputs: ['reduced'], shapeTransform: true,
        attributes: [{ name: 'axes', type: 'integers', value: '-1' }, { name: 'keepdims', type: 'integer', value: 1 }] },
    { kind: 'node', name: 'ArgMax', category: 'Reduction', inputs: ['data'], outputs: ['reduced'],
        shapeTransform: true, outputDataTypes: ['int64'],
        attributes: [
            { name: 'axis', type: 'integer', value: 0 },
            { name: 'keepdims', type: 'integer', value: 1 },
            { name: 'select_last_index', type: 'integer', value: 0 }
        ] },
    { kind: 'node', name: 'ArgMin', category: 'Reduction', inputs: ['data'], outputs: ['reduced'],
        shapeTransform: true, outputDataTypes: ['int64'],
        attributes: [
            { name: 'axis', type: 'integer', value: 0 },
            { name: 'keepdims', type: 'integer', value: 1 },
            { name: 'select_last_index', type: 'integer', value: 0 }
        ] },
    { kind: 'node', name: 'Cast', category: 'Type', inputs: ['input'], outputs: ['output'], typeTransform: true,
        attributes: [{ name: 'to', type: 'datatype', value: 1 }] }
];

view.View = class {

    constructor(host) {
        this._host = host;
        this._defaultOptions = {
            weights: true,
            attributes: false,
            names: false,
            direction: 'vertical',
            mousewheel: 'scroll'
        };
        this._options = { ...this._defaultOptions };
        this._themePreference = 'auto';
        this._themeMediaRules = null;
        this._events = {};
        this._events.selectionchange = () => this._selectionChangeHandler();
        this._model = null;
        this._path = [];
        this._selection = [];
        this._sidebar = new view.Sidebar(this._host);
        this._find = null;
        this._graphEdit = {
            enabled: false,
            pending: null,
            connection: null,
            nodeMenu: null,
            undo: [],
            redo: [],
            viewDirty: false,
            viewRevision: 0,
            redrawing: false,
            renderPromise: null,
            historyPromise: null,
            historyPending: 0,
            positions: new Map(),
            drag: null,
            linkDrag: null,
            addDialog: false,
            warningResolve: null,
            encodingsDisabled: false,
            nextAddedId: 1
        };
        this._modelFactoryService = new view.ModelFactoryService(this._host);
        this._modelFactoryService.import();
        this._worker = this._host.environment('serial') ? null : new view.Worker(this._host);
    }

    async start() {
        try {
            const zip = await import('./zip.js');
            await zip.Archive.import();
            await this._host.view(this);
            const platform = this._host.environment('platform');
            const options = this._host.get('options') || {};
            for (const [name, value] of Object.entries(options)) {
                this._options[name] = value;
            }
            this._element('sidebar-model-button').addEventListener('click', () => {
                this.showModelProperties();
            });
            this._element('sidebar-info-button').addEventListener('click', () => {
                this.showModelStatistics();
            });
            this._element('sidebar-target-button').addEventListener('click', () => {
                this.showTargetProperties();
            });
            this._element('zoom-in-button').addEventListener('click', () => {
                this.zoomIn();
            });
            this._element('zoom-out-button').addEventListener('click', () => {
                this.zoomOut();
            });
            this._element('graph-edit-button').addEventListener('click', async () => {
                if (this._graphEdit.enabled) {
                    await this.exitGraphEdit();
                } else {
                    await this.enterGraphEdit();
                }
            });
            this._element('graph-edit-warning-cancel').addEventListener('click', () => {
                this._closeGraphEditEncodingWarning(false);
            });
            this._element('graph-edit-warning-continue').addEventListener('click', () => {
                this._closeGraphEditEncodingWarning(true);
            });
            this._element('graph-edit-warning-overlay').addEventListener('pointerdown', (event) => {
                if (event.target === this._element('graph-edit-warning-overlay')) {
                    this._closeGraphEditEncodingWarning(false);
                }
            });
            this._element('graph-edit-add-button').addEventListener('click', () => {
                this.showGraphEditAddDialog();
            });
            this._element('graph-edit-add-close').addEventListener('click', () => {
                this._closeGraphEditAddDialog();
            });
            this._element('graph-edit-add-overlay').addEventListener('pointerdown', (event) => {
                if (event.target === this._element('graph-edit-add-overlay')) {
                    this._closeGraphEditAddDialog();
                }
            });
            this._element('graph-edit-undo-button').addEventListener('click', async () => {
                await this.undoGraphEdit();
            });
            this._element('graph-edit-redo-button').addEventListener('click', async () => {
                await this.redoGraphEdit();
            });
            this._element('graph-edit-reset-button').addEventListener('click', async () => {
                await this.resetGraphEdit();
            });
            this._element('graph-edit-layout-button').addEventListener('click', async () => {
                await this.relayoutGraph();
            });
            this._element('graph-edit-redraw-button').addEventListener('click', async () => {
                await this.redrawGraphEdit();
            });
            this._element('graph-edit-infer-button').addEventListener('click', async () => {
                await this.inferGraphShapes();
            });
            this._element('graph-edit-save-button').addEventListener('click', async () => {
                await this.saveOnnxAs();
            });
            this._element('graph-edit-connection-replace').addEventListener('click', () => {
                this.beginGraphEditConnectionReplace();
            });
            this._element('graph-edit-connection-disconnect').addEventListener('click', async () => {
                await this.disconnectGraphEditConnection();
            });
            this._element('graph-edit-connection-cancel').addEventListener('click', () => {
                this._cancelGraphEditSelection();
            });
            const cancelGraphEditOnRightClick = () => {
                if (this._graphEdit.enabled &&
                    (this._graphEdit.pending || this._graphEdit.connection || this._graphEdit.nodeMenu)) {
                    this._cancelGraphEditSelection();
                }
            };
            this._host.document.addEventListener('mousedown', (event) => {
                if (event.button === 2) {
                    cancelGraphEditOnRightClick();
                }
            }, true);
            this._element('target').addEventListener('contextmenu', (event) => {
                if (this._graphEdit.enabled) {
                    event.preventDefault();
                }
            }, true);
            this._host.document.addEventListener('pointerdown', (event) => {
                const menu = this._element('graph-edit-node-menu');
                if (menu && menu.classList.contains('visible') && !menu.contains(event.target)) {
                    this._closeGraphEditNodeMenu();
                }
            }, true);
            this._element('toolbar-path-back-button').addEventListener('click', async () => {
                await this.popTarget();
            });
            this._element('sidebar').addEventListener('mousewheel', (e) => {
                if (e.shiftKey || e.ctrlKey) {
                    e.preventDefault();
                }
            }, { passive: false });
            this._host.document.addEventListener('keydown', (e) => {
                if (this._target && !e.metaKey && !e.ctrlKey) {
                    this._target.select(null);
                }
            });
            this._host.document.addEventListener('keydown', (e) => {
                const element = e.target;
                const tagName = element && typeof element.tagName === 'string' ?
                    element.tagName.toLowerCase() : '';
                const editable = tagName === 'input' || tagName === 'textarea' ||
                    tagName === 'select' || Boolean(element && element.isContentEditable);
                if (editable || e.isComposing || this._graphEdit.warningResolve ||
                    this._graphEdit.addDialog || this._graphEdit.nodeMenu) {
                    return;
                }
                const modifier = platform === 'darwin' ? e.metaKey : e.ctrlKey;
                const code = e.code || '';
                let action = null;
                if (modifier && code === 'KeyS' && !e.shiftKey && !e.altKey && this.activeTarget) {
                    action = () => this.saveOnnxAs();
                } else if (!e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey) {
                    if (code === 'KeyE' && !this._graphEdit.enabled) {
                        action = () => this.enterGraphEdit();
                    } else if (code === 'KeyV' && this._graphEdit.enabled) {
                        action = () => this.exitGraphEdit();
                    } else if (code === 'KeyR') {
                        action = () => this.relayoutGraph();
                    } else if (code === 'KeyI') {
                        action = () => this.inferGraphShapes();
                    }
                }
                if (action) {
                    e.preventDefault();
                    e.stopPropagation();
                    action();
                }
            }, true);
            this._host.document.addEventListener('keydown', (e) => {
                if (!this._graphEdit.enabled) {
                    return;
                }
                const element = e.target;
                const tagName = element && typeof element.tagName === 'string' ?
                    element.tagName.toLowerCase() : '';
                const editable = tagName === 'input' || tagName === 'textarea' ||
                    tagName === 'select' || Boolean(element && element.isContentEditable);
                const modifier = platform === 'darwin' ? e.metaKey : e.ctrlKey;
                if (modifier && (e.code === 'KeyZ' || e.key.toLowerCase() === 'z')) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (e.shiftKey) {
                        this.redoGraphEdit();
                    } else {
                        this.undoGraphEdit();
                    }
                } else if ((e.code === 'KeyQ' || e.key.toLowerCase() === 'q') && !editable &&
                    !e.metaKey && !e.ctrlKey && !e.altKey &&
                    (this._graphEdit.pending || this._graphEdit.connection ||
                        this._graphEdit.nodeMenu || this._graphEdit.linkDrag ||
                        this._graphEdit.addDialog)) {
                    e.preventDefault();
                    e.stopPropagation();
                    this._cancelGraphEditSelection();
                } else {
                    const deleteKey = e.code === 'KeyD' || e.key.toLowerCase() === 'd';
                    if (deleteKey && !editable && !e.metaKey && !e.ctrlKey && !e.altKey) {
                        e.preventDefault();
                        e.stopPropagation();
                        const selected = this._graphEdit.nodeMenu;
                        if (selected && Number.isInteger(selected._graph_edit_output_index)) {
                            this.deleteGraphEditGraphOutput(selected);
                        } else if (selected &&
                            (Number.isInteger(selected._graph_edit_input_index) || selected._graph_edit_data_type)) {
                            this.deleteGraphEditGraphInput(selected);
                        } else if (selected && selected.source) {
                            this.deleteGraphEditNode(selected);
                        } else if (this._graphEdit.connection) {
                            this.disconnectGraphEditConnection();
                        }
                    }
                }
            }, true);
            this._host.document.addEventListener('copy', (e) => {
                const selection = this._host.document.getSelection();
                if (!selection || selection.toString().trim() === '') {
                    if (this._target && this._target.selection.size > 0) {
                        const names = [];
                        for (const element of this._target.selection) {
                            if (element.value && element.value.name) {
                                names.push(element.value.name);
                            }
                        }
                        if (names.length > 0) {
                            e.clipboardData.setData('text/plain', names.join('\n'));
                            e.preventDefault();
                        }
                    }
                }
            });
            if (this._host.type === 'Electron') {
                this._host.update({ 'copy.enabled': false });
                this._host.document.addEventListener('selectionchange', this._events.selectionchange);
            }
            this._menu = new view.Menu(this._host);
            this._menu.add({
                accelerator: platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11',
                execute: async () => await this._host.execute('fullscreen')
            });
            if (this._host.environment('menu')) {
                const menu = this._element('menu');
                const button = this._element('menu-button');
                this._menu.attach(menu, button);
                const file = this._menu.group('&File');
                file.add({
                    label: '&Open...',
                    accelerator: 'CmdOrCtrl+O',
                    execute: async () => await this._host.execute('open')
                });
                if (this._host.type === 'Electron') {
                    this._recents = file.group('Open &Recent');
                    file.add({
                        label: '&Export...',
                        accelerator: 'CmdOrCtrl+Shift+E',
                        execute: async () => await this._host.execute('export'),
                        enabled: () => this.activeTarget
                    });
                    file.add({
                        label: platform === 'darwin' ? '&Close Window' : '&Close',
                        accelerator: 'CmdOrCtrl+W',
                        execute: async () => await this._host.execute('close'),
                    });
                    file.add({
                        label: platform === 'win32' ? 'E&xit' : '&Quit',
                        accelerator: platform === 'win32' ? '' : 'CmdOrCtrl+Q',
                        execute: async () => await this._host.execute('quit'),
                    });
                } else {
                    file.add({
                        label: 'Export as &PNG',
                        accelerator: 'CmdOrCtrl+Shift+E',
                        execute: async () => await this.export(`${this._host.document.title}.png`),
                        enabled: () => this.activeTarget
                    });
                    file.add({
                        label: 'Export as &SVG',
                        accelerator: 'CmdOrCtrl+Alt+E',
                        execute: async () => await this.export(`${this._host.document.title}.svg`),
                        enabled: () => this.activeTarget
                    });
                }
                const edit = this._menu.group('&Edit');
                edit.add({
                    label: '&Find...',
                    accelerator: 'CmdOrCtrl+F',
                    execute: () => this.find(),
                    enabled: () => this.activeTarget
                });
                const view = this._menu.group('&View');
                view.add({
                    label: () => this.options.attributes ? 'Hide &Attributes' : 'Show &Attributes',
                    accelerator: 'CmdOrCtrl+D',
                    execute: () => this.toggle('attributes'),
                    enabled: () => this.activeTarget
                });
                view.add({
                    label: () => this.options.weights ? 'Hide &Weights' : 'Show &Weights',
                    accelerator: 'CmdOrCtrl+I',
                    execute: () => this.toggle('weights'),
                    enabled: () => this.activeTarget
                });
                view.add({
                    label: () => this.options.names ? 'Hide &Names' : 'Show &Names',
                    accelerator: 'CmdOrCtrl+U',
                    execute: () => this.toggle('names'),
                    enabled: () => this.activeTarget
                });
                view.add({
                    label: () => this.options.direction === 'vertical' ? 'Show &Horizontal' : 'Show &Vertical',
                    accelerator: 'CmdOrCtrl+K',
                    execute: () => this.toggle('direction'),
                    enabled: () => this.activeTarget
                });
                view.add({
                    label: () => this.options.mousewheel === 'scroll' ? '&Mouse Wheel: Zoom' : '&Mouse Wheel: Scroll',
                    accelerator: 'CmdOrCtrl+M',
                    execute: () => this.toggle('mousewheel'),
                    enabled: () => this.activeTarget
                });
                if (this._host.type === 'VS Code') {
                    const theme = view.group('&Theme');
                    for (const value of ['auto', 'light', 'dark']) {
                        const name = value.charAt(0).toUpperCase() + value.slice(1);
                        theme.add({
                            label: () => `${this._themePreference === value ? '✓ ' : ''}${name}`,
                            execute: async () => await this._host.execute('set-theme', value)
                        });
                    }
                }
                view.add({});
                if (this._host.type === 'Electron') {
                    view.add({
                        label: '&Reload',
                        accelerator: platform === 'darwin' ? 'CmdOrCtrl+R' : 'F5',
                        execute: async () => await this._host.execute('reload'),
                        enabled: () => this.activeTarget
                    });
                    view.add({});
                }
                view.add({
                    label: 'Zoom &In',
                    accelerator: 'Shift+Up',
                    execute: () => this.zoomIn(),
                    enabled: () => this.activeTarget && this.target
                });
                view.add({
                    label: 'Zoom &Out',
                    accelerator: 'Shift+Down',
                    execute: () => this.zoomOut(),
                    enabled: () => this.activeTarget && this.target
                });
                view.add({
                    label: 'Actual &Size',
                    execute: () => this.resetZoom(),
                    enabled: () => this.activeTarget && this.target
                });
                view.add({});
                view.add({
                    label: '&Properties...',
                    accelerator: 'CmdOrCtrl+Enter',
                    execute: () => this.showTargetProperties(),
                    enabled: () => this.activeTarget
                });
                if (this._host.type === 'Electron' && !this._host.environment('packaged')) {
                    view.add({});
                    view.add({
                        label: '&Developer Tools...',
                        accelerator: 'CmdOrCtrl+Alt+I',
                        execute: async () => await this._host.execute('toggle-developer-tools')
                    });
                }
                const help = this._menu.group('&Help');
                help.add({
                    label: 'Report &Issue',
                    execute: async () => await this._host.execute('report-issue')
                });
                help.add({
                    label: `&About ${this._host.environment('name')}`,
                    execute: async () => await this._host.execute('about')
                });
            }
            const navigator = this._element('toolbar-navigator');
            this._select = new view.TargetSelector(this, navigator);
            this._select.on('change', (sender, target) => this._updateActiveTarget([target]));
            await this._host.start();
        } catch (error) {
            this.error(error, null, null);
        }
    }

    dispose() {
        if (this._worker) {
            this._worker.cancel(true);
        }
    }

    get host() {
        return this._host;
    }

    show(page) {
        if (!page) {
            page = (!this._model && !this.activeTarget) ? 'welcome' : 'default';
        }
        this._host.event('screen_view', {
            screen_name: page,
        });
        if (this._sidebar) {
            this._sidebar.close();
        }
        if (this._menu) {
            this._menu.close();
        }
        this._host.document.body.classList.remove(...Array.from(this._host.document.body.classList).filter((_) => _ !== 'active'));
        this._host.document.body.classList.add(...page.split(' '));
        if (this._target && page === 'default') {
            this._target.register();
        } else if (this._target) {
            this._target.unregister();
        }
        if (page === 'welcome') {
            const element = this._element('open-file-button');
            if (element) {
                element.focus();
            }
        }
        this._page = page;
        if (page === 'default') {
            this._updateGraphEditStatus();
        }
    }

    progress(percent) {
        const bar = this._element('progress-bar');
        if (bar) {
            bar.style.width = `${percent}%`;
        }
    }

    find() {
        if (this._target && this._sidebar.identifier !== 'find') {
            this._target.select(null);
            const sidebar = new view.FindSidebar(this, this._find, this.activeTarget, this.activeSignature);
            sidebar.on('state-changed', (sender, state) => {
                this._find = state;
            });
            sidebar.on('select', (sender, value) => {
                this._target.scrollTo(this._target.select([value], 'sidebar'));
            });
            sidebar.on('focus', (sender, value) => {
                this._target.focus([value]);
            });
            sidebar.on('blur', (sender, value) => {
                this._target.blur([value]);
            });
            sidebar.on('activate', (sender, value) => {
                this._target.scrollTo(this._target.activate(value, 'sidebar'));
            });
            this._sidebar.open(sidebar, 'Find');
        }
    }

    get model() {
        return this._model;
    }

    set model(value) {
        this._model = value;
    }

    get options() {
        return this._options;
    }

    get target() {
        return this._target;
    }

    set target(value) {
        if (this._target !== value) {
            if (this._target) {
                this._target.off('selectionchange', this._events.selectionchange);
                this._target.unregister();
            }
            const enabled = value ? true : false;
            this._host.update({
                'zoom-reset.enabled': enabled,
                'zoom-in.enabled': enabled,
                'zoom-out.enabled': enabled
            });
            this._target = value;
            if (this._target) {
                this._target.on('selectionchange', this._events.selectionchange);
                this._target.register();
            }
        }
    }

    toggle(name) {
        switch (name) {
            case 'names':
            case 'attributes':
            case 'weights':
                this._options[name] = !this._options[name];
                this._reload();
                break;
            case 'direction':
                this._options.direction = this._options.direction === 'vertical' ? 'horizontal' : 'vertical';
                this._reload();
                break;
            case 'mousewheel':
                this._options.mousewheel = this._options.mousewheel === 'scroll' ? 'zoom' : 'scroll';
                break;
            default:
                throw new view.Error(`Unsupported toggle '${name}'.`);
        }
        const options = {};
        for (const [name, value] of Object.entries(this._options)) {
            if (this._defaultOptions[name] !== value) {
                options[name] = value;
            }
        }
        if (Object.entries(options).length === 0) {
            this._host.delete('options');
        } else {
            this._host.set('options', options);
        }
    }

    setTheme(preference, effective) {
        const values = new Set(['auto', 'light', 'dark']);
        this._themePreference = values.has(preference) ? preference : 'auto';
        const resolved = effective === 'dark' ? 'dark' : 'light';
        const root = this._host.document.documentElement;
        root.setAttribute('data-theme', this._themePreference);
        root.setAttribute('data-effective-theme', resolved);
        if (!this._themeMediaRules) {
            this._themeMediaRules = [];
            const collect = (rules) => {
                for (const rule of rules) {
                    if (rule.media && typeof rule.media.mediaText === 'string' &&
                        rule.media.mediaText.includes('prefers-color-scheme: dark')) {
                        this._themeMediaRules.push(rule);
                    }
                    if (rule.cssRules) {
                        collect(rule.cssRules);
                    }
                }
            };
            for (const sheet of this._host.document.styleSheets) {
                try {
                    collect(sheet.cssRules);
                } catch {
                    // Ignore cross-origin stylesheets; HNNX theme rules are local.
                }
            }
        }
        for (const rule of this._themeMediaRules) {
            rule.media.mediaText = resolved === 'dark' ? 'all' : 'not all';
        }
    }

    recents(recents) {
        if (this._recents) {
            this._recents.clear();
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                this._recents.add({
                    label: recent.label,
                    accelerator: `CmdOrCtrl+${(i + 1)}`,
                    execute: () => this._host.execute('open', recent.path)
                });
            }
        }
    }

    _reload() {
        this.show('welcome spinner');
        if (this._model && this._path.length > 0) {
            this._updateTarget(this._model, this._path).catch((error) => {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            });
        }
    }

    _timeout(delay) {
        return new Promise((resolve) => {
            this._host.window.setTimeout(resolve, delay);
        });
    }

    _selectionChangeHandler() {
        if (this._host.type === 'Electron') {
            const selection = this._host.document.getSelection();
            const text = selection.rangeCount === 0 || selection.toString().trim() !== '';
            const graph = this._target && this._target.selection.size > 0;
            this._host.update({ 'copy.enabled': text || graph });
        }
    }

    _element(id) {
        return this._host.document.getElementById(id);
    }

    zoomIn() {
        this._target.zoom *= 1.05;
    }

    zoomOut() {
        this._target.zoom /= 1.05;
    }

    resetZoom() {
        this._target.zoom = 1;
    }

    _graphViewportAnchor() {
        if (!this._target || !this._target.nodes) {
            return null;
        }
        const document = this._host.document;
        const container = document.getElementById('target');
        const origin = document.getElementById('origin');
        const bounds = container ? container.getBoundingClientRect() : null;
        const matrix = origin ? origin.getScreenCTM() : null;
        if (!bounds || !matrix || !matrix.a || !matrix.d) {
            return null;
        }
        const x = (bounds.left + bounds.width / 2 - matrix.e) / matrix.a;
        const y = (bounds.top + bounds.height / 2 - matrix.f) / matrix.d;
        let nearest = null;
        let distance = Number.POSITIVE_INFINITY;
        for (const entry of this._target.nodes.values()) {
            const node = entry ? entry.label : null;
            if (!node || !node.value || !node.element || !Number.isFinite(node.x) || !Number.isFinite(node.y)) {
                continue;
            }
            const dx = node.x - x;
            const dy = node.y - y;
            const score = dx * dx + dy * dy;
            if (score < distance) {
                nearest = node;
                distance = score;
            }
        }
        if (nearest) {
            return { value: nearest.value, rect: nearest.element.getBoundingClientRect(), node: true };
        }
        return null;
    }

    get graphEditEnabled() {
        return this._graphEdit.enabled;
    }

    async enterGraphEdit() {
        if (!this.activeTarget || this._graphEdit.enabled) {
            return;
        }
        if (!await this._confirmGraphEditWithoutEncodings()) {
            return;
        }
        await this.toggleGraphEdit();
    }

    async exitGraphEdit() {
        if (!this._graphEdit.enabled) {
            return;
        }
        await this.toggleGraphEdit();
    }

    async toggleGraphEdit() {
        if (!this.activeTarget) {
            return;
        }
        const container = this._element('target');
        const origin = this._element('origin');
        const viewport = container ? { left: container.scrollLeft, top: container.scrollTop } : null;
        const originTransform = origin ? origin.getAttribute('transform') : null;
        const anchor = this._graphViewportAnchor();
        this._graphEdit.enabled = !this._graphEdit.enabled;
        const preparedInputs = this._graphEdit.enabled && this._prepareGraphEditInputs();
        if (preparedInputs) {
            this._preserveVisibleGraphEditPositions();
            await this.refresh(anchor, { animate: false });
        }
        if (!this._graphEdit.enabled) {
            this._cancelGraphEditSelection();
        }
        this._host.document.documentElement.classList.toggle('onnx-graph-edit', this._graphEdit.enabled);
        if (container && viewport) {
            await new Promise((resolve) => {
                this._host.window.requestAnimationFrame(() => {
                    if (origin && originTransform) {
                        origin.setAttribute('transform', originTransform);
                    }
                    container.scrollLeft = viewport.left;
                    container.scrollTop = viewport.top;
                    if (this._target) {
                        delete this._target._scrollLeft;
                        delete this._target._scrollTop;
                        this._target._updateScrollThumbs();
                    }
                    resolve();
                });
            });
        }
        const button = this._element('graph-edit-button');
        button.textContent = this._graphEdit.enabled ? 'VIEW' : 'EDIT';
        button.setAttribute('title', this._graphEdit.enabled ?
            'Exit ONNX GraphSurgeon Editor (Beta) (V)' :
            'Enter ONNX GraphSurgeon Editor (Beta) (E)');
        this._updateGraphEditStatus();
    }

    async _confirmGraphEditWithoutEncodings() {
        const quantization = this.model && this.model.attachment ? this.model.attachment.quantization : null;
        if (!quantization || quantization.empty) {
            return true;
        }
        const overlay = this._element('graph-edit-warning-overlay');
        if (!overlay) {
            return false;
        }
        overlay.classList.add('visible');
        this._element('graph-edit-warning-cancel').focus();
        const proceed = await new Promise((resolve) => {
            this._graphEdit.warningResolve = resolve;
        });
        if (!proceed) {
            return false;
        }
        this.model.attachment.quantization = new aimet.EncodingFile();
        this._graphEdit.encodingsDisabled = true;
        this._host.document.documentElement.classList.add('onnx-graph-edit-encodings-disabled');
        const anchor = this._graphViewportAnchor();
        await this.refresh(anchor, { animate: false });
        return true;
    }

    _closeGraphEditEncodingWarning(proceed) {
        const overlay = this._element('graph-edit-warning-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
        }
        const resolve = this._graphEdit.warningResolve;
        this._graphEdit.warningResolve = null;
        if (resolve) {
            resolve(Boolean(proceed));
        }
    }

    _prepareGraphEditInputs() {
        let changed = false;
        for (const node of this.activeTarget.nodes || []) {
            for (const argument of node.inputs || []) {
                const source = argument.source;
                if (!Array.isArray(argument.value) || argument.value.length > 0 ||
                    !source || !source.node || !Array.isArray(source.node.input) ||
                    !Array.isArray(source.indices) || source.indices.length !== 1) {
                    continue;
                }
                const index = source.indices[0];
                const value = source.node.input[index];
                if (value && !value.name && !value.initializer) {
                    argument.value.push(value);
                    changed = true;
                }
            }
        }
        return changed;
    }

    showGraphEditAddDialog() {
        if (!this._graphEdit.enabled || !this.activeTarget) {
            return;
        }
        this._cancelGraphEditSelection(false);
        const overlay = this._element('graph-edit-add-overlay');
        const search = this._element('graph-edit-add-search');
        const list = this._element('graph-edit-add-list');
        if (!overlay || !search || !list) {
            return;
        }
        this._graphEdit.addDialog = true;
        overlay.classList.add('visible');
        search.value = '';
        let selected = view.GraphEditCatalog.find((item) => item.name === 'Identity') || view.GraphEditCatalog[0];
        const render = () => {
            list.replaceChildren();
            const query = search.value.trim().toLowerCase();
            const items = view.GraphEditCatalog.filter((item) =>
                !query || `${item.name} ${item.category}`.toLowerCase().includes(query));
            items.forEach((item) => {
                const button = this._host.document.createElement('button');
                button.className = 'graph-edit-add-item';
                button.classList.toggle('selected', item === selected);
                button.type = 'button';
                button.textContent = item.name;
                const category = this._host.document.createElement('small');
                category.textContent = item.category;
                button.appendChild(category);
                button.addEventListener('click', () => {
                    selected = item;
                    render();
                    this._renderGraphEditAddForm(item);
                });
                list.appendChild(button);
            });
            if (items.length === 0) {
                const empty = this._host.document.createElement('div');
                empty.className = 'graph-edit-node-menu-empty';
                empty.textContent = 'No matching operator';
                list.appendChild(empty);
            }
        };
        search.oninput = render;
        render();
        this._renderGraphEditAddForm(selected);
        search.focus();
    }

    _closeGraphEditAddDialog() {
        const overlay = this._element('graph-edit-add-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
        }
        this._graphEdit.addDialog = false;
    }

    _graphEditUniqueName(base, names) {
        base = (base || 'item').replace(/[^A-Za-z0-9_.-]+/g, '_');
        let name = base;
        let suffix = 2;
        while (names.has(name)) {
            name = `${base}_${suffix++}`;
        }
        return name;
    }

    _graphEditTensorType(dataType, dimensions) {
        const shape = {
            dimensions,
            toString: () => dimensions.length === 0 ? '' :
                `[${dimensions.map((dimension) => dimension === null ? '?' : dimension).join(',')}]`
        };
        return {
            dataType,
            shape,
            toString: () => `${dataType}${shape.toString()}`
        };
    }

    _graphEditTensorValue(name, type = null, initializer = null) {
        let identifier = name;
        return {
            get name() {
                return identifier;
            },
            rename(value) {
                if (typeof value === 'string' && value) {
                    identifier = value;
                }
            },
            get type() {
                return this._inferredType || type;
            },
            applyShapeInference(dataType, dimensions) {
                this._inferredType = {
                    dataType,
                    shape: { dimensions, toString: () => `[${dimensions.join(',')}]` },
                    toString: () => `${dataType}[${dimensions.join(',')}]`
                };
            },
            clearShapeInference() {
                delete this._inferredType;
            },
            get initializer() {
                return initializer;
            },
            get description() {
                return '';
            },
            get quantization() {
                return null;
            }
        };
    }

    _graphEditInitializerValue(name, dataType, dimensions, values) {
        const type = this._graphEditTensorType(dataType, dimensions);
        const initializer = {
            name,
            category: 'Initializer',
            type,
            encoding: '|',
            values: values.slice()
        };
        return this._graphEditTensorValue(name, type, initializer);
    }

    _renderGraphEditAddForm(item) {
        const document = this._host.document;
        const form = this._element('graph-edit-add-form');
        const title = this._element('graph-edit-add-title');
        const description = this._element('graph-edit-add-description');
        if (!form || !title || !description) {
            return;
        }
        form.replaceChildren();
        title.textContent = item.name;
        description.textContent = item.description ||
            `Add an ONNX ${item.name} operator and connect its required inputs.`;
        const fields = new Map();
        const section = (name) => {
            const element = document.createElement('div');
            element.className = 'graph-edit-add-section';
            element.textContent = name;
            form.appendChild(element);
        };
        const field = (key, label, options = {}) => {
            const row = document.createElement('label');
            row.className = 'graph-edit-add-field';
            const caption = document.createElement('span');
            caption.textContent = label;
            let control = null;
            if (options.values) {
                control = document.createElement('select');
                for (const [value, text] of options.values) {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = text;
                    control.appendChild(option);
                }
            } else {
                control = document.createElement('input');
                control.type = options.type || 'text';
                control.placeholder = options.placeholder || '';
                control.autocomplete = 'off';
                control.spellcheck = false;
                if (options.list) {
                    control.setAttribute('list', options.list);
                }
            }
            control.value = options.value === undefined ? '' : options.value;
            control.required = Boolean(options.required);
            control.setAttribute('aria-label', label);
            row.append(caption, control);
            form.appendChild(row);
            fields.set(key, control);
            return control;
        };
        const values = this._graphValues().filter((value) => value && value.name && !value.initializer);
        const tensorListId = 'graph-edit-add-tensor-options';
        const dataList = document.createElement('datalist');
        dataList.id = tensorListId;
        for (const value of values) {
            const option = document.createElement('option');
            option.value = value.name;
            dataList.appendChild(option);
        }
        form.appendChild(dataList);
        const tensorNames = new Set(values.map((value) => value.name));
        const nodeNames = new Set((this.activeTarget.nodes || []).map((node) => node.name).filter((name) => name));
        if (item.kind === 'input') {
            section('Graph interface');
            field('name', 'Input name', {
                value: this._graphEditUniqueName('input', tensorNames), required: true
            });
            field('dataType', 'Data type', {
                values: [
                    ['float32', 'float32'], ['float16', 'float16'], ['bfloat16', 'bfloat16'],
                    ['int64', 'int64'], ['int32', 'int32'], ['int16', 'int16'], ['int8', 'int8'],
                    ['uint8', 'uint8'], ['boolean', 'bool']
                ],
                value: 'float32'
            });
            field('shape', 'Shape', { value: '1', placeholder: '1, 3, 224, 224' });
        } else if (item.kind === 'output') {
            section('Graph interface');
            field('source', 'Source tensor', {
                list: tensorListId, value: values[0] ? values[0].name : '', required: true
            });
            field('name', 'Output name', {
                value: this._graphEditUniqueName('output', new Set((this.activeTarget.outputs || []).map((output) => output.name))),
                required: true
            });
        } else {
            section('Identity');
            const base = item.name.charAt(0).toLowerCase() + item.name.slice(1);
            field('nodeName', 'Node name', {
                value: this._graphEditUniqueName(base, nodeNames), required: true
            });
            if (item.split) {
                const outputBase = `${base}_output`;
                const defaultNames = [0, 1].map((index) =>
                    this._graphEditUniqueName(`${outputBase}_${index}`, tensorNames));
                const count = field('outputCount', 'Output count', {
                    type: 'number', value: 2, required: true
                });
                count.min = '1';
                const names = field('outputNames', 'Output tensors (comma-separated)', {
                    value: defaultNames.join(', '), required: true,
                    placeholder: 'split_output_0, split_output_1'
                });
                let generatedNames = names.value;
                count.addEventListener('input', () => {
                    const value = Number(count.value);
                    if (Number.isInteger(value) && value > 0 && names.value === generatedNames) {
                        const outputNames = Array.from({ length: value }, (_, index) =>
                            this._graphEditUniqueName(`${outputBase}_${index}`, tensorNames));
                        generatedNames = outputNames.join(', ');
                        names.value = generatedNames;
                    }
                });
                field('splitSizes', 'Split sizes (optional)', {
                    value: '', placeholder: 'Equal split, or enter 2, 2, 4'
                });
            } else if (item.topk) {
                field('valuesOutputName', 'Values output tensor', {
                    value: this._graphEditUniqueName('topk_values', tensorNames), required: true
                });
                field('indicesOutputName', 'Indices output tensor', {
                    value: this._graphEditUniqueName('topk_indices', tensorNames), required: true
                });
                const k = field('topK', 'K', { type: 'number', value: 1, required: true });
                k.min = '1';
            } else {
                const outputBase = item.outputs && item.outputs[0] ? item.outputs[0] : `${base}_output`;
                field('outputName', 'Output tensor', {
                    value: this._graphEditUniqueName(outputBase, tensorNames), required: true
                });
            }
            section('Inputs');
            for (let index = 0; index < (item.inputs || []).length; index++) {
                const structuralInput =
                    (item.split && index === 1) ||
                    (item.topk && index === 1) ||
                    (item.axesInput && index === 1) ||
                    (item.expand && index === 1) ||
                    (item.slice && index >= 1);
                if (structuralInput) {
                    continue;
                }
                const name = item.inputs[index];
                field(`input:${index}`, item.variadic ? `${name} (comma-separated)` : name, {
                    list: item.variadic ? null : tensorListId,
                    value: '',
                    required: false,
                    placeholder: item.variadic ? 'No inputs selected' : 'No input selected'
                });
            }
            if (item.axesInput) {
                section('Axes');
                const axesRequired = item.axesInput === 'required';
                const axesDefaultLast = item.axesInput === 'optionalDefaultLast';
                field('axes', axesRequired ? 'Axes (comma-separated)' : 'Axes (optional)', {
                    value: axesRequired || axesDefaultLast ? '-1' : '',
                    required: false,
                    placeholder: axesRequired ? '-1, 0' : 'All axes when empty, or -1, 0'
                });
            }
            if (item.expand) {
                section('Target shape');
                field('targetShape', 'Dimensions (comma-separated)', {
                    value: '', required: false, placeholder: '1, 32, 2048'
                });
            }
            if (item.slice) {
                section('Slice ranges');
                field('sliceStarts', 'Starts (comma-separated)', {
                    value: '', required: false, placeholder: '0, 0'
                });
                field('sliceEnds', 'Ends (comma-separated)', {
                    value: '', required: false, placeholder: '9223372036854775807, 128'
                });
                field('sliceAxes', 'Axes (optional)', {
                    value: '', placeholder: '0, 1'
                });
                field('sliceSteps', 'Steps (optional)', {
                    value: '', placeholder: '1, 1'
                });
            }
            if (Array.isArray(item.attributes) && item.attributes.length > 0) {
                section('Attributes');
                for (const attribute of item.attributes) {
                    if (attribute.type === 'datatype') {
                        field(`attribute:${attribute.name}`, attribute.name, {
                            values: [
                                ['1', 'float32'], ['10', 'float16'], ['16', 'bfloat16'],
                                ['7', 'int64'], ['6', 'int32'], ['5', 'int16'], ['3', 'int8'],
                                ['2', 'uint8'], ['9', 'bool']
                            ],
                            value: attribute.value
                        });
                    } else {
                        field(`attribute:${attribute.name}`, attribute.name, {
                            type: attribute.type === 'integer' ? 'number' : 'text',
                            value: attribute.value,
                            placeholder: attribute.type === 'integers' ? '0, 2, 3, 1' : ''
                        });
                    }
                }
            }
        }
        const help = document.createElement('div');
        help.className = 'graph-edit-add-help';
        let helpText = 'Graph interface changes are included in undo, redo, reset, shape inference, and the saved ONNX copy.';
        if (item.kind === 'node') {
            helpText = 'Tensor fields accept existing graph tensor names. The new node is validated when you run INFER SHAPES or SAVE AS.';
        }
        if (item.split) {
            helpText = 'For opset 17, leave Split sizes empty for an equal split. Entering sizes creates only the small int64 split-size constant required by ONNX; it does not create model weights.';
        }
        if (item.topk) {
            helpText = 'K is stored as the small int64 constant required by ONNX. TopK produces a value tensor and an int64 index tensor.';
        } else if (item.axesInput || item.expand || item.slice) {
            helpText = 'Shape parameters are stored as small int64 ONNX constants. They are structural metadata, not trainable model weights.';
        }
        help.textContent = helpText;
        form.appendChild(help);
        const actions = document.createElement('div');
        actions.className = 'graph-edit-add-actions';
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.textContent = 'CANCEL';
        cancel.addEventListener('click', () => this._closeGraphEditAddDialog());
        const add = document.createElement('button');
        add.type = 'button';
        add.className = 'primary';
        add.textContent = `ADD ${item.kind === 'node' ? item.name.toUpperCase() : item.name.toUpperCase()}`;
        add.addEventListener('click', async () => {
            try {
                for (const control of fields.values()) {
                    control.setCustomValidity('');
                }
                const result = await this._addGraphEditItem(item, fields);
                if (result) {
                    this._closeGraphEditAddDialog();
                }
            } catch (error) {
                this._updateGraphEditStatus(`ADD FAILED · ${error.message || String(error)}`);
            }
        });
        actions.append(cancel, add);
        form.appendChild(actions);
    }

    async _addGraphEditItem(item, fields) {
        const read = (key) => {
            const control = fields.get(key);
            return control ? control.value.trim() : '';
        };
        const requireName = (key, label) => {
            const value = read(key);
            if (!value) {
                const control = fields.get(key);
                control.setCustomValidity(`${label} is required.`);
                control.reportValidity();
                return null;
            }
            return value;
        };
        if (item.kind === 'input') {
            const name = requireName('name', 'Input name');
            if (!name) {
                return false;
            }
            if (this._graphValues().some((value) => value.name === name)) {
                throw new Error(`Tensor name '${name}' is already in use.`);
            }
            const dimensions = read('shape').split(',').map((value) => value.trim()).filter((value) => value)
                .map((value) => {
                    if (value === '?') {
                        return null;
                    }
                    return /^-?\d+$/.test(value) ? Number(value) : value;
                });
            await this._addGraphEditInput(name, read('dataType'), dimensions);
            return true;
        }
        if (item.kind === 'output') {
            const sourceName = requireName('source', 'Source tensor');
            const name = requireName('name', 'Output name');
            if (!sourceName || !name) {
                return false;
            }
            const source = this._graphValues().find((value) => value.name === sourceName);
            if (!source) {
                throw new Error(`Tensor '${sourceName}' was not found.`);
            }
            if ((this.activeTarget.outputs || []).some((output) => output.name === name)) {
                throw new Error(`Graph output name '${name}' is already in use.`);
            }
            await this._addGraphEditOutput(name, source);
            return true;
        }
        const nodeName = requireName('nodeName', 'Node name');
        let outputNames = [];
        if (item.split) {
            const countText = requireName('outputCount', 'Output count');
            const namesText = requireName('outputNames', 'Output tensors');
            if (!nodeName || !countText || !namesText) {
                return false;
            }
            const count = Number(countText);
            if (!Number.isInteger(count) || count < 1) {
                throw new Error('Output count must be a positive integer.');
            }
            outputNames = namesText.split(',').map((name) => name.trim()).filter((name) => name);
            if (outputNames.length !== count) {
                throw new Error(`Output count is ${count}, but ${outputNames.length} output names were provided.`);
            }
            if (new Set(outputNames).size !== outputNames.length) {
                throw new Error('Split output tensor names must be unique.');
            }
        } else if (item.topk) {
            const valuesOutputName = requireName('valuesOutputName', 'Values output tensor');
            const indicesOutputName = requireName('indicesOutputName', 'Indices output tensor');
            if (!nodeName || !valuesOutputName || !indicesOutputName) {
                return false;
            }
            if (valuesOutputName === indicesOutputName) {
                throw new Error('TopK output tensor names must be unique.');
            }
            outputNames = [valuesOutputName, indicesOutputName];
        } else {
            const outputName = requireName('outputName', 'Output tensor');
            if (!nodeName || !outputName) {
                return false;
            }
            outputNames = [outputName];
        }
        if (!nodeName) {
            return false;
        }
        if ((this.activeTarget.nodes || []).some((node) => node.name === nodeName)) {
            throw new Error(`Node name '${nodeName}' is already in use.`);
        }
        const existingTensorNames = new Set(this._graphValues().map((value) => value.name).filter((name) => name));
        for (const outputName of outputNames) {
            if (existingTensorNames.has(outputName)) {
                throw new Error(`Tensor name '${outputName}' is already in use.`);
            }
        }
        const inputs = item.variadic ? [] : new Array((item.inputs || []).length).fill(null);
        const initializers = [];
        const parseIntegerList = (key, label, required = false) => {
            const text = read(key);
            if (!text) {
                if (required) {
                    throw new Error(`${label} is required.`);
                }
                return null;
            }
            const values = text.split(',').map((entry) => Number(entry.trim()));
            if (values.some((value) => !Number.isInteger(value))) {
                throw new Error(`${label} must be a comma-separated integer list.`);
            }
            return values;
        };
        const appendInt64Initializer = (suffix, values, inputIndex = null) => {
            const name = this._graphEditUniqueName(`${nodeName}_${suffix}`, existingTensorNames);
            existingTensorNames.add(name);
            const value = this._graphEditInitializerValue(name, 'int64', [values.length], values);
            if (Number.isInteger(inputIndex)) {
                inputs[inputIndex] = value;
            } else {
                inputs.push(value);
            }
            initializers.push({
                name,
                dataType: 'int64',
                dimensions: [values.length],
                values
            });
        };
        for (let index = 0; index < (item.inputs || []).length; index++) {
            const structuralInput =
                (item.split && index === 1) ||
                (item.topk && index === 1) ||
                (item.axesInput && index === 1) ||
                (item.expand && index === 1) ||
                (item.slice && index >= 1);
            if (structuralInput) {
                continue;
            }
            const text = read(`input:${index}`);
            const names = item.variadic ? text.split(',').map((name) => name.trim()).filter((name) => name) : [text];
            for (const name of names) {
                if (!name) {
                    continue;
                }
                const value = this._graphValues().find((entry) => entry.name === name);
                if (!value) {
                    throw new Error(`Tensor '${name}' was not found.`);
                }
                if (item.variadic) {
                    inputs.push(value);
                } else {
                    inputs[index] = value;
                }
            }
        }
        if (item.split) {
            const text = read('splitSizes');
            if (text) {
                const values = text.split(',').map((entry) => Number(entry.trim()));
                if (values.length !== outputNames.length || values.some((value) => !Number.isInteger(value) || value < 0)) {
                    throw new Error(
                        `Split sizes must contain ${outputNames.length} non-negative integers, one per output.`
                    );
                }
                appendInt64Initializer('split_sizes', values, 1);
            }
        }
        if (item.topk) {
            const k = Number(requireName('topK', 'K'));
            if (!Number.isInteger(k) || k < 1) {
                throw new Error('K must be a positive integer.');
            }
            appendInt64Initializer('k', [k], 1);
        }
        if (item.axesInput) {
            const axes = parseIntegerList('axes', 'Axes');
            if (axes) {
                appendInt64Initializer('axes', axes, 1);
            }
        }
        if (item.expand) {
            const shape = parseIntegerList('targetShape', 'Target shape');
            if (shape && shape.some((dimension) => dimension < 0)) {
                throw new Error('Target shape dimensions must be non-negative integers.');
            }
            if (shape) {
                appendInt64Initializer('shape', shape, 1);
            }
        }
        if (item.slice) {
            const starts = parseIntegerList('sliceStarts', 'Slice starts');
            const ends = parseIntegerList('sliceEnds', 'Slice ends');
            const axes = parseIntegerList('sliceAxes', 'Slice axes');
            const steps = parseIntegerList('sliceSteps', 'Slice steps');
            if (Boolean(starts) !== Boolean(ends)) {
                throw new Error('Specify both Slice starts and ends, or leave both empty.');
            }
            if ((axes || steps) && !starts) {
                throw new Error('Specify Slice starts and ends before axes or steps.');
            }
            if (starts && starts.length !== ends.length) {
                throw new Error('Slice starts and ends must have the same number of entries.');
            }
            if (starts && axes && axes.length !== starts.length) {
                throw new Error('Slice axes must have the same number of entries as starts and ends.');
            }
            if (starts && steps && steps.length !== starts.length) {
                throw new Error('Slice steps must have the same number of entries as starts and ends.');
            }
            if (steps && !axes) {
                throw new Error('Specify Slice axes when specifying Slice steps.');
            }
            if (steps && steps.some((step) => step === 0)) {
                throw new Error('Slice steps cannot contain zero.');
            }
            if (starts) {
                appendInt64Initializer('slice_starts', starts, 1);
                appendInt64Initializer('slice_ends', ends, 2);
            }
            if (axes) {
                appendInt64Initializer('slice_axes', axes, 3);
            }
            if (steps) {
                appendInt64Initializer('slice_steps', steps, 4);
            }
        }
        const attributes = {};
        for (const attribute of item.attributes || []) {
            const value = read(`attribute:${attribute.name}`);
            if (attribute.type === 'integers') {
                if (value) {
                    attributes[attribute.name] = value.split(',').map((entry) => Number(entry.trim()));
                    if (attributes[attribute.name].some((entry) => !Number.isInteger(entry))) {
                        throw new Error(`Attribute '${attribute.name}' must be a comma-separated integer list.`);
                    }
                }
            } else if (attribute.type === 'float') {
                const parsed = Number(value);
                if (!Number.isFinite(parsed)) {
                    throw new Error(`Attribute '${attribute.name}' must be a number.`);
                }
                attributes[attribute.name] = parsed;
            } else {
                const parsed = Number(value);
                if (!Number.isInteger(parsed)) {
                    throw new Error(`Attribute '${attribute.name}' must be an integer.`);
                }
                attributes[attribute.name] = parsed;
            }
        }
        await this._addGraphEditNode(item, nodeName, outputNames, inputs, attributes, initializers);
        return true;
    }

    async _commitGraphEditAddition(command, message) {
        command.apply();
        // Undo/redo currently rebuild additions. Keep this true even when the
        // initial insertion can be rendered incrementally.
        command.refreshOnHistory = true;
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._clearGraphEditDerivedState();
        this._closeGraphEditAddDialog();
        const incrementalAdd = command.incrementalAdd;
        const added = incrementalAdd ? await this._addGraphEditVisualNode(incrementalAdd) : false;
        if (added) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
        } else {
            this._deferGraphEditRefresh(message);
            await this.redrawGraphEdit();
        }
        if (command.positionValue) {
            command.positionsAfter = this._graphEditPositionDelta(command.positionValue);
        }
        this._updateGraphEditStatus(added ? `${message}.` : `${message} · Graph view updated.`);
    }

    async _addGraphEditInput(name, dataType, dimensions) {
        const id = `input-${this._graphEdit.nextAddedId++}`;
        const value = this._graphEditTensorValue(name, this._graphEditTensorType(dataType, dimensions));
        const argument = {
            name,
            value: [value],
            type: null,
            visible: true,
            _graph_edit_root: true,
            _graph_edit_added_id: id,
            _graph_edit_data_type: dataType,
            _graph_edit_dimensions: dimensions
        };
        const index = this.activeTarget.inputs.length;
        const apply = () => {
            if (!this.activeTarget.inputs.includes(argument)) {
                this.activeTarget.inputs.splice(Math.min(index, this.activeTarget.inputs.length), 0, argument);
            }
        };
        const revert = () => {
            const position = this.activeTarget.inputs.indexOf(argument);
            if (position >= 0) {
                this.activeTarget.inputs.splice(position, 1);
            }
        };
        await this._commitGraphEditAddition({
            kind: 'add-input', label: `add graph input ${name}`, apply, revert, addedId: id, positionValue: argument
        }, `Added graph input ${name}`);
    }

    async _addGraphEditOutput(name, value) {
        const id = `output-${this._graphEdit.nextAddedId++}`;
        const argument = {
            name,
            value: [value],
            type: null,
            visible: true,
            _graph_edit_root: true,
            _graph_edit_added_id: id,
            _graph_edit_output_index: this.activeTarget.outputs.length
        };
        const index = this.activeTarget.outputs.length;
        const apply = () => {
            if (!this.activeTarget.outputs.includes(argument)) {
                this.activeTarget.outputs.splice(Math.min(index, this.activeTarget.outputs.length), 0, argument);
            }
        };
        const revert = () => {
            const position = this.activeTarget.outputs.indexOf(argument);
            if (position >= 0) {
                this.activeTarget.outputs.splice(position, 1);
            }
        };
        await this._commitGraphEditAddition({
            kind: 'add-output', label: `add graph output ${name}`, apply, revert, addedId: id, positionValue: argument
        }, `Added graph output ${name}`);
    }

    async _addGraphEditNode(item, name, outputNames, inputValues, attributes, initializers = []) {
        const id = `node-${this._graphEdit.nextAddedId++}`;
        const inputSlots = item.variadic && inputValues.length === 0 ? [null] : inputValues;
        // Shape-changing operators should not show the input shape as their
        // output shape. INFER SHAPES will populate it when requested.
        let outputType = null;
        const firstInput = inputSlots.find((value) => value);
        if (!item.shapeTransform && !item.typeTransform && firstInput) {
            outputType = firstInput.type;
        }
        outputNames = Array.isArray(outputNames) ? outputNames : [outputNames];
        const outputValues = outputNames.map((outputName, index) => {
            const dataType = Array.isArray(item.outputDataTypes) ? item.outputDataTypes[index] : null;
            const type = dataType ? this._graphEditTensorType(dataType, []) : outputType;
            return this._graphEditTensorValue(outputName, type);
        });
        const source = {
            name,
            op_type: item.name,
            domain: '',
            input: inputSlots.map((value) => ({ name: value ? value.name : '' })),
            output: outputNames.map((outputName) => ({ name: outputName })),
            _graph_edit_root: true,
            _graph_edit_added_id: id,
            _graph_edit_attributes: attributes,
            _graph_edit_initializers: initializers
        };
        const inputs = [];
        if (item.variadic) {
            const argument = {
                name: item.inputs[0],
                value: inputSlots.map((value) => value || this._graphEditTensorValue('')),
                option: null,
                source: { node: source, indices: inputSlots.map((value, index) => index) }
            };
            inputs.push(argument);
        } else {
            for (let index = 0; index < item.inputs.length; index++) {
                const value = inputSlots[index] || this._graphEditTensorValue('');
                inputs.push({
                    name: item.inputs[index],
                    value: [value],
                    option: Array.isArray(item.optionalInputs) && item.optionalInputs.includes(index) ? 'optional' : null,
                    source: { node: source, indices: [index] }
                });
            }
        }
        const outputs = item.variadicOutputs ? [{
            name: item.outputs[0],
            value: outputValues,
            source: { node: source, indices: outputValues.map((value, index) => index) }
        }] : outputValues.map((value, index) => ({
            name: item.outputs[index] || `output_${index}`,
            value: [value],
            source: { node: source, indices: [index] }
        }));
        const node = {
            name,
            description: '',
            metadata: [],
            type: {
                name: item.name,
                module: 'ai.onnx',
                category: item.category,
                inputs: item.inputs.map((input) => ({ name: input, list: Boolean(item.variadic) })),
                outputs: item.outputs.map((output) => ({ name: output, list: Boolean(item.variadicOutputs) }))
            },
            inputs,
            outputs,
            attributes: Object.entries(attributes).map(([attributeName, value]) => ({
                name: attributeName,
                value,
                type: Array.isArray(value) ? 'int64[]' : 'int64'
            })),
            chain: [],
            source
        };
        const index = this.activeTarget.nodes.length;
        const apply = () => {
            if (!this.activeTarget.nodes.includes(node)) {
                this.activeTarget.nodes.splice(Math.min(index, this.activeTarget.nodes.length), 0, node);
            }
        };
        const revert = () => {
            const position = this.activeTarget.nodes.indexOf(node);
            if (position >= 0) {
                this.activeTarget.nodes.splice(position, 1);
            }
        };
        await this._commitGraphEditAddition({
            kind: 'add-node', label: `add ${item.name} node ${name}`, apply, revert, addedId: id,
            positionValue: node, incrementalAdd: node
        }, `Added ${item.name} node ${name}`);
    }

    async _queueGraphEditHistory(action) {
        const previous = this._graphEdit.historyPromise || Promise.resolve();
        this._graphEdit.historyPending++;
        this._updateGraphEditStatus();
        const current = previous.catch(() => {}).then(action);
        this._graphEdit.historyPromise = current;
        try {
            return await current;
        } finally {
            this._graphEdit.historyPending--;
            if (this._graphEdit.historyPromise === current) {
                this._graphEdit.historyPromise = null;
            }
            const status = this._element('graph-edit-status');
            this._updateGraphEditStatus(status ? status.textContent : null);
        }
    }

    async undoGraphEdit() {
        return await this._queueGraphEditHistory(async () => await this._undoGraphEdit());
    }

    async _undoGraphEdit() {
        if (!this._graphEdit.enabled || this._graphEdit.undo.length === 0) {
            return;
        }
        this._cancelGraphEditSelection(false);
        const command = this._graphEdit.undo.pop();
        command.revert();
        this._graphEdit.redo.push(command);
        this._clearGraphEditDerivedState();
        let incremental = false;
        if (command.visualRemoval) {
            incremental = this._restoreGraphEditVisualNode(command.visualRemoval);
        } else if (command.incrementalAdd) {
            incremental = Boolean(this._removeGraphEditVisualNode(command.incrementalAdd));
        }
        if (incremental) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            this._updateGraphEditStatus(`Undid: ${command.label}.`);
            return;
        }
        if (command.refreshOnHistory === false) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            this._updateGraphEditStatus(`Undid: ${command.label}.`);
            return;
        }
        this._deferGraphEditRefresh(`Undid: ${command.label}`);
        if (command.refreshOnHistory) {
            await this.redrawGraphEdit(command.positionsBefore);
            this._updateGraphEditStatus(`Undid: ${command.label} · Graph view updated.`);
        }
    }

    async redoGraphEdit() {
        return await this._queueGraphEditHistory(async () => await this._redoGraphEdit());
    }

    async _redoGraphEdit() {
        if (!this._graphEdit.enabled || this._graphEdit.redo.length === 0) {
            return;
        }
        this._cancelGraphEditSelection(false);
        const command = this._graphEdit.redo.pop();
        command.apply();
        this._graphEdit.undo.push(command);
        this._clearGraphEditDerivedState();
        let incremental = false;
        if (command.visualRemoval) {
            const value = command.visualRemoval.nodeEntry.label.value;
            command.visualRemoval = this._removeGraphEditVisualNode(value);
            incremental = Boolean(command.visualRemoval);
        } else if (command.incrementalAdd) {
            incremental = await this._addGraphEditVisualNode(command.incrementalAdd);
        }
        if (incremental) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            this._updateGraphEditStatus(`Redid: ${command.label}.`);
            return;
        }
        if (command.refreshOnHistory === false) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            this._updateGraphEditStatus(`Redid: ${command.label}.`);
            return;
        }
        this._deferGraphEditRefresh(`Redid: ${command.label}`);
        if (command.refreshOnHistory) {
            await this.redrawGraphEdit(command.positionsAfter);
            this._updateGraphEditStatus(`Redid: ${command.label} · Graph view updated.`);
        }
    }

    async resetGraphEdit() {
        if (!this._graphEdit.enabled ||
            (this._graphEdit.undo.length === 0 && this._graphEdit.positions.size === 0)) {
            return;
        }
        await this._queueGraphEditRender('Resetting the graph', async () => {
            this._cancelGraphEditSelection(false);
            for (let index = this._graphEdit.undo.length - 1; index >= 0; index--) {
                this._graphEdit.undo[index].revert();
            }
            this._graphEdit.undo = [];
            this._graphEdit.redo = [];
            this._graphEdit.viewDirty = false;
            this._graphEdit.viewRevision++;
            this._graphEdit.positions.clear();
            this._clearGraphShapeInference();
            this._clearGraphEditDerivedState();
            await this.refresh(null, { animate: false });
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            return 'RESET · Restored the graph to its original edit-session state.';
        });
    }

    async graphEditConnection(edge) {
        if (!this._graphEdit.enabled) {
            return false;
        }
        const target = edge ? edge.graphEditTarget : null;
        if (!target) {
            this._updateGraphEditStatus('This connection is not an editable main-graph node input.');
            return true;
        }
        this._cancelGraphEditSelection(false);
        this._focusGraphEditCanvas();
        this._graphEdit.connection = { edge, target, replacing: false };
        if (edge.element) {
            edge.element.classList.add('graph-edit-edge-selected');
        }
        this._host.document.documentElement.classList.add('onnx-graph-edit-connection');
        this._updateGraphEditConnectionActions();
        const name = this._graphEditTargetName(target);
        this._updateGraphEditStatus(
            `CONNECTION SELECTED · ${target.oldValue.name} → ${name} · D to disconnect`
        );
        return true;
    }

    beginGraphEditConnectionReplace() {
        const connection = this._graphEdit.connection;
        if (!this._graphEdit.enabled || !connection) {
            return;
        }
        connection.replacing = true;
        this._host.document.documentElement.classList.add('onnx-graph-edit-connection-replace');
        this._updateGraphEditStatus(`REPLACE · Choose an orange output for ${this._graphEditTargetName(connection.target)}, or press Q.`);
    }

    async disconnectGraphEditConnection() {
        const connection = this._graphEdit.connection;
        if (!this._graphEdit.enabled || !connection) {
            return;
        }
        await this._disconnectGraphEditTarget(connection.target, connection.edge);
    }

    async _disconnectGraphEditTarget(target, edge = null) {
        if (target.graphOutput === true) {
            await this._disconnectGraphEditGraphOutput(target, edge);
            return;
        }
        const source = target.argument.source;
        const sourceIndex = source && Array.isArray(source.indices) ? source.indices[target.valueIndex] : -1;
        if (!source || !source.node || !Array.isArray(source.node.input) || sourceIndex < 0) {
            this._updateGraphEditStatus('This connection does not map to an editable ONNX input slot.');
            return;
        }
        const oldValues = target.argument.value.slice();
        const oldIndices = source.indices.slice();
        const oldSource = source.node.input[sourceIndex];
        const oldValue = target.oldValue;
        const emptyValue = { name: '', type: oldValue ? oldValue.type : null, initializer: null };
        const apply = () => {
            source.node.input[sourceIndex] = emptyValue;
            target.argument.value[target.valueIndex] = emptyValue;
            this._setGraphEditTargetEdgesVisible(target, false, edge);
            this._updateGraphEditPortStates();
        };
        const revert = () => {
            source.node.input[sourceIndex] = oldSource;
            target.argument.value.splice(0, target.argument.value.length, ...oldValues);
            source.indices.splice(0, source.indices.length, ...oldIndices);
            this._setGraphEditTargetEdgesVisible(target, true, edge);
            this._updateGraphEditPortStates();
        };
        const label = `disconnect ${target.oldValue.name} → ${this._graphEditTargetName(target)}`;
        const command = {
            kind: 'disconnect-input',
            label,
            apply,
            revert,
            nodeIndex: target.node.source._graph_edit_index,
            nodeAddedId: target.node.source._graph_edit_added_id,
            inputIndex: sourceIndex
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._cancelGraphEditSelection(false);
        this._clearGraphEditDerivedState();
        const required = target.argument.option !== 'optional';
        this._deferGraphEditRefresh(required ?
            `Disconnected ${label.slice('disconnect '.length)} · Required input is now unresolved and must be reconnected before save.` :
            `Disconnected ${label.slice('disconnect '.length)}`);
    }

    async _disconnectGraphEditGraphOutput(target, edge = null) {
        const argument = target.argument;
        const valueIndex = target.valueIndex;
        if (!argument || argument._graph_edit_root !== true ||
            !Number.isInteger(argument._graph_edit_output_index) ||
            !Array.isArray(argument.value) || !Number.isInteger(valueIndex)) {
            this._updateGraphEditStatus('This graph output does not map to an editable main-graph output.');
            return;
        }
        const oldValue = argument.value[valueIndex];
        if (!oldValue || !oldValue.name) {
            return;
        }
        const emptyValue = { name: '', type: oldValue.type || null, initializer: null };
        const apply = () => {
            argument.value[valueIndex] = emptyValue;
            this._setGraphEditTargetEdgesVisible(target, false, edge);
            this._updateGraphEditPortStates();
        };
        const revert = () => {
            argument.value[valueIndex] = oldValue;
            this._setGraphEditTargetEdgesVisible(target, true, edge);
            this._updateGraphEditPortStates();
        };
        const command = {
            kind: 'disconnect-graph-output',
            label: `disconnect ${oldValue.name} → graph output ${argument.name}`,
            apply,
            revert,
            outputIndex: argument._graph_edit_output_index,
            outputAddedId: argument._graph_edit_added_id,
            name: argument.name
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._cancelGraphEditSelection(false);
        this._clearGraphEditDerivedState();
        this._deferGraphEditRefresh(
            `Disconnected ${oldValue.name} → graph output ${argument.name} · Output must be reconnected before save.`
        );
    }

    async graphEditOptionalInput(node, argument, valueIndex, value) {
        const target = this._graphEditInputTarget(node, argument, valueIndex, null);
        if (!target || !value) {
            return;
        }
        this._graphEdit.pending = { value, port: null };
        await this.graphEditInput(node, argument, valueIndex, null);
    }

    beginGraphEditOptionalInput(node, argument, valueIndex) {
        const target = this._graphEditInputTarget(node, argument, valueIndex, null);
        if (!target) {
            return;
        }
        this._closeGraphEditNodeMenu();
        this._graphEdit.connection = { edge: null, target, replacing: true };
        this._host.document.documentElement.classList.add('onnx-graph-edit-connection-replace');
        this._updateGraphEditStatus(`CONNECT ${this._graphEditTargetName(target)} · Choose an orange output, or press Q.`);
    }

    _showGraphEditPortChoices(titleText, subtitleText, entries, port, event, select) {
        const menu = this._element('graph-edit-node-menu');
        if (!menu) {
            return true;
        }
        this._closeGraphEditNodeMenu();
        this._graphEdit.nodeMenu = entries;
        menu.classList.add('graph-edit-port-choice-menu');
        menu.replaceChildren();
        const document = this._host.document;
        const title = document.createElement('div');
        title.className = 'graph-edit-node-menu-title';
        title.textContent = titleText;
        const subtitle = document.createElement('div');
        subtitle.className = 'graph-edit-node-menu-subtitle';
        subtitle.textContent = subtitleText;
        const search = document.createElement('input');
        search.className = 'graph-edit-port-choice-search';
        search.type = 'search';
        search.placeholder = 'Search tensor or port…';
        search.autocomplete = 'off';
        search.spellcheck = false;
        search.setAttribute('aria-label', 'Search bundled ports');
        const candidates = document.createElement('div');
        candidates.className = 'graph-edit-node-candidates graph-edit-port-choices visible';
        const render = () => {
            candidates.replaceChildren();
            const query = search.value.trim().toLowerCase();
            const filtered = entries.filter((entry) =>
                `${entry.label} ${entry.value && entry.value.name ? entry.value.name : ''}`.toLowerCase().includes(query));
            for (const entry of filtered) {
                const candidate = document.createElement('button');
                candidate.className = 'graph-edit-node-candidate graph-edit-port-choice';
                candidate.textContent = `${entry.label} · ${entry.value.name}`;
                candidate.setAttribute('aria-label', `Choose ${entry.label} tensor ${entry.value.name}`);
                candidate.addEventListener('click', async () => {
                    this._closeGraphEditNodeMenu();
                    await select(entry);
                });
                candidates.appendChild(candidate);
            }
            if (filtered.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'graph-edit-node-candidate-empty';
                empty.textContent = 'No matching ports';
                candidates.appendChild(empty);
            }
        };
        search.addEventListener('input', render);
        render();
        const footer = document.createElement('div');
        footer.className = 'graph-edit-node-menu-footer';
        const close = document.createElement('button');
        close.textContent = 'CLOSE';
        close.addEventListener('click', () => this._closeGraphEditNodeMenu());
        footer.appendChild(close);
        menu.append(title, subtitle, search, candidates, footer);
        menu.classList.add('visible');
        const portBounds = port ? port.getBoundingClientRect() : null;
        let x = portBounds ? portBounds.right : 8;
        let y = portBounds ? portBounds.bottom : 8;
        if (event && Number.isFinite(event.clientX) && event.clientX > 0) {
            x = event.clientX;
        }
        if (event && Number.isFinite(event.clientY) && event.clientY > 0) {
            y = event.clientY;
        }
        const bounds = menu.getBoundingClientRect();
        menu.style.left = `${Math.max(8, Math.min(x, this._host.window.innerWidth - bounds.width - 8))}px`;
        menu.style.top = `${Math.max(8, Math.min(y, this._host.window.innerHeight - bounds.height - 8))}px`;
        search.focus({ preventScroll: true });
        return true;
    }

    graphEditChooseOutput(entries, port, event = null) {
        if (!Array.isArray(entries) || entries.length === 0) {
            return false;
        }
        if (entries.length === 1) {
            return this.graphEditOutput(entries[0].value, port, event);
        }
        return this._showGraphEditPortChoices(
            `Bundled output ×${entries.length}`,
            'Choose the exact ONNX output tensor',
            entries,
            port,
            event,
            async (entry) => await this.graphEditOutput(entry.value, port)
        );
    }

    graphEditChooseInput(node, entries, port, event = null) {
        if (!Array.isArray(entries) || entries.length === 0) {
            return false;
        }
        if (entries.length === 1) {
            const entry = entries[0];
            return this.graphEditInput(node, entry.argument, entry.valueIndex, port);
        }
        return this._showGraphEditPortChoices(
            `Bundled input ×${entries.length}`,
            'Choose the exact ONNX input slot',
            entries,
            port,
            event,
            async (entry) => await this.graphEditInput(node, entry.argument, entry.valueIndex, port)
        );
    }

    async graphEditOutput(value, port) {
        if (!this._graphEdit.enabled) {
            return false;
        }
        if (!value || value.initializer || !value.name) {
            this._updateGraphEditStatus('This output cannot be used as a connection source.');
            return true;
        }
        const connection = this._graphEdit.connection;
        if (connection && connection.replacing) {
            this._graphEdit.pending = { value, port };
            if (connection.target.graphOutput === true) {
                return await this.graphEditGraphOutput(
                    connection.target.argument,
                    connection.target.valueIndex
                );
            }
            return await this.graphEditInput(connection.target.node,
                connection.target.argument, connection.target.valueIndex, null);
        }
        this._cancelGraphEditSelection(false);
        this._focusGraphEditCanvas();
        this._graphEdit.pending = { value, port };
        if (port) {
            port.classList.add('graph-edit-output-port-selected');
        }
        this._host.document.documentElement.classList.add('onnx-graph-edit-source');
        this._updateGraphEditInputPorts();
        this._updateGraphEditStatus(`OUTPUT SELECTED · ${value.name} · Step 2: Choose a blue input port, or press Q.`);
        return true;
    }

    beginGraphEditOutputDrag(value, port, event, entries = null) {
        if (!this._graphEdit.enabled || !value || value.initializer || !value.name ||
            !port || !event || event.button !== 0 || this._graphEdit.linkDrag) {
            return false;
        }
        const drag = {
            value,
            port,
            pointerId: event.pointerId,
            startClientX: event.clientX,
            startClientY: event.clientY,
            clientX: event.clientX,
            clientY: event.clientY,
            active: false,
            frame: null,
            path: null,
            target: null,
            entries: Array.isArray(entries) ? entries : null,
            move: null,
            up: null
        };
        const schedule = () => {
            if (!drag.active || drag.frame || this._graphEdit.linkDrag !== drag) {
                return;
            }
            drag.frame = this._host.window.requestAnimationFrame(() => {
                drag.frame = null;
                const scrolling = this._updateGraphEditLinkDrag(drag);
                if (scrolling) {
                    schedule();
                }
            });
        };
        const move = (moveEvent) => {
            if (moveEvent.pointerId !== drag.pointerId) {
                return;
            }
            if ((moveEvent.buttons & 1) === 0) {
                this._disposeGraphEditLinkDrag();
                this._cancelGraphEditSelection(false);
                this._updateGraphEditStatus('DRAG CANCELLED · Pointer button was released.');
                return;
            }
            drag.clientX = moveEvent.clientX;
            drag.clientY = moveEvent.clientY;
            const dx = drag.clientX - drag.startClientX;
            const dy = drag.clientY - drag.startClientY;
            if (!drag.active && dx * dx + dy * dy > 25) {
                this._activateGraphEditLinkDrag(drag);
            }
            schedule();
            if (drag.active) {
                moveEvent.preventDefault();
                moveEvent.stopPropagation();
            }
        };
        const up = async (upEvent) => {
            if (upEvent.pointerId !== drag.pointerId) {
                return;
            }
            const active = drag.active;
            const cancelled = upEvent.type === 'pointercancel';
            const portTarget = active && !cancelled ?
                this._graphEditPortAt(upEvent.clientX, upEvent.clientY) : null;
            const target = active && portTarget ? this._graphEditDropTarget(portTarget, drag.value) : null;
            if (!active) {
                this._disposeGraphEditLinkDrag();
                this._suppressGraphEditNativeClick(upEvent);
                const result = drag.entries && drag.entries.length > 1 ?
                    this.graphEditChooseOutput(drag.entries, drag.port, upEvent) :
                    this.graphEditOutput(drag.value, drag.port);
                if (result && typeof result.catch === 'function') {
                    result.catch((error) => this.error(error, 'ONNX GraphSurgeon failed.', null));
                }
                return;
            }
            this._disposeGraphEditLinkDrag();
            upEvent.preventDefault();
            upEvent.stopPropagation();
            this._suppressGraphEditNativeClick(upEvent);
            if (cancelled) {
                this._cancelGraphEditSelection(false);
                this._updateGraphEditStatus('DRAG CANCELLED · Select or drag an orange output to try again.');
                return;
            }
            if (!target) {
                this._cancelGraphEditSelection(false);
                this._updateGraphEditStatus('DRAG CANCELLED · Drop on a compatible blue or purple input · Q cancels.');
                return;
            }
            if (target.error) {
                this._cancelGraphEditSelection(false);
                this._updateGraphEditStatus(target.error);
                return;
            }
            if (target.graphOutput) {
                await this.graphEditGraphOutput(target.argument, target.valueIndex);
            } else {
                await this.graphEditInput(target.node, target.argument, target.valueIndex, target.port);
            }
        };
        drag.move = move;
        drag.up = up;
        drag.schedule = schedule;
        this._graphEdit.linkDrag = drag;
        port.classList.add('graph-edit-output-port-drag-armed');
        this._host.document.addEventListener('pointermove', move, true);
        this._host.document.addEventListener('pointerup', up, true);
        this._host.document.addEventListener('pointercancel', up, true);
        return true;
    }

    _suppressGraphEditNativeClick(pointerEvent) {
        const target = pointerEvent ? pointerEvent.target : null;
        const clientX = pointerEvent && Number.isFinite(pointerEvent.clientX) ? pointerEvent.clientX : null;
        const clientY = pointerEvent && Number.isFinite(pointerEvent.clientY) ? pointerEvent.clientY : null;
        const timeStamp = pointerEvent && Number.isFinite(pointerEvent.timeStamp) ? pointerEvent.timeStamp : null;
        const suppress = (event) => {
            const sameTarget = !target || event.target === target ||
                (typeof target.contains === 'function' && target.contains(event.target));
            const samePoint = clientX === null || clientY === null ||
                Math.hypot(event.clientX - clientX, event.clientY - clientY) <= 4;
            const elapsed = timeStamp === null ? 0 : event.timeStamp - timeStamp;
            const immediate = elapsed >= 0 && elapsed <= 100;
            if (sameTarget && samePoint && immediate) {
                event.preventDefault();
                event.stopPropagation();
                this._host.document.removeEventListener('click', suppress, true);
            }
        };
        this._host.document.addEventListener('click', suppress, true);
        this._host.window.setTimeout(() => {
            this._host.document.removeEventListener('click', suppress, true);
        }, 100);
    }

    _focusGraphEditCanvas() {
        const target = this._element('target');
        if (target && typeof target.focus === 'function') {
            target.focus({ preventScroll: true });
        }
    }

    _activateGraphEditLinkDrag(drag) {
        if (!drag || drag.active || this._graphEdit.linkDrag !== drag) {
            return;
        }
        this._cancelGraphEditSelection(false, true);
        drag.active = true;
        this._graphEdit.pending = { value: drag.value, port: drag.port };
        drag.port.classList.add('graph-edit-output-port-selected');
        this._host.document.documentElement.classList.add('onnx-graph-edit-source');
        const origin = this._element('origin');
        if (origin) {
            drag.path = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'path');
            drag.path.setAttribute('class', 'graph-edit-connection-preview');
            origin.appendChild(drag.path);
        }
        this._updateGraphEditInputPorts();
        this._updateGraphEditStatus(
            `DRAG CONNECTION · ${drag.value.name} · Drop on a compatible blue or purple input · Q to cancel.`
        );
        drag.schedule();
    }

    _graphEditClientPoint(clientX, clientY) {
        const origin = this._element('origin');
        const svg = origin ? origin.ownerSVGElement : null;
        const matrix = origin && typeof origin.getScreenCTM === 'function' ? origin.getScreenCTM() : null;
        if (!svg || !matrix) {
            return null;
        }
        const point = svg.createSVGPoint();
        point.x = clientX;
        point.y = clientY;
        return point.matrixTransform(matrix.inverse());
    }

    _graphEditPortAt(clientX, clientY) {
        const element = this._host.document.elementFromPoint(clientX, clientY);
        return element && typeof element.closest === 'function' ?
            element.closest('.graph-edit-input-port') : null;
    }

    _graphEditDropTarget(port, value) {
        if (!port) {
            return null;
        }
        if (port.graphEditGraphOutput === true) {
            const argument = port.graphEditArgument;
            const valueIndex = port.graphEditValueIndex;
            const oldValue = argument && Array.isArray(argument.value) ? argument.value[valueIndex] : null;
            return {
                graphOutput: true,
                argument,
                valueIndex,
                port,
                error: this._validateGraphEditType(oldValue, value)
            };
        }
        const target = this._graphEditInputTarget(
            port.graphEditNode, port.graphEditArgument, port.graphEditValueIndex, port, false);
        if (!target) {
            return { port, error: 'This input cannot be edited.' };
        }
        return { ...target, error: this._validateGraphEdit(target, value) };
    }

    _updateGraphEditLinkDrag(drag) {
        if (!drag || !drag.active || this._graphEdit.linkDrag !== drag) {
            return false;
        }
        const container = this._element('target');
        let scrolling = false;
        if (container) {
            const bounds = container.getBoundingClientRect();
            const zone = Math.max(32, Math.min(64, Math.min(bounds.width, bounds.height) / 5));
            const maximum = 18;
            const velocity = (position, start, end) => {
                if (position < start + zone) {
                    const ratio = Math.min(1, Math.max(0, (start + zone - position) / zone));
                    return -maximum * ratio * ratio;
                }
                if (position > end - zone) {
                    const ratio = Math.min(1, Math.max(0, (position - (end - zone)) / zone));
                    return maximum * ratio * ratio;
                }
                return 0;
            };
            const dx = velocity(drag.clientX, bounds.left, bounds.right);
            const dy = velocity(drag.clientY, bounds.top, bounds.bottom);
            if (dx !== 0 || dy !== 0) {
                const left = container.scrollLeft;
                const top = container.scrollTop;
                container.scrollLeft += dx;
                container.scrollTop += dy;
                scrolling = container.scrollLeft !== left || container.scrollTop !== top;
                if (scrolling && this._target) {
                    this._target._updateScrollThumbs();
                }
            }
        }
        const bounds = drag.port.getBoundingClientRect();
        const start = this._graphEditClientPoint(
            bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
        const end = this._graphEditClientPoint(drag.clientX, drag.clientY);
        if (drag.path && start && end) {
            if (this._options.direction === 'horizontal') {
                const x = (start.x + end.x) / 2;
                drag.path.setAttribute('d', `M${start.x},${start.y} C${x},${start.y} ${x},${end.y} ${end.x},${end.y}`);
            } else {
                const y = (start.y + end.y) / 2;
                drag.path.setAttribute('d', `M${start.x},${start.y} C${start.x},${y} ${end.x},${y} ${end.x},${end.y}`);
            }
        }
        const port = this._graphEditPortAt(drag.clientX, drag.clientY);
        if (drag.target !== port) {
            if (drag.target) {
                drag.target.classList.remove(
                    'graph-edit-input-port-drag-target', 'graph-edit-input-port-drag-invalid');
            }
            drag.target = port;
        }
        const target = port ? this._graphEditDropTarget(port, drag.value) : null;
        const invalid = Boolean(target && target.error);
        if (port) {
            port.classList.toggle('graph-edit-input-port-drag-target', !invalid);
            port.classList.toggle('graph-edit-input-port-drag-invalid', invalid);
        }
        if (drag.path) {
            drag.path.classList.toggle('valid', Boolean(target && !invalid));
            drag.path.classList.toggle('invalid', invalid);
        }
        return scrolling;
    }

    _disposeGraphEditLinkDrag() {
        const drag = this._graphEdit.linkDrag;
        if (!drag) {
            return;
        }
        this._graphEdit.linkDrag = null;
        if (drag.frame) {
            this._host.window.cancelAnimationFrame(drag.frame);
        }
        if (drag.target) {
            drag.target.classList.remove(
                'graph-edit-input-port-drag-target', 'graph-edit-input-port-drag-invalid');
        }
        if (drag.path) {
            drag.path.remove();
        }
        if (drag.port) {
            drag.port.classList.remove('graph-edit-output-port-drag-armed');
        }
        this._host.document.removeEventListener('pointermove', drag.move, true);
        this._host.document.removeEventListener('pointerup', drag.up, true);
        this._host.document.removeEventListener('pointercancel', drag.up, true);
    }

    async graphEditInput(node, argument, valueIndex, port) {
        if (!this._graphEdit.enabled || !this._graphEdit.pending) {
            return false;
        }
        const value = this._graphEdit.pending.value;
        const target = this._graphEditInputTarget(node, argument, valueIndex, port);
        if (!target) {
            return true;
        }
        const error = this._validateGraphEdit(target, value);
        if (error) {
            this._updateGraphEditStatus(error);
            return true;
        }
        const oldValue = target.oldValue;
        const source = target.argument.source;
        const sourceIndex = source && Array.isArray(source.indices) ? source.indices[target.valueIndex] : -1;
        const oldSource = source && source.node && Array.isArray(source.node.input) && sourceIndex >= 0 ? source.node.input[sourceIndex] : null;
        const edgeState = this._captureGraphEditTargetEdgeState(target);
        let incrementalValid = true;
        const apply = () => {
            target.argument.value[target.valueIndex] = value;
            if (source && source.node && Array.isArray(source.node.input) && sourceIndex >= 0) {
                source.node.input[sourceIndex] = { name: value.name };
            }
            if (incrementalValid) {
                return this._previewGraphEditTargetValue(target, value, edgeState);
            }
            this._updateGraphEditPortStates();
            return false;
        };
        const revert = () => {
            target.argument.value[target.valueIndex] = oldValue;
            if (source && source.node && Array.isArray(source.node.input) && sourceIndex >= 0) {
                source.node.input[sourceIndex] = oldSource;
            }
            if (incrementalValid) {
                this._restoreGraphEditTargetEdgeState(edgeState);
            }
            this._updateGraphEditPortStates();
        };
        const previousName = oldValue.name || '(disconnected)';
        const label = `${target.node.name || target.node.type.name}.${target.argument.name}: ${previousName} → ${value.name}`;
        const command = {
            label,
            apply,
            revert,
            nodeIndex: target.node.source._graph_edit_index,
            nodeAddedId: target.node.source._graph_edit_added_id,
            inputIndex: sourceIndex,
            value: value.name,
            invalidateIncremental: () => {
                incrementalValid = false;
            }
        };
        const previewed = apply();
        command.refreshOnHistory = !previewed;
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._cancelGraphEditSelection(false);
        this._clearGraphEditDerivedState();
        if (previewed) {
            this._updateGraphEditStatus(`Changed ${label}.`);
        } else {
            this._deferGraphEditRefresh(`Changed ${label}`);
            await this.redrawGraphEdit();
            this._updateGraphEditStatus(`Changed ${label} · Graph view updated.`);
        }
        return true;
    }

    async graphEditGraphOutput(argument, valueIndex) {
        if (!this._graphEdit.enabled || !this._graphEdit.pending) {
            return false;
        }
        if (!argument || argument._graph_edit_root !== true ||
            !Number.isInteger(argument._graph_edit_output_index) ||
            !Array.isArray(argument.value) || !Number.isInteger(valueIndex)) {
            this._updateGraphEditStatus('The first editor version supports outputs of the main ONNX graph only.');
            return true;
        }
        const value = this._graphEdit.pending.value;
        const oldValue = argument.value[valueIndex];
        const error = this._validateGraphEditType(oldValue, value);
        if (error) {
            this._updateGraphEditStatus(error);
            return true;
        }
        const oldName = argument.name;
        const target = {
            graphOutput: true,
            argument,
            valueIndex,
            oldValue,
            port: null
        };
        const edgeState = this._captureGraphEditTargetEdgeState(target);
        let incrementalValid = true;
        const apply = () => {
            argument.value[valueIndex] = value;
            if (incrementalValid) {
                return this._previewGraphEditTargetValue(target, value, edgeState);
            }
            this._updateGraphEditPortStates();
            return false;
        };
        const revert = () => {
            argument.value[valueIndex] = oldValue;
            if (incrementalValid) {
                this._restoreGraphEditTargetEdgeState(edgeState);
            }
            this._updateGraphEditPortStates();
        };
        const label = `graph output ${oldName}: ${oldValue.name} → ${value.name}`;
        const command = {
            kind: 'graph-output',
            label,
            apply,
            revert,
            outputIndex: argument._graph_edit_output_index,
            outputAddedId: argument._graph_edit_added_id,
            value: value.name,
            name: oldName,
            invalidateIncremental: () => {
                incrementalValid = false;
            }
        };
        const previewed = apply();
        command.refreshOnHistory = !previewed;
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._cancelGraphEditSelection(false);
        this._clearGraphEditDerivedState();
        if (previewed) {
            this._updateGraphEditStatus(`Changed ${label}.`);
        } else {
            this._deferGraphEditRefresh(`Changed ${label}`);
            await this.redrawGraphEdit();
            this._updateGraphEditStatus(`Changed ${label} · Graph view updated.`);
        }
        return true;
    }

    _updateGraphEditInputPorts() {
        const value = this._graphEdit.pending ? this._graphEdit.pending.value : null;
        const ports = this._host.document.querySelectorAll('.graph-edit-input-port');
        for (const port of ports) {
            const graphOutput = port.graphEditGraphOutput === true;
            const target = graphOutput ? null : this._graphEditInputTarget(
                port.graphEditNode, port.graphEditArgument, port.graphEditValueIndex, port, false);
            const oldValue = graphOutput && port.graphEditArgument && Array.isArray(port.graphEditArgument.value) ?
                port.graphEditArgument.value[port.graphEditValueIndex] : null;
            let error = 'This input cannot be edited.';
            if (graphOutput && value) {
                error = this._validateGraphEditType(oldValue, value);
            } else if (target && value) {
                error = this._validateGraphEdit(target, value);
            }
            port.classList.toggle('graph-edit-input-port-invalid', Boolean(error));
            port.setAttribute('aria-disabled', error ? 'true' : 'false');
            if (port.graphEditTitle) {
                const name = graphOutput ? `graph output ${port.graphEditArgument.name}` :
                    `${port.graphEditNode.name || port.graphEditNode.type.name}.${port.graphEditArgument.name}`;
                port.graphEditTitle.textContent = error ? `${name}: ${error}` : `Connect ${value.name} to ${name}`;
            }
        }
    }

    async saveOnnxAs() {
        if (!this.activeTarget) {
            return;
        }
        const dirty = this._graphEdit.undo.length > 0;
        const invalid = dirty ? this._invalidGraphEditConnections() : [];
        if (invalid.length > 0) {
            this._updateGraphEditStatus(
                `SAVE BLOCKED · Reconnect ${invalid.length} required graph connection${invalid.length === 1 ? '' : 's'}: ${invalid.slice(0, 3).join(', ')}`
            );
            return;
        }
        // View/Edit controls the editing UI only. The undo stack represents the
        // current session model state and must be serialized in either mode.
        const edits = dirty ? this._graphEditEdits() : null;
        this._updateGraphEditStatus('Saving ONNX…');
        try {
            await this._host.execute('save-onnx-as', { edits });
        } catch (error) {
            this._updateGraphEditStatus(`Save failed: ${error.message || String(error)}`);
        }
    }

    _graphEditEdits() {
        const addedInputs = (this.activeTarget.inputs || [])
            .filter((argument) => argument._graph_edit_added_id)
            .map((argument) => ({
                kind: 'add-input',
                id: argument._graph_edit_added_id,
                name: argument.name,
                dataType: argument._graph_edit_data_type,
                dimensions: argument._graph_edit_dimensions
            }));
        const addedNodes = (this.activeTarget.nodes || [])
            .filter((node) => node.source && node.source._graph_edit_added_id)
            .map((node) => ({
                kind: 'add-node',
                id: node.source._graph_edit_added_id,
                op: node.type.name,
                domain: node.source.domain || '',
                name: node.name,
                inputs: (node.inputs || []).flatMap((argument) =>
                    (argument.value || []).map((value) => value.name)),
                outputs: (node.outputs || []).flatMap((argument) =>
                    (argument.value || []).map((value) => value.name)),
                attributes: node.source._graph_edit_attributes || {},
                initializers: node.source._graph_edit_initializers || []
            }));
        const addedOutputs = (this.activeTarget.outputs || [])
            .filter((argument) => argument._graph_edit_added_id)
            .map((argument) => ({
                kind: 'add-output',
                id: argument._graph_edit_added_id,
                name: argument.name,
                value: argument.value && argument.value[0] ? argument.value[0].name : ''
            }));
        const edits = new Map();
        for (const command of this._graphEdit.undo) {
            if (command.kind === 'add-input' || command.kind === 'add-node' || command.kind === 'add-output' ||
                command.nodeAddedId || command.outputAddedId || command.inputAddedId) {
                continue;
            }
            if (command.kind === 'graph-output') {
                edits.set(`graph-output:${command.outputIndex}`, {
                    kind: 'graph-output',
                    outputIndex: command.outputIndex,
                    value: command.value,
                    name: command.name
                });
            } else if (command.kind === 'disconnect-graph-output') {
                edits.set(`graph-output:${command.outputIndex}`, {
                    kind: 'disconnect-graph-output',
                    outputIndex: command.outputIndex,
                    name: command.name
                });
            } else if (command.kind === 'rename-graph-output') {
                edits.set(`rename-graph-output:${command.outputIndex}`, {
                    kind: 'rename-graph-output',
                    outputIndex: command.outputIndex,
                    name: command.name
                });
            } else if (command.kind === 'disconnect-input') {
                edits.set(`node-input:${command.nodeIndex}:${command.inputIndex}`, {
                    kind: 'disconnect-input',
                    nodeIndex: command.nodeIndex,
                    inputIndex: command.inputIndex
                });
            } else if (command.kind === 'delete-node') {
                edits.set(`delete-node:${command.nodeIndex}`, {
                    kind: 'delete-node',
                    nodeIndex: command.nodeIndex
                });
            } else if (command.kind === 'delete-input') {
                edits.set(`delete-input:${command.inputIndex}`, {
                    kind: 'delete-input', inputIndex: command.inputIndex
                });
            } else if (command.kind === 'delete-output') {
                edits.set(`delete-output:${command.outputIndex}`, {
                    kind: 'delete-output', outputIndex: command.outputIndex
                });
            } else if (command.kind === 'rename-node') {
                edits.set(`rename-node:${command.nodeIndex}`, {
                    kind: 'rename-node',
                    nodeIndex: command.nodeIndex,
                    name: command.name
                });
            } else if (command.kind === 'rename-output') {
                edits.set(`rename-output:${command.nodeIndex}:${command.outputIndex}`, {
                    kind: 'rename-output',
                    nodeIndex: command.nodeIndex,
                    outputIndex: command.outputIndex,
                    name: command.name
                });
            } else {
                edits.set(`node-input:${command.nodeIndex}:${command.inputIndex}`, {
                    kind: 'node-input',
                    nodeIndex: command.nodeIndex,
                    inputIndex: command.inputIndex,
                    value: command.value
                });
            }
        }
        return [...addedInputs, ...addedNodes, ...Array.from(edits.values()), ...addedOutputs];
    }

    async inferGraphShapes() {
        if (!this.activeTarget) {
            return;
        }
        const invalid = this._graphEdit.undo.length > 0 ? this._invalidGraphEditConnections() : [];
        if (invalid.length > 0) {
            this._updateGraphEditStatus(
                `SHAPE INFERENCE BLOCKED · Reconnect required graph connection${invalid.length === 1 ? '' : 's'}: ${invalid.slice(0, 3).join(', ')}`
            );
            return;
        }
        this._cancelGraphEditSelection(false);
        this._updateGraphEditStatus(this._graphEdit.enabled ?
            'Running ONNX shape inference on the current edited graph…' :
            'Running ONNX shape inference…');
        try {
            const result = await this._host.execute('infer-onnx-shapes', {
                edits: this._graphEditEdits()
            });
            if (result) {
                await this.graphEditShapeInferenceResult(result);
            }
        } catch (error) {
            this._showGraphEditInferenceError({
                message: error.message || String(error),
                summary: error.message || String(error)
            });
        }
    }

    graphEditSaveResult(result) {
        if (result && result.error) {
            this._updateGraphEditStatus(`Save failed: ${result.error}`);
        } else if (result && result.path) {
            this._updateGraphEditStatus(`Saved ONNX: ${result.path}`);
        } else {
            this._updateGraphEditStatus();
        }
    }

    async graphEditShapeInferenceResult(result) {
        if (result && result.error) {
            this._showGraphEditInferenceError(result.error);
            return;
        }
        if (!result || !Array.isArray(result.tensors)) {
            this._updateGraphEditStatus();
            return;
        }
        this._clearGraphShapeInference();
        const values = new Map(this._graphValues().map((value) => [value.name, value]));
        let applied = 0;
        for (const tensor of result.tensors) {
            const value = tensor && values.get(tensor.name);
            if (value && typeof value.applyShapeInference === 'function' &&
                typeof tensor.dataType === 'string' && Array.isArray(tensor.dimensions)) {
                value.applyShapeInference(tensor.dataType, tensor.dimensions);
                applied++;
            }
        }
        await this._queueGraphEditRender('Applying inferred shapes', async () => {
            await this.refresh(null, { animate: false });
            const warnings = Array.isArray(result.warnings) ? result.warnings
                .map((warning) => typeof warning === 'string' ? warning : warning && warning.message)
                .filter((warning) => warning) : [];
            const outcome = warnings.length > 0 ? 'SHAPE INFERENCE COMPLETED WITH WARNING' : 'SHAPE INFERENCE PASSED';
            const warning = warnings.length > 0 ? ` · ${warnings.join(' · ')}` : '';
            return `${outcome} · ${applied} tensor${applied === 1 ? '' : 's'} updated · ${result.nodes || 0} nodes processed${warning}`;
        });
    }

    _graphEditInferenceNode(diagnostic) {
        const info = diagnostic && diagnostic.node;
        const nodes = this.activeTarget ? this.activeTarget.nodes || [] : [];
        if (!info) {
            return null;
        }
        if (info.name) {
            const match = nodes.find((node) => node.name === info.name);
            if (match) {
                return match;
            }
        }
        if (Number.isInteger(info.index)) {
            return nodes.find((node) => node.source &&
                node.source._graph_edit_index === info.index) || nodes[info.index] || null;
        }
        return null;
    }

    _focusGraphEditInferenceNode(node) {
        if (node && this._target) {
            this._target.scrollTo(this._target.select([node], 'sidebar'));
        }
    }

    _showGraphEditInferenceError(error) {
        const diagnostic = error && typeof error === 'object' ?
            error : { message: String(error || 'Unknown shape inference error.') };
        const message = diagnostic.message || diagnostic.summary || 'Unknown shape inference error.';
        const summary = diagnostic.summary || message;
        const node = this._graphEditInferenceNode(diagnostic);
        const nodeInfo = diagnostic.node || {};
        const nodeLabel = nodeInfo.name || node && node.name || '';
        const opType = nodeInfo.opType || (node && node.type ? node.type.name : '');
        const location = [nodeLabel, opType && `(${opType})`].filter((value) => value).join(' ');
        const statusSummary = summary.replace(/\s+/g, ' ').slice(0, 180);
        this._updateGraphEditStatus(
            `SHAPE INFERENCE FAILED${location ? ` · ${location}` : ''} · ${statusSummary}${statusSummary.length < summary.length ? '…' : ''}`
        );
        const status = this._element('graph-edit-status');
        if (status) {
            status.classList.add('invalid');
        }
        const menu = this._element('graph-edit-node-menu');
        if (!menu) {
            return;
        }
        this._closeGraphEditNodeMenu();
        this._graphEdit.nodeMenu = diagnostic;
        menu.replaceChildren();
        menu.classList.add('graph-edit-inference-menu');
        const document = this._host.document;
        const title = document.createElement('div');
        title.className = 'graph-edit-node-menu-title';
        title.textContent = 'Shape inference failed';
        const subtitle = document.createElement('div');
        subtitle.className = 'graph-edit-node-menu-subtitle';
        subtitle.textContent = location ?
            `Location · ${location}${Number.isInteger(nodeInfo.index) ? ` · node ${nodeInfo.index}` : ''}` :
            'ONNX could not identify a single failing node';
        menu.append(title, subtitle);
        const problemSection = document.createElement('div');
        problemSection.className = 'graph-edit-node-menu-section';
        problemSection.textContent = 'What does not match';
        const problem = document.createElement('div');
        problem.className = 'graph-edit-inference-problem';
        problem.textContent = summary;
        menu.append(problemSection, problem);
        const tensorText = (tensor) => {
            const dimensions = Array.isArray(tensor.dimensions) ?
                `[${tensor.dimensions.map((dimension) => dimension === null ? '?' : dimension).join(', ')}]` : '';
            return `${tensor.name || '(unnamed)'} · ${tensor.dataType || 'type unknown'}${dimensions}`;
        };
        const appendTensors = (label, tensors) => {
            if (!Array.isArray(tensors) || tensors.length === 0) {
                return;
            }
            const section = document.createElement('div');
            section.className = 'graph-edit-node-menu-section';
            section.textContent = label;
            menu.appendChild(section);
            for (const tensor of tensors) {
                const row = document.createElement('div');
                row.className = 'graph-edit-inference-tensor';
                row.textContent = tensorText(tensor);
                menu.appendChild(row);
            }
        };
        appendTensors('Observed inputs', diagnostic.inputs);
        appendTensors('Declared outputs', diagnostic.outputs);
        if (message && message !== summary) {
            const details = document.createElement('details');
            details.className = 'graph-edit-inference-details';
            const detailsTitle = document.createElement('summary');
            detailsTitle.textContent = 'Raw ONNX error';
            const raw = document.createElement('pre');
            raw.textContent = message;
            details.append(detailsTitle, raw);
            menu.appendChild(details);
        }
        const footer = document.createElement('div');
        footer.className = 'graph-edit-node-menu-footer';
        if (node) {
            const show = document.createElement('button');
            show.className = 'primary';
            show.textContent = 'SHOW NODE';
            show.addEventListener('click', () => this._focusGraphEditInferenceNode(node));
            footer.appendChild(show);
        }
        const close = document.createElement('button');
        close.textContent = 'CLOSE';
        close.addEventListener('click', () => this._closeGraphEditNodeMenu());
        footer.appendChild(close);
        menu.appendChild(footer);
        menu.classList.add('visible');
        const button = this._element('graph-edit-infer-button');
        const anchor = button ? button.getBoundingClientRect() : { left: 12, top: 12, bottom: 12 };
        const bounds = menu.getBoundingClientRect();
        const left = Math.max(8, Math.min(anchor.left, this._host.window.innerWidth - bounds.width - 8));
        let top = anchor.top - bounds.height - 8;
        if (top < 8) {
            top = Math.min(anchor.bottom + 8, this._host.window.innerHeight - bounds.height - 8);
        }
        menu.style.left = `${left}px`;
        menu.style.top = `${Math.max(8, top)}px`;
        if (node) {
            this._focusGraphEditInferenceNode(node);
        }
    }

    _graphEditInputTarget(node, argument, valueIndex, port, update = true) {
        if (!node || !argument || !Array.isArray(argument.value) || !Number.isInteger(valueIndex)) {
            if (update) {
                this._updateGraphEditStatus('Select a blue input port on an operator node.');
            }
            return null;
        }
        if (!node.source || node.source._graph_edit_root !== true ||
            (!Number.isInteger(node.source._graph_edit_index) && !node.source._graph_edit_added_id)) {
            if (update) {
                this._updateGraphEditStatus('The first editor version supports the main ONNX graph only.');
            }
            return null;
        }
        const oldValue = argument.value[valueIndex];
        if (!oldValue || oldValue.initializer) {
            if (update) {
                this._updateGraphEditStatus('Initializer inputs cannot be rewired in this editor version.');
            }
            return null;
        }
        return { node, argument, valueIndex, oldValue, port };
    }

    _validateGraphEdit(target, value) {
        const typeError = this._validateGraphEditType(target.oldValue, value);
        if (typeError) {
            return typeError;
        }
        const nodes = this.activeTarget && Array.isArray(this.activeTarget.nodes) ? this.activeTarget.nodes : [];
        const producers = new Map();
        const consumersByValue = new Map();
        for (const node of nodes) {
            for (const output of node.outputs || []) {
                for (const outputValue of output.value || []) {
                    producers.set(outputValue, node);
                }
            }
            for (const input of node.inputs || []) {
                for (const inputValue of input.value || []) {
                    const inputConsumers = consumersByValue.get(inputValue) || [];
                    inputConsumers.push(node);
                    consumersByValue.set(inputValue, inputConsumers);
                }
            }
        }
        const producer = producers.get(value) || null;
        if (!producer) {
            return null;
        }
        if (producer === target.node) {
            return 'A node cannot consume its own output.';
        }
        const consumers = new Map();
        for (const node of nodes) {
            consumers.set(node, []);
        }
        for (const [outputValue, outputProducer] of producers) {
            for (const consumer of consumersByValue.get(outputValue) || []) {
                consumers.get(outputProducer).push(consumer);
            }
        }
        const queue = [target.node];
        const visited = new Set();
        while (queue.length > 0) {
            const node = queue.shift();
            if (node === producer) {
                return 'This connection would create a cycle.';
            }
            if (!visited.has(node)) {
                visited.add(node);
                queue.push(...(consumers.get(node) || []));
            }
        }
        return null;
    }

    _validateGraphEditType(oldValue, newValue) {
        const oldType = oldValue ? oldValue.type : null;
        const newType = newValue ? newValue.type : null;
        if (oldType && newType && oldType.dataType && newType.dataType && oldType.dataType !== newType.dataType) {
            return `Type mismatch: ${oldType.dataType} input cannot use ${newType.dataType}.`;
        }
        return null;
    }

    async relayoutGraph() {
        if (!this.activeTarget) {
            return;
        }
        await this._queueGraphEditRender('Re-laying out the graph', async () => {
            this._cancelGraphEditSelection(false);
            this._graphEdit.positions.clear();
            this._graphEdit.viewDirty = false;
            this._graphEdit.viewRevision++;
            await this.refresh(null, { animate: false });
            return 'Graph rebuilt and re-laid out. No model reload was needed.';
        });
    }

    _deferGraphEditRefresh(message) {
        this._graphEdit.viewDirty = true;
        this._graphEdit.viewRevision++;
        this._host.document.documentElement.classList.add('onnx-graph-edit-view-dirty');
        this._updateGraphEditStatus(`${message} · REFRESH VIEW to redraw the latest graph state.`);
    }

    async redrawGraphEdit(positionOverrides = null) {
        await this._queueGraphEditRender('Refreshing the graph view', async () => {
            if (!this._graphEdit.enabled || !this._graphEdit.viewDirty) {
                return null;
            }
            this._cancelGraphEditSelection(false);
            // Structural edits still need a new SVG graph, but should never
            // silently rearrange the user's working layout. RE-LAYOUT is the
            // only action that intentionally clears these positions.
            this._preserveVisibleGraphEditPositions();
            if (positionOverrides) {
                for (const [value, position] of positionOverrides) {
                    this._graphEdit.positions.set(value, position);
                }
            }
            const revision = this._graphEdit.viewRevision;
            this._graphEdit.redrawing = true;
            try {
                await this.refresh(null, { animate: false });
            } finally {
                this._graphEdit.redrawing = false;
            }
            if (revision === this._graphEdit.viewRevision) {
                this._graphEdit.viewDirty = false;
                this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
                return 'Graph view updated. Manual node positions were preserved.';
            }
            return 'Graph view refreshed, but newer edits are still pending · Press REFRESH VIEW again.';
        });
    }

    async _queueGraphEditRender(label, task) {
        const previous = this._graphEdit.renderPromise || Promise.resolve();
        const queued = Boolean(this._graphEdit.renderPromise);
        let message = null;
        const current = previous.catch(() => null).then(async (previousMessage) => {
            const nextMessage = await task();
            return nextMessage === null || nextMessage === undefined ? previousMessage : nextMessage;
        });
        this._graphEdit.renderPromise = current;
        this._updateGraphEditStatus(queued ? `${label} after the current graph update…` : `${label}…`);
        try {
            message = await current;
        } finally {
            if (this._graphEdit.renderPromise === current) {
                this._graphEdit.renderPromise = null;
                this._updateGraphEditStatus(message);
            }
        }
    }

    _rerouteGraphEditEdge(edge) {
        if (!edge || !edge.from || !edge.to) {
            return;
        }
        const from = { x: edge.from.x, y: edge.from.y };
        const to = { x: edge.to.x, y: edge.to.y };
        const gap = 18;
        const values = (node, output) => {
            const ports = [];
            const seen = new Set();
            for (const argument of output ? node.outputs || [] : node.inputs || []) {
                for (const value of argument && Array.isArray(argument.value) ? argument.value : []) {
                    if (value && !value.initializer && (!output || value.name && !seen.has(value))) {
                        seen.add(value);
                        ports.push({ argument, value });
                    }
                }
            }
            return ports;
        };
        const offset = (node, output) => {
            const ports = values(node, output);
            if (ports.length === 0) {
                return 0.5;
            }
            const target = edge.graphEditTarget;
            const value = edge.value && edge.value.value;
            const index = ports.findIndex((port) => output ?
                port.value === value || value && port.value.name === value.name :
                target && port.argument === target.argument &&
                    port.argument.value[target.valueIndex] === port.value);
            return ((index < 0 ? 0 : index) + 1) / (ports.length + 1);
        };
        const anchor = (node, side, portOffset) => {
            if (side === 'top' || side === 'bottom') {
                return {
                    x: node.x - node.width / 2 + node.width * portOffset,
                    y: node.y + (side === 'top' ? -node.height / 2 : node.height / 2)
                };
            }
            return {
                x: node.x + (side === 'left' ? -node.width / 2 : node.width / 2),
                y: node.y - node.height / 2 + node.height * portOffset
            };
        };
        let route = null;
        if (this._options.direction === 'horizontal') {
            const source = anchor(edge.from, 'right', offset(edge.from, true));
            const target = anchor(edge.to, 'left', offset(edge.to, false));
            const distance = target.x - source.x;
            const stub = distance >= 0 ? Math.min(gap, distance / 3) : gap;
            const exit = { x: source.x + stub, y: source.y };
            const entry = { x: target.x - stub, y: target.y };
            if (distance >= 0) {
                const x = (exit.x + entry.x) / 2;
                route = [source, exit, { x, y: exit.y }, { x, y: entry.y }, entry, target];
            } else {
                const top = Math.min(
                    edge.from.y - edge.from.height / 2,
                    edge.to.y - edge.to.height / 2,
                    exit.y,
                    entry.y
                ) - gap;
                const bottom = Math.max(
                    edge.from.y + edge.from.height / 2,
                    edge.to.y + edge.to.height / 2,
                    exit.y,
                    entry.y
                ) + gap;
                const topCost = Math.abs(exit.y - top) + Math.abs(entry.y - top);
                const bottomCost = Math.abs(exit.y - bottom) + Math.abs(entry.y - bottom);
                const y = topCost <= bottomCost ? top : bottom;
                route = [source, exit, { x: exit.x, y }, { x: entry.x, y }, entry, target];
            }
        } else {
            const source = anchor(edge.from, 'bottom', offset(edge.from, true));
            const target = anchor(edge.to, 'top', offset(edge.to, false));
            const distance = target.y - source.y;
            const stub = distance >= 0 ? Math.min(gap, distance / 3) : gap;
            const exit = { x: source.x, y: source.y + stub };
            const entry = { x: target.x, y: target.y - stub };
            if (distance >= 0) {
                const y = (exit.y + entry.y) / 2;
                route = [source, exit, { x: exit.x, y }, { x: entry.x, y }, entry, target];
            } else {
                const left = Math.min(
                    edge.from.x - edge.from.width / 2,
                    edge.to.x - edge.to.width / 2,
                    exit.x,
                    entry.x
                ) - gap;
                const right = Math.max(
                    edge.from.x + edge.from.width / 2,
                    edge.to.x + edge.to.width / 2,
                    exit.x,
                    entry.x
                ) + gap;
                const leftCost = Math.abs(exit.x - left) + Math.abs(entry.x - left);
                const rightCost = Math.abs(exit.x - right) + Math.abs(entry.x - right);
                const x = leftCost <= rightCost ? left : right;
                route = [source, exit, { x, y: exit.y }, { x, y: entry.y }, entry, target];
            }
        }
        edge.graphEditRoute = route;
        edge.points = [from, ...route, to];
        edge.x = (from.x + to.x) / 2;
        edge.y = (from.y + to.y) / 2;
        if (edge.element) {
            edge.update();
        }
    }

    _graphEditEdgeSnapshot(edge) {
        if (!edge || !edge.from || !edge.to || !Array.isArray(edge.points) || edge.points.length < 2) {
            return null;
        }
        const points = edge.points.map((point) => ({ x: point.x, y: point.y }));
        const distances = [0];
        for (let index = 1; index < points.length; index++) {
            const previous = points[index - 1];
            const point = points[index];
            distances.push(distances[index - 1] + Math.hypot(point.x - previous.x, point.y - previous.y));
        }
        const total = distances[distances.length - 1];
        const weights = distances.map((distance, index) => total > 0 ? distance / total : index / (points.length - 1));
        return {
            points,
            weights,
            from: { x: edge.from.x, y: edge.from.y },
            to: { x: edge.to.x, y: edge.to.y },
            x: edge.x,
            y: edge.y
        };
    }

    _deformGraphEditEdge(edge, snapshot) {
        if (!edge || !snapshot) {
            this._rerouteGraphEditEdge(edge);
            return;
        }
        const source = { x: edge.from.x - snapshot.from.x, y: edge.from.y - snapshot.from.y };
        const target = { x: edge.to.x - snapshot.to.x, y: edge.to.y - snapshot.to.y };
        edge.points = snapshot.points.map((point, index) => {
            const weight = snapshot.weights[index];
            return {
                x: point.x + source.x * (1 - weight) + target.x * weight,
                y: point.y + source.y * (1 - weight) + target.y * weight
            };
        });
        edge.graphEditRoute = null;
        edge.x = Number.isFinite(snapshot.x) ? snapshot.x + (source.x + target.x) / 2 :
            (edge.from.x + edge.to.x) / 2;
        edge.y = Number.isFinite(snapshot.y) ? snapshot.y + (source.y + target.y) / 2 :
            (edge.from.y + edge.to.y) / 2;
        if (edge.element) {
            edge.update();
        }
    }

    _graphEditNodeEdges(node) {
        const edges = [];
        if (this._target && this._target.edges) {
            for (const entry of this._target.edges.values()) {
                const edge = entry.label;
                if (edge && (edge.from === node || edge.to === node)) {
                    edges.push(edge);
                }
            }
        }
        return edges;
    }

    _removeGraphEditVisualNode(value) {
        if (!this._target || !this._target.nodes || !this._target.edges) {
            return false;
        }
        let nodeKey = null;
        let visualNode = null;
        for (const [key, entry] of this._target.nodes) {
            if (entry && entry.label && entry.label.value === value) {
                if (!visualNode || (entry.label.element && entry.label.element.isConnected)) {
                    nodeKey = key;
                    visualNode = entry.label;
                }
                if (visualNode.element && visualNode.element.isConnected) {
                    break;
                }
            }
        }
        if (!visualNode) {
            return null;
        }
        const record = {
            nodeKey,
            nodeEntry: this._target.nodes.get(nodeKey),
            nodeElement: visualNode.element ? {
                element: visualNode.element,
                parent: visualNode.element.parentNode,
                next: visualNode.element.nextSibling
            } : null,
            edges: [],
            position: this._graphEdit.positions.get(value) || null
        };
        for (const [key, entry] of Array.from(this._target.edges)) {
            const edge = entry && entry.label;
            if (!edge || (edge.from !== visualNode && edge.to !== visualNode)) {
                continue;
            }
            const elements = [edge.element, edge.hitTest, edge.labelElement].map((element) => element ? {
                element, parent: element.parentNode, next: element.nextSibling
            } : null);
            record.edges.push({ key, entry, elements });
            if (edge.hitTest) {
                this._target._focusable.delete(edge.hitTest);
            }
            if (edge.value && Array.isArray(edge.value._edges)) {
                const index = edge.value._edges.indexOf(edge);
                if (index >= 0) {
                    edge.value._edges.splice(index, 1);
                }
            }
            for (const item of elements) {
                const element = item ? item.element : null;
                if (element && element.parentNode) {
                    element.parentNode.removeChild(element);
                }
            }
            this._target.edges.delete(key);
        }
        if (visualNode.element && visualNode.element.parentNode) {
            visualNode.element.parentNode.removeChild(visualNode.element);
        }
        this._target.nodes.delete(nodeKey);
        delete this._target._graphEditPorts;
        this._graphEdit.positions.delete(value);
        this._updateGraphEditPortStates();
        // A stale renderer entry can have an element that is no longer the
        // live SVG node. Treat that as an incremental-removal miss so the
        // caller performs the position-preserving graph redraw instead of
        // reporting success while leaving the visible node behind.
        const duplicate = Array.from(this._target.nodes.values()).some((entry) =>
            entry && entry.label && entry.label.value === value && entry.label.element &&
            entry.label.element.isConnected);
        return record.nodeElement && record.nodeElement.parent &&
            !record.nodeElement.element.isConnected && !duplicate ? record : null;
    }

    _restoreGraphEditVisualNode(record) {
        if (!record || !this._target || !record.nodeEntry ||
            !record.nodeElement || !record.nodeElement.parent ||
            !record.nodeElement.parent.isConnected) {
            return false;
        }
        this._target.nodes.set(record.nodeKey, record.nodeEntry);
        const restore = (item) => {
            if (item && item.parent && item.element) {
                const next = item.next && item.next.parentNode === item.parent ? item.next : null;
                item.parent.insertBefore(item.element, next);
            }
        };
        restore(record.nodeElement);
        for (const edge of record.edges) {
            this._target.edges.set(edge.key, edge.entry);
            const visualEdge = edge.entry.label;
            if (visualEdge && visualEdge.value) {
                visualEdge.value._edges = visualEdge.value._edges || [];
                if (!visualEdge.value._edges.includes(visualEdge)) {
                    visualEdge.value._edges.push(visualEdge);
                }
            }
            for (const item of edge.elements) {
                restore(item);
            }
            if (edge.entry.label && edge.entry.label.hitTest) {
                this._target._focusable.set(edge.entry.label.hitTest, edge.entry.label);
            }
        }
        if (record.position) {
            this._graphEdit.positions.set(record.nodeEntry.label.value, record.position);
        }
        delete this._target._graphEditPorts;
        this._updateGraphEditPortStates();
        return true;
    }

    _graphEditTargetEdges(target, preferred = null) {
        const edges = new Set();
        if (preferred) {
            edges.add(preferred);
        }
        if (!target || !this._target || !this._target.edges) {
            return Array.from(edges);
        }
        for (const entry of this._target.edges.values()) {
            const edge = entry.label;
            const candidate = edge ? edge.graphEditTarget : null;
            if (!candidate) {
                continue;
            }
            const same = target.graphOutput === true ?
                candidate.graphOutput === true &&
                    candidate.argument === target.argument &&
                    candidate.valueIndex === target.valueIndex :
                candidate.graphOutput !== true &&
                    candidate.node === target.node &&
                    candidate.argument === target.argument &&
                    candidate.valueIndex === target.valueIndex;
            if (same) {
                edges.add(edge);
            }
        }
        return Array.from(edges);
    }

    _refreshGraphEditEdgePorts(edges) {
        if (!this._target) {
            return;
        }
        delete this._target._graphEditPorts;
        const nodes = new Set();
        for (const edge of edges || []) {
            if (edge) {
                nodes.add(edge.from);
                nodes.add(edge.to);
            }
        }
        for (const node of nodes) {
            if (node && node._graphEditInputPorts) {
                node._graphEditInputPorts.update();
            }
            if (node && node._graphEditOutputPorts) {
                node._graphEditOutputPorts.update();
            }
        }
    }

    _setGraphEditTargetEdgesVisible(target, visible, preferred = null) {
        const edges = this._graphEditTargetEdges(target, preferred);
        for (const edge of edges) {
            for (const element of [edge.element, edge.hitTest, edge.labelElement]) {
                if (element) {
                    element.classList.toggle('graph-edit-edge-disconnected', !visible);
                }
            }
        }
        this._refreshGraphEditEdgePorts(edges);
    }

    _captureGraphEditTargetEdgeState(target) {
        return this._graphEditTargetEdges(target).map((edge) => ({
            edge,
            from: edge.from,
            value: edge.value,
            points: Array.isArray(edge.points) ? edge.points.map((point) => ({ ...point })) : null,
            graphEditRoute: Array.isArray(edge.graphEditRoute) ?
                edge.graphEditRoute.map((point) => ({ ...point })) : null,
            x: edge.x,
            y: edge.y,
            targetOldValue: edge.graphEditTarget ? edge.graphEditTarget.oldValue : null,
            disconnected: Boolean(edge.element &&
                edge.element.classList.contains('graph-edit-edge-disconnected'))
        }));
    }

    _restoreGraphEditTargetEdgeState(state) {
        if (state && Array.isArray(state.created)) {
            for (const edge of state.created) {
                this._removeGraphEditVisualEdge(edge);
            }
            state.created = [];
        }
        for (const item of state) {
            const edge = item.edge;
            edge.from = item.from;
            edge.value = item.value;
            edge.points = item.points ? item.points.map((point) => ({ ...point })) : item.points;
            edge.graphEditRoute = item.graphEditRoute ?
                item.graphEditRoute.map((point) => ({ ...point })) : null;
            edge.x = item.x;
            edge.y = item.y;
            if (edge.graphEditTarget) {
                edge.graphEditTarget.oldValue = item.targetOldValue;
            }
            for (const element of [edge.element, edge.hitTest, edge.labelElement]) {
                if (element) {
                    element.classList.toggle('graph-edit-edge-disconnected', item.disconnected);
                }
            }
            if (!item.disconnected && edge.points && edge.element) {
                edge.update();
            }
        }
        this._refreshGraphEditEdgePorts(state.map((item) => item.edge));
    }

    _previewGraphEditTargetValue(target, value, state) {
        if (!Array.isArray(state) || !this._target ||
            !this._target._values || !this._target._values.has(value.name)) {
            this._updateGraphEditPortStates();
            return false;
        }
        const source = this._target._values.get(value.name);
        if (!source || !source.from) {
            this._updateGraphEditPortStates();
            return false;
        }
        if (state.length === 0) {
            const edge = this._createGraphEditVisualEdge(target, source, value);
            if (!edge) {
                this._updateGraphEditPortStates();
                return false;
            }
            state.created = [edge];
            this._updateGraphEditPortStates();
            return true;
        }
        for (const item of state) {
            const edge = item.edge;
            const previousFrom = edge.from;
            edge.from = source.from;
            edge.value = source;
            if (edge.graphEditTarget) {
                edge.graphEditTarget.oldValue = value;
            }
            this._rerouteGraphEditEdge(edge);
            for (const element of [edge.element, edge.hitTest, edge.labelElement]) {
                if (element) {
                    element.classList.remove('graph-edit-edge-disconnected');
                }
            }
            this._refreshGraphEditEdgePorts([{ from: previousFrom, to: edge.to }, edge]);
        }
        this._updateGraphEditPortStates();
        return true;
    }

    _removeGraphEditVisualEdge(edge) {
        if (!edge || !this._target || !this._target.edges) {
            return;
        }
        const key = `${edge.v}:${edge.w}:${edge.name || ''}`;
        this._target.edges.delete(key);
        if (edge.hitTest && this._target._focusable) {
            this._target._focusable.delete(edge.hitTest);
        }
        if (edge.value && Array.isArray(edge.value._edges)) {
            const index = edge.value._edges.indexOf(edge);
            if (index >= 0) {
                edge.value._edges.splice(index, 1);
            }
        }
        for (const element of [edge.element, edge.hitTest, edge.labelElement]) {
            if (element && element.parentNode) {
                element.parentNode.removeChild(element);
            }
        }
        this._refreshGraphEditEdgePorts([edge]);
    }

    _createGraphEditVisualEdge(target, source, value) {
        if (!target || !source || !source.from || !this._target || !this._target._document) {
            return null;
        }
        let destination = null;
        for (const entry of this._target.nodes.values()) {
            const candidate = entry && entry.label;
            if (candidate && candidate.value === (target.graphOutput ? target.argument : target.node)) {
                destination = candidate;
                break;
            }
        }
        if (!destination) {
            return null;
        }
        const document = this._target._document;
        const edgePaths = document.getElementById('edge-paths');
        const edgeHitTests = document.getElementById('edge-paths-hit-test');
        const edgeLabels = document.getElementById('edge-labels');
        if (!edgePaths || !edgeHitTests || !edgeLabels) {
            return null;
        }
        const edge = new view.Edge(source.from, destination);
        edge.value = source;
        edge.graphEditTarget = target;
        edge.graphEditTarget.oldValue = value;
        edge.id = `edge-${value.name}-${++this._graphEdit.nextAddedId}`;
        this._target.setEdge(edge);
        source._edges = source._edges || [];
        source._edges.push(edge);
        edge.build(document, edgePaths, edgeHitTests, edgeLabels);
        if (edge.hitTest) {
            this._target._focusable.set(edge.hitTest, edge);
        }
        this._rerouteGraphEditEdge(edge);
        this._refreshGraphEditEdgePorts([edge]);
        return edge;
    }

    async _addGraphEditVisualNode(node) {
        if (!node || !this._target || !this._target._document || !this._target.nodes) {
            return false;
        }
        const document = this._target._document;
        const nodeGroup = document.getElementById('nodes');
        if (!nodeGroup) {
            return false;
        }
        const visualNode = this._target.createNode(node);
        this._target.setNode(visualNode);
        const inputValues = [];
        for (const argument of node.inputs || []) {
            for (let valueIndex = 0; valueIndex < (argument.value || []).length; valueIndex++) {
                const value = argument.value[valueIndex];
                if (value && value.name && !value.initializer) {
                    const visualValue = this._target.createValue(value);
                    visualValue.to.push(visualNode);
                    inputValues.push({
                        source: visualValue,
                        target: { node, argument, valueIndex, oldValue: value, port: null }
                    });
                }
            }
        }
        for (const argument of node.outputs || []) {
            for (const value of argument.value || []) {
                if (value && value.name) {
                    this._target.createValue(value).from = visualNode;
                }
            }
        }
        visualNode.build(document, nodeGroup);
        await visualNode.measure();
        await visualNode.layout();
        const sources = inputValues.map((item) => item.source.from).filter((value) => value);
        const horizontal = this._options.direction === 'horizontal';
        if (sources.length > 0) {
            const x = sources.reduce((sum, source) => sum + source.x, 0) / sources.length;
            const y = sources.reduce((sum, source) => sum + source.y, 0) / sources.length;
            visualNode.x = horizontal ? x + 140 : x;
            visualNode.y = horizontal ? y : y + 110;
        } else {
            visualNode.x = horizontal ? 120 : 0;
            visualNode.y = horizontal ? 0 : 100;
        }
        visualNode.update();
        for (const item of inputValues) {
            this._createGraphEditVisualEdge(item.target, item.source, item.source.value);
        }
        this._graphEdit.positions.set(node, { x: visualNode.x, y: visualNode.y });
        delete this._target._graphEditPorts;
        this._updateGraphEditPortStates();
        this._expandGraphEditCanvas();
        return true;
    }

    _updateGraphEditPortStates() {
        for (const element of this._host.document.querySelectorAll('.graph-edit-input-port')) {
            const argument = element.graphEditArgument;
            const valueIndex = element.graphEditValueIndex;
            const value = argument && Array.isArray(argument.value) ? argument.value[valueIndex] : null;
            const connected = Boolean(value && value.name);
            if (element.graphEditGraphOutput === true) {
                element.classList.toggle('graph-edit-graph-output-port-missing', !connected);
            } else {
                const optional = Boolean(argument && argument.option === 'optional');
                element.classList.toggle('graph-edit-input-port-connected', optional && connected);
                element.classList.toggle('graph-edit-input-port-required-missing', !optional && !connected);
            }
        }
    }

    _expandGraphEditCanvas(contentBounds = null) {
        const origin = this._element('origin');
        const canvas = this._element('canvas');
        const background = this._element('background');
        const container = this._element('target');
        if (!origin || !canvas || !background || !container || !this._target) {
            return;
        }
        const bounds = contentBounds || origin.getBBox();
        const transform = origin.transform.baseVal.consolidate();
        const matrix = transform ? transform.matrix : { e: 0, f: 0 };
        const margin = 100;
        const width = this._target._width || Number(canvas.getAttribute('width')) || 0;
        const height = this._target._height || Number(canvas.getAttribute('height')) || 0;
        const left = bounds.x + matrix.e;
        const top = bounds.y + matrix.f;
        const right = left + bounds.width;
        const bottom = top + bounds.height;
        const extraLeft = Math.max(0, Math.ceil(margin - left));
        const extraTop = Math.max(0, Math.ceil(margin - top));
        const extraRight = Math.max(0, Math.ceil(right + margin - width));
        const extraBottom = Math.max(0, Math.ceil(bottom + margin - height));
        if (extraLeft === 0 && extraTop === 0 && extraRight === 0 && extraBottom === 0) {
            return;
        }
        const nextWidth = width + extraLeft + extraRight;
        const nextHeight = height + extraTop + extraBottom;
        const zoom = this._target._zoom || 1;
        origin.setAttribute('transform', `translate(${matrix.e + extraLeft},${matrix.f + extraTop}) scale(1)`);
        background.setAttribute('width', nextWidth);
        background.setAttribute('height', nextHeight);
        canvas.setAttribute('viewBox', `0 0 ${nextWidth} ${nextHeight}`);
        canvas.setAttribute('width', nextWidth);
        canvas.setAttribute('height', nextHeight);
        canvas.style.width = `${nextWidth * zoom}px`;
        canvas.style.height = `${nextHeight * zoom}px`;
        this._target._width = nextWidth;
        this._target._height = nextHeight;
        if (extraLeft > 0) {
            container.scrollLeft += extraLeft * zoom;
        }
        if (extraTop > 0) {
            container.scrollTop += extraTop * zoom;
        }
        this._target._updateScrollThumbs();
    }

    _updateGraphEditDraggedNode(drag) {
        if (!drag || !this._target) {
            return;
        }
        const dx = (drag.clientX - drag.startClientX) / drag.scaleX;
        const dy = (drag.clientY - drag.startClientY) / drag.scaleY;
        drag.node.x = drag.startX + dx;
        drag.node.y = drag.startY + dy;
        for (const edge of drag.edges) {
            this._deformGraphEditEdge(edge, drag.routes.get(edge));
        }
        delete this._target._graphEditPorts;
        const nodes = new Set([drag.node]);
        for (const edge of drag.edges) {
            nodes.add(edge.from);
            nodes.add(edge.to);
        }
        for (const node of nodes) {
            if (node === drag.node) {
                node.update();
            } else {
                if (node._graphEditInputPorts) {
                    node._graphEditInputPorts.update();
                }
                if (node._graphEditOutputPorts) {
                    node._graphEditOutputPorts.update();
                }
            }
        }
        this._expandGraphEditCanvas({
            x: drag.node.x - drag.node.width / 2,
            y: drag.node.y - drag.node.height / 2,
            width: drag.node.width,
            height: drag.node.height
        });
    }

    beginGraphEditNodeDrag(node, event) {
        if (!this._graphEdit.enabled || !node || event.button !== 0 ||
            event.target.closest('.graph-edit-input-port, .graph-edit-output-port')) {
            return false;
        }
        this._cancelGraphEditSelection(false);
        event.preventDefault();
        event.stopPropagation();
        const origin = this._element('origin');
        const matrix = origin ? origin.getScreenCTM() : null;
        const scaleX = matrix && Math.abs(matrix.a) > 0.001 ? Math.abs(matrix.a) : 1;
        const scaleY = matrix && Math.abs(matrix.d) > 0.001 ? Math.abs(matrix.d) : 1;
        const edges = this._graphEditNodeEdges(node);
        const routes = new Map(edges.map((edge) => [edge, this._graphEditEdgeSnapshot(edge)]));
        const drag = {
            node,
            edges,
            routes,
            pointerId: event.pointerId,
            startClientX: event.clientX,
            startClientY: event.clientY,
            clientX: event.clientX,
            clientY: event.clientY,
            startX: node.x,
            startY: node.y,
            scaleX,
            scaleY,
            moved: false,
            frame: null
        };
        this._graphEdit.drag = drag;
        node.element.setPointerCapture(event.pointerId);
        node.element.classList.add('graph-edit-node-dragging');
        for (const edge of edges) {
            if (edge.element) {
                edge.element.classList.add('graph-edit-edge-dragging');
            }
            if (edge.labelElement) {
                edge.labelElement.classList.add('graph-edit-edge-dragging');
            }
        }
        this._host.document.documentElement.classList.add('onnx-graph-edit-node-dragging');
        let up = null;
        const move = (moveEvent) => {
            if (moveEvent.pointerId !== drag.pointerId) {
                return;
            }
            if ((moveEvent.buttons & 1) === 0) {
                up(moveEvent);
                return;
            }
            drag.clientX = moveEvent.clientX;
            drag.clientY = moveEvent.clientY;
            const dx = drag.clientX - drag.startClientX;
            const dy = drag.clientY - drag.startClientY;
            drag.moved = drag.moved || dx * dx + dy * dy > 9;
            if (!drag.frame) {
                drag.frame = this._host.window.requestAnimationFrame(() => {
                    drag.frame = null;
                    this._updateGraphEditDraggedNode(drag);
                });
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
        };
        up = (upEvent) => {
            if (upEvent.pointerId !== drag.pointerId) {
                return;
            }
            if (drag.frame) {
                this._host.window.cancelAnimationFrame(drag.frame);
                drag.frame = null;
            }
            this._updateGraphEditDraggedNode(drag);
            if (node.element.hasPointerCapture(upEvent.pointerId)) {
                node.element.releasePointerCapture(upEvent.pointerId);
            }
            node.element.removeEventListener('pointermove', move);
            node.element.removeEventListener('pointerup', up);
            node.element.removeEventListener('pointercancel', up);
            node.element.classList.remove('graph-edit-node-dragging');
            for (const edge of edges) {
                if (edge.element) {
                    edge.element.classList.remove('graph-edit-edge-dragging');
                }
                if (edge.labelElement) {
                    edge.labelElement.classList.remove('graph-edit-edge-dragging');
                }
            }
            this._host.document.documentElement.classList.remove('onnx-graph-edit-node-dragging');
            this._graphEdit.drag = null;
            if (drag.moved) {
                this._graphEdit.positions.set(node.value, { x: node.x, y: node.y });
                this._expandGraphEditCanvas();
                this._updateGraphEditStatus(
                    `Moved ${node.value.name || node.value.type.name} · Only ${edges.length} connected line${edges.length === 1 ? '' : 's'} redrawn.`
                );
                this._suppressGraphEditNativeClick(upEvent);
            }
            upEvent.preventDefault();
            upEvent.stopPropagation();
        };
        node.element.addEventListener('pointermove', move);
        node.element.addEventListener('pointerup', up);
        node.element.addEventListener('pointercancel', up);
        return true;
    }

    _applyGraphEditPositions(graph) {
        if (!graph || this._graphEdit.positions.size === 0) {
            return;
        }
        const moved = new Set();
        const routes = new Map();
        for (const entry of graph.edges.values()) {
            const edge = entry.label;
            routes.set(edge, this._graphEditEdgeSnapshot(edge));
        }
        for (const entry of graph.nodes.values()) {
            const node = entry.label;
            const position = node && this._graphEdit.positions.get(node.value);
            if (position) {
                const changed = Math.abs(node.x - position.x) > 0.01 ||
                    Math.abs(node.y - position.y) > 0.01;
                node.x = position.x;
                node.y = position.y;
                if (changed) {
                    moved.add(node);
                }
            }
        }
        if (moved.size > 0) {
            for (const entry of graph.edges.values()) {
                const edge = entry.label;
                if (edge && (moved.has(edge.from) || moved.has(edge.to))) {
                    this._deformGraphEditEdge(edge, routes.get(edge));
                }
            }
            delete graph._graphEditPorts;
        }
    }

    _preserveVisibleGraphEditPositions() {
        if (!this._target || !this._target.nodes) {
            return;
        }
        for (const entry of this._target.nodes.values()) {
            const node = entry && entry.label;
            if (node && node.value && Number.isFinite(node.x) && Number.isFinite(node.y)) {
                this._graphEdit.positions.set(node.value, { x: node.x, y: node.y });
            }
        }
    }

    _graphEditPositionDelta(value) {
        const position = this._graphEdit.positions.get(value);
        return position ? new Map([[value, { x: position.x, y: position.y }]]) : new Map();
    }

    _cancelGraphEditSelection(update = true, preserveLinkDrag = false) {
        if (!preserveLinkDrag) {
            this._disposeGraphEditLinkDrag();
        }
        this._closeGraphEditAddDialog();
        this._closeGraphEditNodeMenu();
        const pending = this._graphEdit.pending;
        if (pending && pending.port) {
            pending.port.classList.remove('graph-edit-output-port-selected');
        }
        const connection = this._graphEdit.connection;
        if (connection && connection.edge && connection.edge.element) {
            connection.edge.element.classList.remove('graph-edit-edge-selected');
        }
        this._graphEdit.pending = null;
        this._graphEdit.connection = null;
        this._host.document.documentElement.classList.remove('onnx-graph-edit-source');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-connection');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-connection-replace');
        if (update) {
            this._updateGraphEditStatus();
        }
    }

    _resetGraphEdit() {
        this._graphEdit.enabled = false;
        this._graphEdit.pending = null;
        this._graphEdit.connection = null;
        this._graphEdit.nodeMenu = null;
        this._graphEdit.undo = [];
        this._graphEdit.redo = [];
        this._graphEdit.viewDirty = false;
        this._graphEdit.viewRevision = 0;
        this._graphEdit.redrawing = false;
        this._graphEdit.renderPromise = null;
        this._graphEdit.historyPromise = null;
        this._graphEdit.historyPending = 0;
        this._graphEdit.positions.clear();
        this._graphEdit.drag = null;
        this._graphEdit.encodingsDisabled = false;
        this._closeGraphEditEncodingWarning(false);
        this._disposeGraphEditLinkDrag();
        this._closeGraphEditAddDialog();
        this._host.document.documentElement.classList.remove('onnx-graph-edit');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-source');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-connection');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-connection-replace');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-node-dragging');
        this._host.document.documentElement.classList.remove('onnx-graph-edit-encodings-disabled');
        const button = this._element('graph-edit-button');
        if (button) {
            button.textContent = 'EDIT';
            button.setAttribute('title', 'Enter ONNX GraphSurgeon Editor (Beta) (E)');
        }
        this._updateGraphEditStatus();
    }

    _clearGraphEditDerivedState() {
        this._clearGraphShapeInference();
        // AIMET encodings describe the model as loaded. The beta ONNX editor
        // intentionally does not rewrite or rebind those external encodings.
    }

    _graphValues() {
        const graph = this.activeTarget;
        const values = new Set();
        const collect = (arguments_) => {
            for (const argument of arguments_ || []) {
                for (const value of argument && Array.isArray(argument.value) ? argument.value : []) {
                    if (value && value.name) {
                        values.add(value);
                    }
                }
            }
        };
        if (graph) {
            collect(graph.inputs);
            collect(graph.outputs);
            for (const node of graph.nodes || []) {
                collect(node.inputs);
                collect(node.outputs);
            }
        }
        return Array.from(values);
    }

    _clearGraphShapeInference() {
        for (const value of this._graphValues()) {
            if (typeof value.clearShapeInference === 'function') {
                value.clearShapeInference();
            }
        }
    }

    _updateGraphEditStatus(message) {
        const status = this._element('graph-edit-status');
        const dirty = this._graphEdit.undo.length > 0;
        const invalid = this._invalidGraphEditConnections();
        if (status) {
            let fallback = dirty ?
                `ONNX GRAPHSURGEON · ${this._graphEdit.undo.length} unsaved change${this._graphEdit.undo.length === 1 ? '' : 's'} · Select a connection or orange output` :
                'ONNX GRAPHSURGEON · Select a connection line, or an orange output';
            if (this._graphEdit.viewDirty) {
                fallback = `VIEW REFRESH PENDING · ${this._graphEdit.undo.length} unsaved change${this._graphEdit.undo.length === 1 ? '' : 's'} · Press REFRESH VIEW`;
            }
            if (invalid.length > 0) {
                fallback = `INCOMPLETE GRAPH · ${invalid.length} required connection${invalid.length === 1 ? '' : 's'} disconnected · Reconnect before save`;
            }
            status.textContent = message || fallback;
            status.classList.toggle('invalid', invalid.length > 0);
        }
        const undo = this._element('graph-edit-undo-button');
        const redo = this._element('graph-edit-redo-button');
        const reset = this._element('graph-edit-reset-button');
        const redraw = this._element('graph-edit-redraw-button');
        const layout = this._element('graph-edit-layout-button');
        const save = this._element('graph-edit-save-button');
        if (undo) {
            undo.disabled = this._graphEdit.undo.length === 0 || this._graphEdit.historyPending > 0;
            undo.title = undo.disabled ? 'No graph edit to undo' :
                `Undo: ${this._graphEdit.undo.at(-1).label}`;
        }
        if (redo) {
            redo.disabled = this._graphEdit.redo.length === 0 || this._graphEdit.historyPending > 0;
            redo.title = redo.disabled ? 'No graph edit to redo' :
                `Redo: ${this._graphEdit.redo.at(-1).label}`;
        }
        if (reset) {
            reset.disabled = this._graphEdit.undo.length === 0 && this._graphEdit.positions.size === 0;
            reset.title = reset.disabled ? 'No unsaved edits or manual positions to reset' :
                `Restore the edit-session graph and ${this._graphEdit.positions.size} manual position${this._graphEdit.positions.size === 1 ? '' : 's'}`;
        }
        if (redraw) {
            redraw.disabled = !this._graphEdit.viewDirty || this._graphEdit.redrawing;
            redraw.title = 'The graph view is already up to date';
            if (this._graphEdit.viewDirty) {
                redraw.title = 'Refresh the graph view from the edited model without changing the layout';
            }
            if (this._graphEdit.redrawing) {
                redraw.title = 'Refreshing the graph view…';
            }
        }
        if (layout) {
            layout.disabled = Boolean(this._graphEdit.renderPromise);
            layout.textContent = layout.disabled ? 'LAYOUT…' : 'RE-LAYOUT';
            layout.title = layout.disabled ? 'Graph update in progress' :
                'Rebuild and re-layout the current graph (R)';
        }
        if (save) {
            const saveInvalid = dirty ? invalid : [];
            save.disabled = !this.activeTarget || saveInvalid.length > 0;
            save.title = 'Save the current ONNX to a chosen file';
            if (saveInvalid.length > 0) {
                save.title = `Reconnect required graph connections before save: ${saveInvalid.join(', ')}`;
            }
        }
    }

    _invalidGraphEditConnections() {
        const invalid = [];
        const nodes = this.activeTarget && Array.isArray(this.activeTarget.nodes) ? this.activeTarget.nodes : [];
        for (const node of nodes) {
            if (!node.source || node.source._graph_edit_root !== true) {
                continue;
            }
            for (const argument of node.inputs || []) {
                if (argument.option === 'optional') {
                    continue;
                }
                for (let index = 0; index < (argument.value || []).length; index++) {
                    const value = argument.value[index];
                    if (value && !value.initializer && !value.name) {
                        const suffix = argument.value.length > 1 ? `[${index}]` : '';
                        invalid.push(`${node.name || node.type.name}.${argument.name}${suffix}`);
                    }
                }
            }
        }
        const outputs = this.activeTarget && Array.isArray(this.activeTarget.outputs) ? this.activeTarget.outputs : [];
        for (const argument of outputs) {
            if (argument._graph_edit_root !== true) {
                continue;
            }
            for (let index = 0; index < (argument.value || []).length; index++) {
                const value = argument.value[index];
                if (!value || !value.name) {
                    const suffix = argument.value.length > 1 ? `[${index}]` : '';
                    invalid.push(`graph output ${argument.name}${suffix}`);
                }
            }
        }
        return invalid;
    }

    _graphEditTargetName(target) {
        if (target.graphOutput === true) {
            const suffix = target.argument.value.length > 1 ? `[${target.valueIndex}]` : '';
            return `graph output ${target.argument.name}${suffix}`;
        }
        const suffix = target.argument.value.length > 1 ? `[${target.valueIndex}]` : '';
        return `${target.node.name || target.node.type.name}.${target.argument.name}${suffix}`;
    }

    async renameGraphEditNode(node, name) {
        name = typeof name === 'string' ? name.trim() : '';
        if (!name) {
            this._updateGraphEditStatus('Node name cannot be empty.');
            return;
        }
        if ((this.activeTarget.nodes || []).some((entry) => entry !== node && entry.name === name)) {
            this._updateGraphEditStatus(`Node name '${name}' is already in use.`);
            return;
        }
        const oldName = node.name;
        if (oldName === name) {
            return;
        }
        const apply = () => {
            node.name = name;
            node.source.name = name;
        };
        const revert = () => {
            node.name = oldName;
            node.source.name = oldName;
        };
        const command = {
            kind: 'rename-node',
            label: `rename node ${oldName || node.type.name} → ${name}`,
            apply,
            revert,
            nodeIndex: node.source._graph_edit_index,
            nodeAddedId: node.source._graph_edit_added_id,
            name
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._deferGraphEditRefresh(`Renamed node ${oldName || node.type.name} → ${name}`);
    }

    async renameGraphEditOutput(node, argument, valueIndex, name) {
        name = typeof name === 'string' ? name.trim() : '';
        if (!name) {
            this._updateGraphEditStatus('Output tensor name cannot be empty.');
            return;
        }
        const value = argument && Array.isArray(argument.value) ? argument.value[valueIndex] : null;
        const source = argument ? argument.source : null;
        const outputIndex = source && Array.isArray(source.indices) ? source.indices[valueIndex] : -1;
        if (!value || typeof value.rename !== 'function' || outputIndex < 0) {
            this._updateGraphEditStatus('This output does not map to an editable ONNX output slot.');
            return;
        }
        const existing = this._graphValues().find((entry) => entry !== value && entry.name === name);
        if (existing) {
            this._updateGraphEditStatus(`Tensor name '${name}' is already in use.`);
            return;
        }
        const oldName = value.name;
        if (oldName === name) {
            return;
        }
        const graphOutputs = (this.activeTarget.outputs || []).filter((entry) =>
            Array.isArray(entry.value) && entry.value.includes(value));
        const rename = (next) => {
            value.rename(next);
            for (const output of graphOutputs) {
                if (output.value.length === 1) {
                    output.name = next;
                }
            }
        };
        const command = {
            kind: 'rename-output',
            label: `rename tensor ${oldName} → ${name}`,
            apply: () => rename(name),
            revert: () => rename(oldName),
            nodeIndex: node.source._graph_edit_index,
            nodeAddedId: node.source._graph_edit_added_id,
            outputIndex,
            name
        };
        command.apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._clearGraphEditDerivedState();
        this._deferGraphEditRefresh(`Renamed output tensor ${oldName} → ${name}`);
    }

    async renameGraphEditGraphOutput(argument, name) {
        name = typeof name === 'string' ? name.trim() : '';
        if (!name) {
            this._updateGraphEditStatus('Graph output name cannot be empty.');
            return;
        }
        if (!argument || argument._graph_edit_root !== true ||
            !Number.isInteger(argument._graph_edit_output_index) ||
            !Array.isArray(argument.value) || argument.value.length !== 1) {
            this._updateGraphEditStatus('This graph output is not editable.');
            return;
        }
        const existing = (this.activeTarget.outputs || []).find((entry) =>
            entry !== argument && entry.name === name);
        if (existing) {
            this._updateGraphEditStatus(`Graph output name '${name}' is already in use.`);
            return;
        }
        const oldName = argument.name;
        if (oldName === name) {
            return;
        }
        const value = argument.value[0];
        const renameTensor = value && value.name === oldName && typeof value.rename === 'function';
        const apply = () => {
            argument.name = name;
            if (renameTensor) {
                value.rename(name);
            }
        };
        const revert = () => {
            argument.name = oldName;
            if (renameTensor) {
                value.rename(oldName);
            }
        };
        const command = {
            kind: 'rename-graph-output',
            label: `rename graph output ${oldName} → ${name}`,
            apply,
            revert,
            outputIndex: argument._graph_edit_output_index,
            outputAddedId: argument._graph_edit_added_id,
            name
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._clearGraphEditDerivedState();
        this._deferGraphEditRefresh(`Renamed graph output ${oldName} → ${name}`);
    }

    graphEditGraphOutputMenu(argument, event) {
        if (!this._graphEdit.enabled || !argument || argument._graph_edit_root !== true ||
            !Number.isInteger(argument._graph_edit_output_index)) {
            return false;
        }
        event.preventDefault();
        event.stopPropagation();
        this._cancelGraphEditSelection(false);
        const menu = this._element('graph-edit-node-menu');
        if (!menu) {
            return true;
        }
        this._graphEdit.nodeMenu = argument;
        menu.replaceChildren();
        const document = this._host.document;
        const title = document.createElement('div');
        title.className = 'graph-edit-node-menu-title';
        title.textContent = argument.name;
        menu.appendChild(title);
        const subtitle = document.createElement('div');
        subtitle.className = 'graph-edit-node-menu-subtitle';
        subtitle.textContent = 'Graph output · external interface';
        menu.appendChild(subtitle);
        const section = document.createElement('div');
        section.className = 'graph-edit-node-menu-section';
        section.textContent = 'Identity';
        menu.appendChild(section);
        const field = document.createElement('div');
        field.className = 'graph-edit-node-field';
        const label = document.createElement('label');
        label.textContent = 'Output name';
        const name = document.createElement('input');
        name.type = 'text';
        name.value = argument.name;
        name.spellcheck = false;
        name.setAttribute('aria-label', 'Graph output name');
        const apply = document.createElement('button');
        apply.textContent = 'APPLY';
        const rename = async () => await this.renameGraphEditGraphOutput(argument, name.value);
        apply.addEventListener('click', rename);
        name.addEventListener('keydown', async (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                await rename();
            }
        });
        field.append(label, name, apply);
        menu.appendChild(field);
        const source = document.createElement('div');
        source.className = 'graph-edit-node-menu-warning';
        const value = Array.isArray(argument.value) ? argument.value[0] : null;
        source.textContent = `Current source tensor: ${value && value.name ? value.name : '(none)'}`;
        menu.appendChild(source);
        const footer = document.createElement('div');
        footer.className = 'graph-edit-node-menu-footer';
        const clear = document.createElement('button');
        clear.className = 'danger';
        clear.textContent = 'DISCONNECT';
        clear.disabled = !value || !value.name;
        clear.title = 'Temporarily disconnect this graph output; save will be blocked until it is reconnected';
        clear.addEventListener('click', async () => {
            await this._disconnectGraphEditGraphOutput({
                graphOutput: true,
                argument,
                valueIndex: 0,
                oldValue: value
            });
        });
        const remove = document.createElement('button');
        remove.className = 'danger';
        remove.textContent = 'DELETE OUTPUT';
        remove.title = 'Delete this graph output';
        remove.addEventListener('click', async () => await this.deleteGraphEditGraphOutput(argument));
        const close = document.createElement('button');
        close.textContent = 'CLOSE';
        close.addEventListener('click', () => this._closeGraphEditNodeMenu());
        footer.append(clear, remove, close);
        menu.appendChild(footer);
        menu.classList.add('visible');
        const bounds = menu.getBoundingClientRect();
        const left = Math.max(8, Math.min(event.clientX, this._host.window.innerWidth - bounds.width - 8));
        const top = Math.max(8, Math.min(event.clientY, this._host.window.innerHeight - bounds.height - 8));
        menu.style.left = `${left}px`;
        menu.style.top = `${top}px`;
        return true;
    }

    graphEditGraphInputMenu(argument, event) {
        if (!this._graphEdit.enabled || !argument || argument._graph_edit_root !== true ||
            (!Number.isInteger(argument._graph_edit_input_index) && !argument._graph_edit_added_id)) {
            return false;
        }
        event.preventDefault();
        event.stopPropagation();
        this._cancelGraphEditSelection(false);
        const menu = this._element('graph-edit-node-menu');
        if (!menu) {
            return true;
        }
        this._graphEdit.nodeMenu = argument;
        menu.replaceChildren();
        const document = this._host.document;
        const title = document.createElement('div');
        title.className = 'graph-edit-node-menu-title';
        title.textContent = argument.name;
        const subtitle = document.createElement('div');
        subtitle.className = 'graph-edit-node-menu-subtitle';
        subtitle.textContent = 'Graph input · external interface';
        const warning = document.createElement('div');
        warning.className = 'graph-edit-node-menu-warning';
        warning.textContent = 'Deleting an input disconnects its consumers. Reconnect required ports before saving.';
        const footer = document.createElement('div');
        footer.className = 'graph-edit-node-menu-footer';
        const remove = document.createElement('button');
        remove.className = 'danger';
        remove.textContent = 'DELETE INPUT';
        remove.title = 'Delete this graph input';
        remove.addEventListener('click', async () => await this.deleteGraphEditGraphInput(argument));
        const close = document.createElement('button');
        close.textContent = 'CLOSE';
        close.addEventListener('click', () => this._closeGraphEditNodeMenu());
        footer.append(remove, close);
        menu.append(title, subtitle, warning, footer);
        menu.classList.add('visible');
        const bounds = menu.getBoundingClientRect();
        menu.style.left = `${Math.max(8, Math.min(event.clientX, this._host.window.innerWidth - bounds.width - 8))}px`;
        menu.style.top = `${Math.max(8, Math.min(event.clientY, this._host.window.innerHeight - bounds.height - 8))}px`;
        return true;
    }

    graphEditNodeMenu(node, event) {
        if (!this._graphEdit.enabled || !node || !node.source ||
            node.source._graph_edit_root !== true ||
            (!Number.isInteger(node.source._graph_edit_index) && !node.source._graph_edit_added_id)) {
            return false;
        }
        event.graphEditNodeMenu = true;
        event.preventDefault();
        event.stopPropagation();
        this._cancelGraphEditSelection(false);
        const menu = this._element('graph-edit-node-menu');
        if (!menu) {
            return true;
        }
        this._graphEdit.nodeMenu = node;
        menu.replaceChildren();
        const document = this._host.document;
        const title = document.createElement('div');
        title.className = 'graph-edit-node-menu-title';
        title.textContent = node.name || node.type.name;
        menu.appendChild(title);
        const subtitle = document.createElement('div');
        subtitle.className = 'graph-edit-node-menu-subtitle';
        subtitle.textContent = `${node.type.name} · left-click edit`;
        menu.appendChild(subtitle);
        const identitySection = document.createElement('div');
        identitySection.className = 'graph-edit-node-menu-section';
        identitySection.textContent = 'Identity';
        menu.appendChild(identitySection);
        const nodeField = document.createElement('div');
        nodeField.className = 'graph-edit-node-field';
        const nodeLabel = document.createElement('label');
        nodeLabel.textContent = 'Node name';
        const nodeName = document.createElement('input');
        nodeName.type = 'text';
        nodeName.value = node.name;
        nodeName.placeholder = node.type.name;
        nodeName.spellcheck = false;
        nodeName.setAttribute('aria-label', 'Node name');
        const applyNodeName = document.createElement('button');
        applyNodeName.textContent = 'APPLY';
        const renameNode = async () => await this.renameGraphEditNode(node, nodeName.value);
        applyNodeName.addEventListener('click', renameNode);
        nodeName.addEventListener('keydown', async (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                await renameNode();
            }
        });
        nodeField.append(nodeLabel, nodeName, applyNodeName);
        menu.appendChild(nodeField);
        const inputSection = document.createElement('div');
        inputSection.className = 'graph-edit-node-menu-section';
        inputSection.textContent = 'Inputs';
        menu.appendChild(inputSection);
        const inputs = [];
        for (const argument of node.inputs || []) {
            if (!Array.isArray(argument.value) || argument.value.length === 0) {
                argument.value = [this._graphEditTensorValue('')];
            }
            for (let valueIndex = 0; valueIndex < argument.value.length; valueIndex++) {
                inputs.push({ argument, valueIndex, value: argument.value[valueIndex] });
            }
        }
        if (inputs.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'graph-edit-node-menu-empty';
            empty.textContent = 'This operator has no tensor inputs.';
            menu.appendChild(empty);
        }
        const values = this._graphValues().filter((value) => value && value.name && !value.initializer);
        for (const item of inputs) {
            const optional = item.argument.option === 'optional';
            const initializer = Boolean(item.value && item.value.initializer);
            const connected = Boolean(item.value && item.value.name);
            const row = document.createElement('div');
            row.className = `graph-edit-node-port ${optional ? 'graph-edit-node-port-optional' : 'graph-edit-node-port-required'}${connected ? ' graph-edit-node-port-connected' : ' graph-edit-node-port-disconnected'}`;
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = connected;
            checkbox.disabled = !optional || initializer;
            checkbox.title = 'Required ONNX inputs cannot be disabled';
            if (optional) {
                checkbox.title = connected ? 'Disconnect this optional input' : 'Enable this optional input';
            }
            let control = checkbox;
            if (!optional || initializer) {
                control = document.createElement('span');
                control.className = `graph-edit-node-port-indicator ${initializer ? 'initializer' : 'required'}`;
                control.textContent = initializer ? '\u25A0' : '\u25CF';
                control.title = initializer ? 'Initializer input is read-only' : 'Required input';
            }
            const name = document.createElement('div');
            name.className = 'graph-edit-node-port-name';
            const suffix = item.argument.value.length > 1 ? `[${item.valueIndex}]` : '';
            name.textContent = `${item.argument.name}${suffix}`;
            const state = document.createElement('div');
            state.className = 'graph-edit-node-port-state';
            state.textContent = 'REQUIRED';
            if (initializer) {
                state.textContent = 'INITIALIZER';
            } else if (optional) {
                state.textContent = connected ? 'OPTIONAL · ON' : 'OPTIONAL · NO INPUT';
            } else if (!connected) {
                state.textContent = 'REQUIRED · MISSING';
            }
            const search = document.createElement('input');
            search.type = 'search';
            search.placeholder = 'Search tensor or choose No input';
            search.value = item.value && item.value.name ? item.value.name : '';
            search.autocomplete = 'off';
            search.spellcheck = false;
            search.disabled = initializer;
            search.setAttribute('aria-label', `Tensor for ${optional ? 'optional ' : ''}input ${item.argument.name}`);
            const candidates = document.createElement('div');
            candidates.className = 'graph-edit-node-candidates';
            candidates.setAttribute('role', 'listbox');
            const pick = document.createElement('button');
            pick.textContent = 'PICK';
            pick.title = 'Choose a source directly on the graph';
            pick.disabled = initializer;
            pick.addEventListener('click', () => this.beginGraphEditOptionalInput(
                node, item.argument, item.valueIndex));
            const clear = document.createElement('button');
            clear.textContent = optional ? 'NO INPUT' : 'CLEAR';
            clear.title = optional ?
                'Disconnect this input' :
                'Temporarily disconnect this required input; save will be blocked until it is reconnected';
            clear.disabled = initializer || !connected;
            clear.addEventListener('click', async () => {
                const target = this._graphEditInputTarget(node, item.argument, item.valueIndex, null);
                if (target && target.oldValue.name) {
                    await this._disconnectGraphEditTarget(target);
                }
            });
            const actions = document.createElement('div');
            actions.className = 'graph-edit-node-port-actions';
            actions.append(pick, clear);
            let connectedValue = item.value && item.value.name ? item.value.name : '';
            let filteredValues = [];
            const connectValue = async (value) => {
                if (value && value.name !== connectedValue) {
                    connectedValue = value.name;
                    search.value = value.name;
                    await this.graphEditOptionalInput(node, item.argument, item.valueIndex, value);
                }
            };
            const disconnectValue = async () => {
                if (connectedValue) {
                    const target = this._graphEditInputTarget(node, item.argument, item.valueIndex, null);
                    if (target && target.oldValue.name) {
                        await this._disconnectGraphEditTarget(target);
                    }
                } else {
                    checkbox.checked = false;
                    search.value = '';
                    candidates.classList.remove('visible');
                    this._updateGraphEditStatus(`No input selected for ${item.argument.name}.`);
                }
            };
            const renderCandidates = (query) => {
                candidates.replaceChildren();
                const normalized = query.trim().toLowerCase();
                filteredValues = values.filter((value) =>
                    !normalized || value.name.toLowerCase().includes(normalized));
                const showNone = !initializer && (!normalized || 'no input'.includes(normalized));
                if (showNone) {
                    const none = document.createElement('button');
                    none.type = 'button';
                    none.className = 'graph-edit-node-candidate graph-edit-node-candidate-none';
                    none.textContent = 'No input';
                    none.setAttribute('role', 'option');
                    none.classList.toggle('current', !connectedValue);
                    none.addEventListener('mousedown', (event) => event.preventDefault());
                    none.addEventListener('click', disconnectValue);
                    candidates.appendChild(none);
                }
                const shown = filteredValues.slice(0, 100);
                for (const value of shown) {
                    const candidate = document.createElement('button');
                    candidate.type = 'button';
                    candidate.className = 'graph-edit-node-candidate';
                    candidate.textContent = value.name;
                    candidate.setAttribute('role', 'option');
                    candidate.classList.toggle('current', value.name === connectedValue);
                    candidate.addEventListener('mousedown', (event) => event.preventDefault());
                    candidate.addEventListener('click', async () => await connectValue(value));
                    candidates.appendChild(candidate);
                }
                if (shown.length === 0 && !showNone) {
                    const empty = document.createElement('div');
                    empty.className = 'graph-edit-node-candidate-empty';
                    empty.textContent = 'No matching tensor';
                    candidates.appendChild(empty);
                } else if (filteredValues.length > shown.length) {
                    const more = document.createElement('div');
                    more.className = 'graph-edit-node-candidate-empty';
                    more.textContent = `${filteredValues.length - shown.length} more · keep typing to narrow`;
                    candidates.appendChild(more);
                }
                candidates.classList.add('visible');
            };
            checkbox.addEventListener('change', async () => {
                if (!checkbox.checked) {
                    const target = this._graphEditInputTarget(node, item.argument, item.valueIndex, null);
                    if (target && target.oldValue.name) {
                        await this._disconnectGraphEditTarget(target);
                    }
                } else if (search.value) {
                    const value = values.find((entry) => entry.name === search.value);
                    if (value) {
                        await connectValue(value);
                    } else {
                        checkbox.checked = false;
                        search.setCustomValidity('Choose an existing tensor from the filtered suggestions.');
                        search.reportValidity();
                    }
                } else {
                    checkbox.checked = false;
                    search.focus();
                    this._updateGraphEditStatus(`Choose a tensor for optional input ${item.argument.name}.`);
                }
            });
            const connect = async () => {
                const value = values.find((entry) => entry.name === search.value);
                if (value) {
                    await connectValue(value);
                } else if (search.value) {
                    search.setCustomValidity('Choose an existing tensor from the filtered suggestions.');
                    search.reportValidity();
                }
            };
            search.addEventListener('focus', () => {
                search.select();
                renderCandidates('');
            });
            search.addEventListener('input', () => {
                search.setCustomValidity('');
                renderCandidates(search.value);
            });
            search.addEventListener('blur', () => {
                this._host.window.setTimeout(() => {
                    candidates.classList.remove('visible');
                    if (!search.value) {
                        search.value = connectedValue;
                    }
                }, 120);
            });
            search.addEventListener('keydown', async (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    const exact = values.find((entry) => entry.name === search.value);
                    if (exact) {
                        await connectValue(exact);
                    } else if (filteredValues.length === 1) {
                        await connectValue(filteredValues[0]);
                    } else {
                        await connect();
                    }
                } else if (event.key === 'Escape') {
                    candidates.classList.remove('visible');
                    search.value = connectedValue;
                    search.blur();
                }
            });
            row.append(control, name, state, search, actions, candidates);
            menu.appendChild(row);
        }
        const outputSection = document.createElement('div');
        outputSection.className = 'graph-edit-node-menu-section';
        outputSection.textContent = 'Outputs';
        menu.appendChild(outputSection);
        for (const argument of node.outputs || []) {
            for (let valueIndex = 0; valueIndex < argument.value.length; valueIndex++) {
                const value = argument.value[valueIndex];
                if (!value || !value.name || value.initializer) {
                    continue;
                }
                const field = document.createElement('div');
                field.className = 'graph-edit-node-field graph-edit-node-output-field';
                const label = document.createElement('label');
                const suffix = argument.value.length > 1 ? `[${valueIndex}]` : '';
                label.textContent = `${argument.name}${suffix}`;
                const outputName = document.createElement('input');
                outputName.type = 'text';
                outputName.value = value.name;
                outputName.spellcheck = false;
                outputName.setAttribute('aria-label', `Output tensor ${value.name}`);
                const applyOutputName = document.createElement('button');
                applyOutputName.textContent = 'RENAME';
                const renameOutput = async () => await this.renameGraphEditOutput(
                    node, argument, valueIndex, outputName.value);
                applyOutputName.addEventListener('click', renameOutput);
                outputName.addEventListener('keydown', async (event) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        await renameOutput();
                    }
                });
                field.append(label, outputName, applyOutputName);
                menu.appendChild(field);
            }
        }
        const warning = document.createElement('div');
        warning.className = 'graph-edit-node-menu-warning';
        warning.textContent = 'ONNX-internal references are renamed together. External files are not updated.';
        menu.appendChild(warning);
        const footer = document.createElement('div');
        footer.className = 'graph-edit-node-menu-footer';
        const remove = document.createElement('button');
        remove.className = 'danger';
        remove.textContent = 'DELETE NODE';
        const deleteError = this._validateGraphEditNodeDelete(node);
        remove.disabled = Boolean(deleteError);
        remove.title = deleteError || 'Delete this node; its consumers will be left disconnected until reconnected';
        remove.addEventListener('click', async () => await this.deleteGraphEditNode(node));
        const close = document.createElement('button');
        close.textContent = 'CLOSE';
        close.addEventListener('click', () => this._closeGraphEditNodeMenu());
        footer.append(remove, close);
        menu.appendChild(footer);
        menu.classList.add('visible');
        const bounds = menu.getBoundingClientRect();
        const left = Math.max(8, Math.min(event.clientX, this._host.window.innerWidth - bounds.width - 8));
        const top = Math.max(8, Math.min(event.clientY, this._host.window.innerHeight - bounds.height - 8));
        menu.style.left = `${left}px`;
        menu.style.top = `${top}px`;
        return true;
    }

    _closeGraphEditNodeMenu() {
        const menu = this._element('graph-edit-node-menu');
        if (menu) {
            menu.classList.remove('visible');
            menu.classList.remove('graph-edit-port-choice-menu');
            menu.classList.remove('graph-edit-inference-menu');
        }
        this._graphEdit.nodeMenu = null;
    }

    _validateGraphEditNodeDelete(node) {
        return node && node.source && node.source._graph_edit_root === true ? null :
            'This node does not map to an editable ONNX graph node.';
    }

    async deleteGraphEditGraphOutput(argument) {
        if (!argument || !argument._graph_edit_root) {
            return;
        }
        const outputs = this.activeTarget.outputs;
        const index = outputs.indexOf(argument);
        if (index < 0) {
            return;
        }
        const apply = () => outputs.splice(outputs.indexOf(argument), 1);
        const revert = () => outputs.splice(Math.min(index, outputs.length), 0, argument);
        const command = {
            kind: 'delete-output', label: `delete graph output ${argument.name}`, apply, revert,
            outputIndex: argument._graph_edit_output_index, outputAddedId: argument._graph_edit_added_id,
            refreshOnHistory: true, positionsBefore: this._graphEditPositionDelta(argument)
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._clearGraphEditDerivedState();
        this._deferGraphEditRefresh(`Deleted graph output ${argument.name}`);
        await this.redrawGraphEdit();
        this._updateGraphEditStatus(`Deleted graph output ${argument.name} · Graph view updated.`);
    }

    async deleteGraphEditGraphInput(argument) {
        if (!argument || !argument._graph_edit_root) {
            return;
        }
        const inputs = this.activeTarget.inputs;
        const index = inputs.indexOf(argument);
        if (index < 0) {
            return;
        }
        const value = argument.value && argument.value[0];
        const affected = [];
        for (const node of this.activeTarget.nodes || []) {
            for (const input of node.inputs || []) {
                for (let valueIndex = 0; valueIndex < (input.value || []).length; valueIndex++) {
                    if (input.value[valueIndex] === value) {
                        const sourceIndex = input.source && input.source.indices ? input.source.indices[valueIndex] : -1;
                        affected.push({ input, valueIndex, sourceIndex, oldValue: value });
                    }
                }
            }
        }
        const apply = () => {
            inputs.splice(inputs.indexOf(argument), 1);
            for (const target of affected) {
                const empty = this._graphEditTensorValue('', target.oldValue.type);
                target.input.value[target.valueIndex] = empty;
                if (target.input.source && target.input.source.node && target.sourceIndex >= 0) {
                    target.input.source.node.input[target.sourceIndex] = { name: '' };
                }
            }
            this._updateGraphEditPortStates();
        };
        const revert = () => {
            inputs.splice(Math.min(index, inputs.length), 0, argument);
            for (const target of affected) {
                target.input.value[target.valueIndex] = target.oldValue;
                if (target.input.source && target.input.source.node && target.sourceIndex >= 0) {
                    target.input.source.node.input[target.sourceIndex] = { name: target.oldValue.name };
                }
            }
            this._updateGraphEditPortStates();
        };
        const command = {
            kind: 'delete-input', label: `delete graph input ${argument.name}`, apply, revert,
            inputIndex: argument._graph_edit_input_index, inputAddedId: argument._graph_edit_added_id,
            refreshOnHistory: true, positionsBefore: this._graphEditPositionDelta(argument)
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._clearGraphEditDerivedState();
        this._deferGraphEditRefresh(`Deleted graph input ${argument.name}`);
        await this.redrawGraphEdit();
        this._updateGraphEditStatus(`Deleted graph input ${argument.name} · Graph view updated.`);
    }

    async deleteGraphEditNode(node) {
        const error = this._validateGraphEditNodeDelete(node);
        if (error) {
            this._updateGraphEditStatus(error);
            return;
        }
        const nodes = this.activeTarget.nodes;
        const index = nodes.indexOf(node);
        if (index < 0) {
            return;
        }
        const outputs = new Set((node.outputs || []).flatMap((argument) => argument.value || []).filter((value) => value && value.name));
        const affectedInputs = [];
        const affectedOutputs = [];
        for (const other of nodes) {
            if (other === node) {
                continue;
            }
            for (const argument of other.inputs || []) {
                for (let valueIndex = 0; valueIndex < (argument.value || []).length; valueIndex++) {
                    if (outputs.has(argument.value[valueIndex])) {
                        const sourceIndex = argument.source && argument.source.indices ? argument.source.indices[valueIndex] : -1;
                        affectedInputs.push({ argument, valueIndex, source: argument.source, sourceIndex, oldValue: argument.value[valueIndex] });
                    }
                }
            }
        }
        for (const argument of this.activeTarget.outputs || []) {
            for (let valueIndex = 0; valueIndex < (argument.value || []).length; valueIndex++) {
                if (outputs.has(argument.value[valueIndex])) {
                    affectedOutputs.push({ argument, valueIndex, oldValue: argument.value[valueIndex] });
                }
            }
        }
        const apply = () => {
            const current = nodes.indexOf(node);
            if (current >= 0) {
                nodes.splice(current, 1);
            }
            for (const target of affectedInputs) {
                const empty = this._graphEditTensorValue('', target.oldValue.type);
                target.argument.value[target.valueIndex] = empty;
                if (target.source && target.source.node && target.sourceIndex >= 0) {
                    target.source.node.input[target.sourceIndex] = { name: '' };
                }
            }
            for (const target of affectedOutputs) {
                target.argument.value[target.valueIndex] = this._graphEditTensorValue('', target.oldValue.type);
            }
            this._updateGraphEditPortStates();
        };
        const revert = () => {
            nodes.splice(Math.min(index, nodes.length), 0, node);
            for (const target of affectedInputs) {
                target.argument.value[target.valueIndex] = target.oldValue;
                if (target.source && target.source.node && target.sourceIndex >= 0) {
                    target.source.node.input[target.sourceIndex] = { name: target.oldValue.name };
                }
            }
            for (const target of affectedOutputs) {
                target.argument.value[target.valueIndex] = target.oldValue;
            }
            this._updateGraphEditPortStates();
        };
        const label = `delete node ${node.name || node.type.name}`;
        const command = {
            kind: 'delete-node',
            label,
            apply,
            revert,
            nodeIndex: node.source._graph_edit_index,
            nodeAddedId: node.source._graph_edit_added_id,
            refreshOnHistory: true,
            positionsBefore: this._graphEditPositionDelta(node)
        };
        apply();
        this._graphEdit.undo.push(command);
        this._graphEdit.redo = [];
        this._closeGraphEditNodeMenu();
        this._clearGraphEditDerivedState();
        // Graph interface additions rebuild or extend the renderer's root
        // entries. A later node removal can otherwise match an obsolete
        // renderer entry and leave the live SVG node behind while the model
        // and undo stack already contain the deletion.
        const interfaceChanged = this._graphEdit.undo.slice(0, -1).some((entry) =>
            entry.kind === 'add-input' || entry.kind === 'add-output' ||
            entry.kind === 'delete-input' || entry.kind === 'delete-output');
        command.visualRemoval = interfaceChanged ? null : this._removeGraphEditVisualNode(node);
        if (command.visualRemoval) {
            this._graphEdit.viewDirty = false;
            this._host.document.documentElement.classList.remove('onnx-graph-edit-view-dirty');
            this._updateGraphEditStatus(`Deleted ${node.name || node.type.name}.`);
        } else {
            this._deferGraphEditRefresh(`Deleted ${node.name || node.type.name}`);
            await this.redrawGraphEdit();
            this._updateGraphEditStatus(`Deleted ${node.name || node.type.name} · Graph view updated.`);
        }
    }

    _updateGraphEditConnectionActions() {
        const connection = this._graphEdit.connection;
        const target = connection ? connection.target : null;
        const label = this._element('graph-edit-connection-label');
        const disconnect = this._element('graph-edit-connection-disconnect');
        if (label && target) {
            label.textContent = `${target.oldValue.name} → ${this._graphEditTargetName(target)}`;
        }
        if (disconnect) {
            const editable = Boolean(target && target.oldValue && target.oldValue.name &&
                !target.oldValue.initializer);
            const optional = Boolean(target && target.argument.option === 'optional');
            disconnect.disabled = !editable;
            let title = 'This connection cannot be disconnected';
            if (editable) {
                title = optional ?
                    'Disconnect this optional ONNX input' :
                    'Temporarily disconnect this required input; save will be blocked until it is reconnected';
            }
            disconnect.title = title;
        }
    }

    _invalidateGraphEditIncrementalHistory() {
        const commands = [...this._graphEdit.undo, ...this._graphEdit.redo];
        for (const command of commands) {
            if (typeof command.invalidateIncremental === 'function') {
                command.invalidateIncremental();
            }
            // Node-removal records and captured edge objects belong to the
            // previous SVG tree. Future history operations must rebuild from
            // the edited model instead of restoring detached DOM nodes.
            command.visualRemoval = null;
            command.refreshOnHistory = true;
        }
    }

    async refresh(anchor, options = {}) {
        const snapshot = new Map();
        if (this._target) {
            for (const [key, entry] of this._target.nodes) {
                const label = entry.label;
                if (label && label.x !== undefined) {
                    snapshot.set(label.value || key, {
                        x: label.x, y: label.y,
                        width: label.width || 0, height: label.height || 0
                    });
                }
            }
        }
        const document = this._host.document;
        const container = document.getElementById('target');
        const zoom = this._target ? this._target._zoom : 1;
        const blocks = this._target ? this._target.blocks : null;
        if (blocks && blocks.size > 0 && this._path.length > 0) {
            this._path[0].state = Object.assign(this._path[0].state || {}, { blocks });
        }
        const origin = document.getElementById('origin');
        let previous = null;
        if (origin && this.activeTarget) {
            previous = origin.getScreenCTM();
            const oldChildren = Array.from(origin.children);
            const graph = this.activeTarget;
            const groups = graph.groups || false;
            const viewGraph = new view.Graph(this, groups);
            const state = this._path && this._path.length > 0 && this._path[0] ? this._path[0].state : null;
            if (state && state.blocks) {
                viewGraph.blocks = state.blocks;
            }
            viewGraph.add(graph, this.activeSignature);
            viewGraph.addTunnels();
            viewGraph.build(document, origin);
            const newChildren = Array.from(origin.children).filter((child) => !oldChildren.includes(child));
            for (const child of newChildren) {
                child.style.visibility = 'hidden';
            }
            await viewGraph.measure();
            const status = await viewGraph.layout(this._worker);
            if (status === '') {
                this._applyGraphEditPositions(viewGraph);
                viewGraph.alignFanoutSources();
                for (const child of oldChildren) {
                    if (child.parentNode === origin) {
                        origin.removeChild(child);
                    }
                }
                viewGraph.update();
                viewGraph.updateTunnels();
                for (const child of newChildren) {
                    child.style.removeProperty('visibility');
                }
                origin.setAttribute('transform', 'translate(0,0) scale(1)');
                document.getElementById('background').setAttribute('width', 0);
                document.getElementById('background').setAttribute('height', 0);
                viewGraph.restore(state);
                this.target = viewGraph;
            } else {
                for (const child of Array.from(origin.children)) {
                    if (!oldChildren.includes(child)) {
                        origin.removeChild(child);
                    }
                }
            }
        } else {
            await this.render(this.activeTarget, this.activeSignature);
        }
        this._invalidateGraphEditIncrementalHistory();
        this.show(null);
        if (this._target) {
            this._target.zoom = zoom;
            if (container && anchor) {
                const anchorNode = this._target.find(anchor.value);
                if (anchorNode instanceof grapher.Node && anchorNode.element) {
                    let newRect = anchorNode.element.getBoundingClientRect();
                    if (!anchor.node && anchorNode.definition && anchorNode.definition.element) {
                        newRect = anchorNode.definition.element.getBoundingClientRect();
                    }
                    if (container.scrollWidth > container.clientWidth) {
                        container.scrollLeft += (newRect.left - anchor.rect.left);
                    }
                    if (container.scrollHeight > container.clientHeight) {
                        container.scrollTop += (newRect.top - anchor.rect.top);
                    }
                }
                delete this._target._scrollLeft;
                delete this._target._scrollTop;
            }
            const current = origin ? origin.getScreenCTM() : null;
            const ox = previous && current ? (previous.e - current.e) / current.a : 0;
            const oy = previous && current ? (previous.f - current.f) / current.d : 0;
            const animateTransition = (snapshot) => {
                if (!this._target || snapshot.size === 0) {
                    return;
                }
                const duration = 300;
                let startTime = 0;
                const animations = [];
                for (const [key, entry] of this._target.nodes) {
                    const label = entry.label;
                    if (!label || !label.element) {
                        continue;
                    }
                    const modelKey = label.value || key;
                    const old = snapshot.get(modelKey);
                    const isCluster = this._target.children(key).length > 0;
                    if (old) {
                        if (isCluster) {
                            animations.push({
                                type: 'cluster', element: label.element, rect: label.rectangle,
                                fromX: old.x + ox, fromY: old.y + oy, toX: label.x, toY: label.y,
                                fromW: old.width, fromH: old.height, toW: label.width, toH: label.height
                            });
                        } else {
                            const fw = old.width;
                            const fh = old.height;
                            const tw = label.width;
                            const th = label.height;
                            animations.push({
                                type: 'node', element: label.element,
                                fromX: old.x - fw / 2 + ox, fromY: old.y - fh / 2 + oy,
                                toX: label.x - tw / 2, toY: label.y - th / 2
                            });
                        }
                    } else {
                        label.element.style.opacity = '0';
                        animations.push({ type: 'fadein', element: label.element });
                    }
                }
                for (const edge of this._target.edges.values()) {
                    const label = edge.label;
                    if (!label || !label.element) {
                        continue;
                    }
                    const fromNode = snapshot.get(label.from.value || edge.v);
                    const toNode = snapshot.get(label.to.value || edge.w);
                    if (fromNode && toNode) {
                        const newFrom = this._target.node(edge.v);
                        const newTo = this._target.node(edge.w);
                        if (newFrom && newTo) {
                            const dfx = fromNode.x - newFrom.label.x;
                            const dfy = fromNode.y - newFrom.label.y;
                            const dtx = toNode.x - newTo.label.x;
                            const dty = toNode.y - newTo.label.y;
                            const edgeOx = (dfx + dtx) / 2 + ox;
                            const edgeOy = (dfy + dty) / 2 + oy;
                            if (Math.abs(edgeOx) > 0.5 || Math.abs(edgeOy) > 0.5) {
                                const labelTransform = label.labelElement ? label.labelElement.getAttribute('transform') : null;
                                animations.push({
                                    type: 'edge', element: label.element,
                                    hitTest: label.hitTest, labelElement: label.labelElement,
                                    labelTransform,
                                    fromX: edgeOx, fromY: edgeOy
                                });
                            }
                        }
                    } else {
                        label.element.style.opacity = '0';
                        animations.push({ type: 'fadein', element: label.element });
                        if (label.hitTest) {
                            label.hitTest.style.opacity = '0';
                            animations.push({ type: 'fadein', element: label.hitTest });
                        }
                        if (label.labelElement) {
                            label.labelElement.style.opacity = '0';
                            animations.push({ type: 'fadein', element: label.labelElement });
                        }
                    }
                }
                for (const anim of animations) {
                    if (anim.type === 'node') {
                        anim.element.setAttribute('transform', `translate(${anim.fromX},${anim.fromY})`);
                    } else if (anim.type === 'cluster') {
                        anim.element.setAttribute('transform', `translate(${anim.fromX},${anim.fromY})`);
                        if (anim.rect) {
                            anim.rect.setAttribute('x', -anim.fromW / 2);
                            anim.rect.setAttribute('y', -anim.fromH / 2);
                            anim.rect.setAttribute('width', anim.fromW);
                            anim.rect.setAttribute('height', anim.fromH);
                        }
                    } else if (anim.type === 'edge') {
                        const t = `translate(${anim.fromX},${anim.fromY})`;
                        anim.element.setAttribute('transform', t);
                        if (anim.hitTest) {
                            anim.hitTest.setAttribute('transform', t);
                        }
                        if (anim.labelElement) {
                            anim.labelElement.setAttribute('transform', `translate(${anim.fromX},${anim.fromY}) ${anim.labelTransform || ''}`);
                        }
                    } else if (anim.type === 'fadein') {
                        anim.element.style.opacity = '0';
                    }
                }
                const tick = (now) => {
                    if (!startTime) {
                        startTime = now;
                    }
                    const elapsed = now - startTime;
                    const t = Math.min(elapsed / duration, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    for (const anim of animations) {
                        if (anim.type === 'node') {
                            const x = anim.fromX + (anim.toX - anim.fromX) * ease;
                            const y = anim.fromY + (anim.toY - anim.fromY) * ease;
                            anim.element.setAttribute('transform', `translate(${x},${y})`);
                        } else if (anim.type === 'cluster') {
                            const x = anim.fromX + (anim.toX - anim.fromX) * ease;
                            const y = anim.fromY + (anim.toY - anim.fromY) * ease;
                            anim.element.setAttribute('transform', `translate(${x},${y})`);
                            if (anim.rect) {
                                const w = anim.fromW + (anim.toW - anim.fromW) * ease;
                                const h = anim.fromH + (anim.toH - anim.fromH) * ease;
                                anim.rect.setAttribute('x', -w / 2);
                                anim.rect.setAttribute('y', -h / 2);
                                anim.rect.setAttribute('width', w);
                                anim.rect.setAttribute('height', h);
                            }
                        } else if (anim.type === 'edge') {
                            const x = anim.fromX * (1 - ease);
                            const y = anim.fromY * (1 - ease);
                            const tr = `translate(${x},${y})`;
                            anim.element.setAttribute('transform', tr);
                            if (anim.hitTest) {
                                anim.hitTest.setAttribute('transform', tr);
                            }
                            if (anim.labelElement) {
                                anim.labelElement.setAttribute('transform', `translate(${x},${y}) ${anim.labelTransform || ''}`);
                            }
                        } else if (anim.type === 'fadein') {
                            anim.element.style.opacity = String(ease);
                        }
                    }
                    if (t < 1) {
                        this._host.window.requestAnimationFrame(tick);
                    } else {
                        for (const anim of animations) {
                            if (anim.type === 'fadein') {
                                anim.element.style.removeProperty('opacity');
                            } else if (anim.type === 'edge') {
                                anim.element.removeAttribute('transform');
                                if (anim.hitTest) {
                                    anim.hitTest.removeAttribute('transform');
                                }
                                if (anim.labelElement) {
                                    if (anim.labelTransform) {
                                        anim.labelElement.setAttribute('transform', anim.labelTransform);
                                    } else {
                                        anim.labelElement.removeAttribute('transform');
                                    }
                                }
                            }
                        }
                    }
                };
                this._host.window.requestAnimationFrame(tick);
            };
            if (options.animate !== false) {
                animateTransition(snapshot);
            }
        }
    }

    async error(error, name, screen) {
        if (this._sidebar) {
            this._sidebar.close();
        }
        this.exception(error, false);
        const repository = this._host.environment('repository');
        const knowns = [
            { message: /^Invalid value identifier/, issue: '540' },
            { message: /^Cannot read property/, issue: '647' },
            { message: /^Duplicate value /, issue: '1364' },
            { message: /^EPERM: operation not permitted/, issue: '551' },
            { message: /^EACCES: permission denied/, issue: '504' },
            { message: /^Offset is outside the bounds of the DataView/, issue: '563' },
            { message: /^Invalid string length/, issue: '648' },
            { message: /^Unknown function /, issue: '546' },
            { message: /^Unsupported file content/, issue: '550' },
            { message: /^Unsupported Protocol Buffers content/, issue: '593' },
            { message: /^Unsupported Protocol Buffers text content/, issue: '594' },
            { message: /^Unsupported JSON content/, issue: '595' },
            { message: /^Unknown type name '__torch__\./, issue: '969' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto \(Unexpected end of file\)\./, issue: '1155' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto \(Cannot read properties of undefined \(reading 'ModelProto'\)\)\./, issue: '1156' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto/, issue: '549' }
        ];
        const known = knowns.find((known) => (!known.name || known.name === error.name) && error.message.match(known.message));
        const url = known && known.issue ? `${repository}/issues/${known.issue}` : `${repository}/issues`;
        const message = error.message;
        name = name || error.name;
        const report = !message.startsWith('Invalid file content.') && this.host.environment('packaged');
        await this._host.message(message, true, report ? 'Report' : 'OK');
        if (report) {
            this._host.openURL(url);
        }
        this.show(screen);
    }

    accept(file, size) {
        const attachment = this._model && typeof file === 'string' && file.toLowerCase().endsWith('.encodings');
        return attachment || this._modelFactoryService.accept(file, size);
    }

    async open(context) {
        this._sidebar.close();
        await this._timeout(2);
        try {
            const model = await this._modelFactoryService.open(context);
            this._resetGraphEdit();
            const format = [];
            if (model.format) {
                format.push(model.format);
            }
            if (model.producer) {
                format.push(`(${model.producer})`);
            }
            if (format.length > 0) {
                this._host.event('model_open', {
                    model_format: model.format || '',
                    model_producer: model.producer || ''
                });
            }
            await this._timeout(20);
            const path = [];
            const modules = Array.isArray(model.functions) ? model.modules.concat(model.functions) : model.modules;
            let target = modules.length > 0 ? modules[0] : null;
            for (const module of modules) {
                if (Array.isArray(module.nodes) && module.nodes.length > 0) {
                    target = module;
                    break;
                }
            }
            if (target) {
                const signature = Array.isArray(target.signatures) && target.signatures.length > 0 ? target.signatures[0] : null;
                path.push({ target, signature });
            }
            return await this._updateTarget(model, path);
        } catch (error) {
            error.context = !error.context && context && context.identifier ? context.identifier : error.context || '';
            throw error;
        }
    }

    async attach(context) {
        if (this._model) {
            const attachment = new metadata.Attachment();
            if (await attachment.open(context)) {
                attachment.bind(this._model);
                this._model.attachment = attachment;
                // Quantization badges can change measured node widths. Swap
                // the refreshed nodes and edges as one frame so a transition
                // cannot temporarily leave the nodes beside final edge paths.
                await this.refresh(null, { animate: false });
                return true;
            }
        }
        return false;
    }

    async _updateActiveTarget(stack) {
        this._sidebar.close();
        if (this._model) {
            this.show('welcome spinner');
            try {
                await this._updateTarget(this._model, stack);
            } catch (error) {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            }
        }
    }

    get activeTarget() {
        if (this._path.length > 0) {
            return this._path[0].target;
        }
        return null;
    }

    get activeSignature() {
        if (this._path.length > 0) {
            return this._path[0].signature;
        }
        return null;
    }

    async _updateTarget(model, path) {
        const lastModel = this._model;
        const lastPath = this._path;
        try {
            await this._updatePath(model, path);
            return this._model;
        } catch (error) {
            await this._updatePath(lastModel, lastPath);
            throw error;
        }
    }

    async _updatePath(model, stack) {
        this.model = model;
        this._path = stack;
        const status = await this.render(this.activeTarget, this.activeSignature);
        if (status === 'cancel') {
            this.model = null;
            this._path = [];
            this._activeTarget = null;
        }
        this.show(null);
        const path = this._element('toolbar-path');
        const back = this._element('toolbar-path-back-button');
        while (path.children.length > 1) {
            path.removeChild(path.lastElementChild);
        }
        if (status === '') {
            if (this._path.length <= 1) {
                back.style.opacity = 0;
            } else {
                back.style.opacity = 1;
                const last = this._path.length - 2;
                const count = Math.min(2, last);
                const document = this.host.document;
                if (count < last) {
                    const element = document.createElement('button');
                    element.setAttribute('class', 'toolbar-path-name-button');
                    element.innerHTML = '&hellip;';
                    path.appendChild(element);
                }
                for (let i = count; i >= 0; i--) {
                    const target = this._path[i].target;
                    const element = document.createElement('button');
                    element.setAttribute('class', 'toolbar-path-name-button');
                    element.addEventListener('click', async () => {
                        if (i > 0) {
                            this._path = this._path.slice(i);
                            await this._updateTarget(this._model, this._path);
                        } else {
                            await this.showTargetProperties(target);
                        }
                    });
                    let name = '';
                    if (target && target.identifier) {
                        name = target.identifier;
                    } else if (target && target.name) {
                        name = target.name;
                    }
                    if (name.length > 24) {
                        element.setAttribute('title', name);
                        const truncated = name.substring(name.length - 24, name.length);
                        element.innerHTML = '&hellip;';
                        const text = document.createTextNode(truncated);
                        element.appendChild(text);
                    } else {
                        element.removeAttribute('title');
                        if (name) {
                            element.textContent = name;
                        } else {
                            element.innerHTML = '&nbsp;';
                        }
                    }
                    path.appendChild(element);
                }
            }
            this._select.update(model, stack);
            const button = this._element('sidebar-target-button');
            if (stack.length > 0) {
                const type = stack[stack.length - 1].type || 'graph';
                const name = type.charAt(0).toUpperCase() + type.slice(1);
                button.setAttribute('title', `${name} Properties`);
                button.style.display = 'block';
            } else {
                button.style.display = 'none';
            }
        }
    }

    async pushTarget(graph, context) {
        if (graph && graph !== this.activeTarget && Array.isArray(graph.nodes)) {
            this._sidebar.close();
            if (context && this._path.length > 0) {
                this._path[0].state = { context, zoom: this._target.zoom, blocks: this._target.blocks };
            }
            const signature = Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures[0] : null;
            const entry = { target: graph, signature };
            const stack = [entry].concat(this._path);
            await this._updateTarget(this._model, stack);
        }
    }

    async popTarget() {
        if (this._path.length > 1) {
            this._sidebar.close();
            return await this._updateTarget(this._model, this._path.slice(1));
        }
        return null;
    }

    async render(target, signature) {
        this.target = null;
        const element = this._element('target');
        while (element.lastChild) {
            element.removeChild(element.lastChild);
        }
        let status = '';
        if (target) {
            const document = this._host.document;
            const graph = target;
            const groups = graph.groups || false;
            const nodes = graph.nodes;
            this._host.event('graph_view', {
                graph_node_count: nodes.length,
                graph_skip: 0
            });
            const viewGraph = new view.Graph(this, groups);
            const state = this._path && this._path.length > 0 && this._path[0] ? this._path[0].state : null;
            if (state && state.blocks) {
                viewGraph.blocks = state.blocks;
            }
            viewGraph.add(graph, signature);
            viewGraph.addTunnels();
            viewGraph.build(document);
            await viewGraph.measure();
            status = await viewGraph.layout(this._worker);
            if (status === '') {
                this._applyGraphEditPositions(viewGraph);
                viewGraph.alignFanoutSources();
                viewGraph.update();
                viewGraph.updateTunnels();
                viewGraph.restore(state);
                this.target = viewGraph;
            }
        }
        return status;
    }

    async export(file) {
        const window = this.host.window;
        const lastIndex = file.lastIndexOf('.');
        const extension = lastIndex === -1 ? 'png' : file.substring(lastIndex + 1).toLowerCase();
        if (this.activeTarget && (extension === 'png' || extension === 'svg')) {
            const canvas = this._element('canvas');
            const clone = canvas.cloneNode(true);
            const document = this._host.document;
            const applyStyleSheet = (element, name) => {
                let rules = [];
                for (const styleSheet of document.styleSheets) {
                    if (styleSheet && styleSheet.href && styleSheet.href.endsWith(`/${name}`)) {
                        rules = styleSheet.cssRules;
                        break;
                    }
                }
                const nodes = element.getElementsByTagName('*');
                for (const node of nodes) {
                    for (const rule of rules) {
                        if (node.matches(rule.selectorText)) {
                            for (const item of rule.style) {
                                node.style[item] = rule.style[item];
                            }
                        }
                    }
                }
            };
            applyStyleSheet(clone, 'grapher.css');
            clone.setAttribute('id', 'export');
            clone.removeAttribute('viewBox');
            clone.removeAttribute('width');
            clone.removeAttribute('height');
            clone.style.removeProperty('opacity');
            clone.style.removeProperty('display');
            clone.style.removeProperty('width');
            clone.style.removeProperty('height');
            const background = clone.querySelector('#background');
            clone.getElementById('edge-paths-hit-test').remove();
            const origin = clone.querySelector('#origin');
            origin.setAttribute('transform', 'translate(0,0) scale(1)');
            background.removeAttribute('width');
            background.removeAttribute('height');
            const parent = canvas.parentElement;
            parent.insertBefore(clone, canvas);
            const size = clone.getBBox();
            parent.removeChild(clone);
            parent.removeChild(canvas);
            parent.appendChild(canvas);
            const delta = (Math.min(size.width, size.height) / 2.0) * 0.1;
            const width = Math.ceil(delta + size.width + delta);
            const height = Math.ceil(delta + size.height + delta);
            origin.setAttribute('transform', `translate(${(delta - size.x)}, ${(delta - size.y)}) scale(1)`);
            clone.setAttribute('width', width);
            clone.setAttribute('height', height);
            background.setAttribute('width', width);
            background.setAttribute('height', height);
            background.setAttribute('fill', '#fff');
            const data = new window.XMLSerializer().serializeToString(clone);
            if (extension === 'svg') {
                const blob = new window.Blob([data], { type: 'image/svg' });
                await this._host.export(file, blob);
            }
            if (extension === 'png') {
                const blob = await new Promise((resolve, reject) => {
                    this.show('welcome spinner');
                    this.progress(0);
                    const image = new window.Image();
                    image.onload = async () => {
                        try {
                            let targetWidth = Math.ceil(width * 2);
                            let targetHeight = Math.ceil(height * 2);
                            let scale = 1;
                            if (targetWidth > 100000 || targetHeight > 100000) {
                                scale = Math.min(scale, 100000 / Math.max(targetWidth, targetHeight));
                            }
                            if (targetWidth * targetHeight * scale * scale > 500000000) {
                                scale = Math.min(scale, Math.sqrt(500000000 / (targetWidth * targetHeight)));
                            }
                            if (scale < 1) {
                                targetWidth = Math.floor(targetWidth * scale);
                                targetHeight = Math.floor(targetHeight * scale);
                            }
                            const drawScale = targetWidth / width;
                            const size = Math.min(targetWidth, 4096);
                            const encoder = new png.Encoder(window, targetWidth, targetHeight);
                            const canvas = this._host.document.createElement('canvas');
                            canvas.width = size;
                            canvas.height = 4096;
                            const context = canvas.getContext('2d');
                            for (let y = 0; y < targetHeight; y += 4096) {
                                const h = Math.min(4096, targetHeight - y);
                                const data = new Uint8Array(targetWidth * h * 4);
                                for (let x = 0; x < targetWidth; x += size) {
                                    const w = Math.min(size, targetWidth - x);
                                    context.setTransform(drawScale, 0, 0, drawScale, -x, -y);
                                    context.drawImage(image, 0, 0);
                                    const tileData = context.getImageData(0, 0, w, h);
                                    for (let row = 0; row < h; row++) {
                                        const src = row * w * 4;
                                        const dst = row * targetWidth * 4 + x * 4;
                                        data.set(tileData.data.subarray(src, src + w * 4), dst);
                                    }
                                }
                                /* eslint-disable-next-line no-await-in-loop */
                                await encoder.write(data, h);
                                this.progress((y + h) / targetHeight * 100);
                            }
                            const buffer = await encoder.toBuffer();
                            this.progress(0);
                            this.show('default');
                            resolve(new window.Blob([buffer], { type: 'image/png' }));
                        } catch (error) {
                            this.progress(0);
                            this.show('default');
                            reject(error);
                        }
                    };
                    image.onerror = (error) => {
                        this.progress(0);
                        this.show('default');
                        reject(error);
                    };
                    image.src = `data:image/svg+xml;base64,${this._host.window.btoa(unescape(encodeURIComponent(data)))}`;
                });
                await this._host.export(file, blob);
            }
        }
    }

    showModelProperties() {
        if (!this._model) {
            return;
        }
        try {
            const sidebar = new view.ModelSidebar(this, this.model);
            this._sidebar.open(sidebar, 'Model Properties');
        } catch (error) {
            this.error(error, 'Error showing model properties.', null);
        }
    }

    showModelStatistics() {
        if (this.model) {
            const sidebar = new view.StatisticsSidebar(this, this.model, this.activeTarget);
            this._sidebar.open(sidebar, 'Model Statistics');
        }
    }

    showTargetProperties(target) {
        if (this._sidebar.identifier === 'target' && !target) {
            this.showModelProperties();
            return;
        }
        target = target || this.activeTarget;
        if (!target) {
            return;
        }
        try {
            const sidebar = new view.TargetSidebar(this, target, this.activeSignature);
            sidebar.on('show-definition', async (/* sender, e */) => {
                await this.showDefinition(target);
            });
            sidebar.on('focus', (sender, value) => {
                this._target.focus([value]);
            });
            sidebar.on('blur', (sender, value) => {
                this._target.blur([value]);
            });
            sidebar.on('select', (sender, value) => {
                this._target.scrollTo(this._target.select([value], 'sidebar'));
            });
            sidebar.on('activate', (sender, value) => {
                this._target.scrollTo(this._target.activate(value, 'sidebar'));
            });
            sidebar.on('deactivate', () => {
                this._target.select(null);
            });
            let title = null;
            const type = target.type || 'graph';
            switch (type) {
                case 'graph':
                    title = 'Graph Properties';
                    break;
                case 'function':
                    title = 'Function Properties';
                    break;
                case 'weights':
                    title = 'Weights Properties';
                    break;
                default:
                    throw new view.Error(`Unsupported graph type '${type}'.`);
            }
            this._sidebar.open(sidebar, title);
        } catch (error) {
            this.error(error, 'Error showing target properties.', null);
        }
    }

    showNodeProperties(node, source) {
        if (node) {
            try {
                if (this._menu) {
                    this._menu.close();
                }
                const sidebar = new view.NodeSidebar(this, node);
                sidebar.on('show-definition', async (/* sender, e */) => {
                    await this.showDefinition(node.type);
                });
                sidebar.on('focus', (sender, value) => {
                    this._target.focus([value]);
                });
                sidebar.on('blur', (sender, value) => {
                    this._target.blur([value]);
                });
                sidebar.on('select', (sender, value) => {
                    this._target.scrollTo(this._target.select([value], 'sidebar'));
                });
                sidebar.on('activate', (sender, value) => {
                    this._target.scrollTo(this._target.activate(value, 'sidebar'));
                });
                this._sidebar.open(sidebar, 'Node Properties', source);
            } catch (error) {
                this.error(error, 'Error showing node properties.', null);
            }
        }
    }

    showConnectionProperties(value, from, to, source) {
        try {
            if (this._menu) {
                this._menu.close();
            }
            const sidebar = new view.ConnectionSidebar(this, value, from, to);
            sidebar.on('focus', (sender, value) => {
                this._target.focus([value]);
            });
            sidebar.on('blur', (sender, value) => {
                this._target.blur([value]);
            });
            sidebar.on('select', (sender, value) => {
                this._target.scrollTo(this._target.select([value], 'sidebar'));
            });
            sidebar.on('activate', (sender, value) => {
                this._target.scrollTo(this._target.activate(value, 'sidebar'));
            });
            this._sidebar.open(sidebar, 'Connection Properties', source);
        } catch (error) {
            this.error(error, 'Error showing connection properties.', null);
        }
    }

    showTensorProperties(value, source, options) {
        try {
            if (this._menu) {
                this._menu.close();
            }
            options = options || {};
            const sidebar = new view.TensorSidebar(this, value, options.quantizationTitle);
            sidebar.on('focus', (sender, value) => {
                this._target.focus([value]);
            });
            sidebar.on('blur', () => {
                this._target.blur(null);
            });
            sidebar.on('select', (sender, value) => {
                this._target.scrollTo(this._target.select([value], 'sidebar'));
            });
            sidebar.on('activate', (sender, value) => {
                this._target.scrollTo(this._target.activate(value, 'sidebar'));
            });
            this._sidebar.open(sidebar, options.title || 'Tensor Properties', source);
        } catch (error) {
            this.error(error, 'Error showing tensor properties.', null);
        }
    }

    exception(error, fatal) {
        if (error && !error.context && this._model && this._model.identifier) {
            error.context = this._model.identifier;
        }
        this._host.exception(error, fatal);
    }

    async showDefinition(type) {
        if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
            if (type.nodes && type.nodes.length > 0) {
                await this.pushTarget(type);
            }
            if (type.type !== 'weights') {
                const sidebar = new view.DocumentationSidebar(this, type);
                sidebar.on('navigate', (sender, e) => {
                    this._host.openURL(e.link);
                });
                const title = type.type === 'function' ? 'Function Documentation' : 'Documentation';
                this._sidebar.open(sidebar, title, 'sidebar');
            }
        }
    }

    about() {
        this._host.document.getElementById('version').innerText = this._host.version;
        const handler = () => {
            this._host.window.removeEventListener('keydown', handler);
            this._host.document.body.removeEventListener('click', handler);
            this._host.document.body.classList.remove('about');
        };
        this._host.window.addEventListener('keydown', handler);
        this._host.document.body.addEventListener('click', handler);
        this._host.document.body.classList.add('about');
    }
};

view.Menu = class {

    constructor(host) {
        this.items = [];
        this._darwin = host.environment('platform') === 'darwin';
        this._document = host.document;
        this._window = host.window;
        this._stack = [];
        this._root = [];
        this._buttons = [];
        this._accelerators = new Map();
        this._keyCodes = new Map([
            ['Backspace', 0x08], ['Enter', 0x0D], ['Escape', 0x1B],
            ['Left', 0x25], ['Up', 0x26], ['Right', 0x27], ['Down', 0x28],
            ['F5', 0x74], ['F11', 0x7a]
        ]);
        this._symbols = new Map([
            ['Backspace', '&#x232B;'], ['Enter', '&#x23ce;'],
            ['Up', '&#x2191;'], ['Down', '&#x2193;'],
        ]);
        this._keydown = (e) => {
            this._alt = false;
            const code = e.keyCode | (e.altKey ? 0x0200 : 0) | (e.shiftKey ? 0x0100 : 0);
            const modifier = (e.ctrlKey ? 0x0400 : 0) | (e.metaKey ? 0x0800 : 0);
            if ((code | modifier) === 0x0212) { // Alt
                this._alt = true;
            } else {
                const action =
                    this._accelerators.get(code | modifier) ||
                    this._accelerators.get(code | ((e.ctrlKey && !this._darwin) || (e.metaKey && this._darwin) ? 0x1000 : 0));
                if (action && this._execute(action)) {
                    e.preventDefault();
                } else {
                    const item = this._mnemonic(code | modifier);
                    if (item && this._activate(item)) {
                        e.preventDefault();
                    }
                }
            }
        };
        this._keyup = (e) => {
            if (e.keyCode === 0x0012 && this._alt) { // Alt
                if (this._stack.length === 0) {
                    if (this.open()) {
                        e.preventDefault();
                    }
                } else if (this._stack.length === 1) {
                    if (this.close()) {
                        e.preventDefault();
                    }
                } else {
                    this._stack = [this];
                    if (this._root.length > 1) {
                        this._root =  [this];
                        this._rebuild();
                    }
                    this._update();
                    e.preventDefault();
                }
            }
            this._alt = false;
        };
        this._next = () => {
            const button = this._element.ownerDocument.activeElement;
            const index = this._buttons.indexOf(button);
            if (index !== -1 && index < this._buttons.length - 1) {
                const next = this._buttons[index + 1];
                next.focus();
            }
        };
        this._previous = () => {
            const button = this._element.ownerDocument.activeElement;
            const index = this._buttons.indexOf(button);
            if (index > 0) {
                const next = this._buttons[index - 1];
                next.focus();
            }
        };
        this._push = () => {
            const button = this._element.ownerDocument.activeElement;
            if (button && button.getAttribute('data-type') === 'group') {
                button.click();
            }
        };
        this._pop = () => {
            if (this._stack.length > 1) {
                this._deactivate();
            }
        };
        this._exit = () => {
            this._deactivate();
            if (this._stack.length === 0) {
                this.close();
            }
        };
        host.window.addEventListener('keydown', this._keydown);
        host.window.addEventListener('keyup', this._keyup);
    }

    attach(element, button) {
        this._element = element;
        button.addEventListener('click', (e) => {
            this.toggle();
            e.preventDefault();
        });
    }

    add(value) {
        const item = new view.Menu.Command(value);
        this.register(item, item.accelerator);
    }

    group(label) {
        const item = new view.Menu.Group(this, label);
        item.identifier = `menu-item-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.register(item.accelerator);
        return item;
    }

    toggle() {
        if (this._element.style.opacity >= 1) {
            this.close();
        } else {
            this._root = [this];
            this._stack = [this];
            this.open();
        }
    }

    open() {
        if (this._element) {
            if (this._stack.length === 0) {
                this.toggle();
                this._stack = [this];
            }
            this._rebuild();
            this._update();
            this.register(this._exit, 'Escape');
            this.register(this._previous, 'Up');
            this.register(this._next, 'Down');
            this.register(this._pop, 'Left');
            this.register(this._push, 'Right');
        }
    }

    close() {
        if (this._element) {
            this.unregister(this._exit);
            this.unregister(this._previous);
            this.unregister(this._next);
            this.unregister(this._pop);
            this.unregister(this._push);
            this._element.style.opacity = 0;
            this._element.style.left = '-17em';
            const button = this._element.ownerDocument.activeElement;
            if (this._buttons.indexOf(button) > 0) {
                button.blur();
            }
            while (this._root.length > 1) {
                this._deactivate();
            }
            this._stack = [];
        }
    }

    register(action, accelerator) {
        let shortcut = '';
        if (accelerator) {
            let shift = false;
            let alt = false;
            let ctrl = false;
            let cmd = false;
            let cmdOrCtrl = false;
            let key = '';
            for (const part of accelerator.split('+')) {
                switch (part) {
                    case 'CmdOrCtrl': cmdOrCtrl = true; break;
                    case 'Cmd': cmd = true; break;
                    case 'Ctrl': ctrl = true; break;
                    case 'Alt': alt = true; break;
                    case 'Shift': shift = true; break;
                    default: key = part; break;
                }
            }
            if (key !== '') {
                if (this._darwin) {
                    shortcut += ctrl ? '&#x2303' : '';
                    shortcut += alt ? '&#x2325;' : '';
                    shortcut += shift ? '&#x21e7;' : '';
                    shortcut += cmdOrCtrl || cmd ? '&#x2318;' : '';
                    shortcut += this._symbols.has(key) ? this._symbols.get(key) : key;
                } else {
                    shortcut += cmdOrCtrl || ctrl ? 'Ctrl+' : '';
                    shortcut += alt ? 'Alt+' : '';
                    shortcut += shift ? 'Shift+' : '';
                    shortcut += key;
                }
                let code = (cmdOrCtrl ? 0x1000 : 0) | (cmd ? 0x0800 : 0) | (ctrl ? 0x0400 : 0) | (alt ? 0x0200 : 0) | (shift ? 0x0100 : 0);
                code |= this._keyCodes.has(key) ? this._keyCodes.get(key) : key.charCodeAt(0);
                this._accelerators.set(code, action);
            }
        }
        return shortcut;
    }

    unregister(action) {
        this._accelerators = new Map(Array.from(this._accelerators.entries()).filter(([, value]) => value !== action));
    }

    _execute(action) {
        const window = this._window;
        if (typeof action === 'function') {
            action();
            return true;
        }
        switch (action ? action.type : null) {
            case 'group': {
                while (this._stack.length > this._root.length) {
                    this._stack.pop();
                }
                this._root.push({ items: [action] });
                this._stack.push(action);
                this._rebuild();
                this._update();
                return true;
            }
            case 'command': {
                this.close();
                window.setTimeout(() => action.execute(), 10);
                return true;
            }
            default: {
                return false;
            }
        }
    }

    _mnemonic(code) {
        const key = /[a-zA-Z0-9]/.test(String.fromCharCode(code & 0x00FF));
        const modifier = (code & 0xFF00) !== 0;
        const alt = (code & 0xFF00) === 0x0200;
        if (alt && key) {
            this.open();
        }
        if (this._stack.length > 0 && key && (alt || !modifier)) {
            const key = String.fromCharCode(code & 0x00FF);
            const group = this._stack.length > 0 ? this._stack[this._stack.length - 1] : this;
            const item = group.items.find((item) => key === item.mnemonic && (item.type === 'group' || item.type === 'command') && item.enabled);
            if (item) {
                return item;
            }
        }
        return null;
    }

    _activate(item) {
        switch (item ? item.type : null) {
            case 'group': {
                this._stack.push(item);
                this._rebuild();
                this._update();
                return true;
            }
            case 'command': {
                return this._execute(item);
            }
            default: {
                return false;
            }
        }
    }

    _deactivate() {
        if (this._root.length > 1) {
            this._root.pop();
            const group = this._stack.pop();
            this._rebuild();
            this._update();
            if (group) {
                const button = this._buttons.find((button) => button.getAttribute('id') === group.identifier);
                if (button) {
                    button.focus();
                }
            }
        } else if (this._stack.length > 0) {
            this._stack.pop();
            this._update();
        }
    }

    _label(item, mnemonic) {
        delete item.mnemonic;
        const value = item.label;
        if (value) {
            const index = value.indexOf('&');
            if (index !== -1) {
                if (mnemonic) {
                    item.mnemonic = value[index + 1].toUpperCase();
                    return `${value.substring(0, index)}<u>${value[index + 1]}</u>${value.substring(index + 2)}`;
                }
                return value.substring(0, index) + value.substring(index + 1);
            }
        }
        return value || '';
    }

    _rebuild() {
        this._element.replaceChildren();
        const root = this._root[this._root.length - 1];
        for (const group of root.items) {
            const container = this._document.createElement('div');
            container.setAttribute('id', group.identifier);
            container.setAttribute('class', 'menu-group');
            container.innerHTML = "<div class='menu-group-header'></div>";
            for (const item of group.items) {
                switch (item.type) {
                    case 'group':
                    case 'command': {
                        const button = this._document.createElement('button');
                        button.setAttribute('class', 'menu-command');
                        button.setAttribute('id', item.identifier);
                        button.setAttribute('data-type', item.type);
                        button.addEventListener('mouseenter', () => button.focus());
                        button.addEventListener('click', () => this._execute(item));
                        const accelerator = this._document.createElement('span');
                        accelerator.setAttribute('class', 'menu-shortcut');
                        if (item.type === 'group') {
                            accelerator.innerHTML = '&#10095;';
                        } else if (item.shortcut) {
                            accelerator.innerHTML = item.shortcut;
                        }
                        button.appendChild(accelerator);
                        const content = this._document.createElement('span');
                        content.setAttribute('class', 'menu-label');
                        button.appendChild(content);
                        container.appendChild(button);
                        break;
                    }
                    case 'separator': {
                        const element = this._document.createElement('div');
                        element.setAttribute('class', 'menu-separator');
                        element.setAttribute('id', item.identifier);
                        container.appendChild(element);
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            this._element.appendChild(container);
        }
        this._element.style.opacity = 1.0;
        this._element.style.left = '0px';
        if (this._root.length > 1) {
            this._element.style.width = 'auto';
            this._element.style.maxWidth = '60%';
        } else {
            this._element.style.removeProperty('width');
            this._element.style.maxWidth = 'auto';
        }
    }

    _update() {
        this._buttons = [];
        const selected = this._stack.length > 0 ? this._stack[this._stack.length - 1] : null;
        const root = this._root[this._root.length - 1];
        for (const group of root.items) {
            let visible = false;
            let block = false;
            const active = this._stack.length <= 1 || this._stack[1] === group;
            const container = this._document.getElementById(group.identifier);
            container.childNodes[0].innerHTML = this._label(group, this === selected);
            for (const item of group.items) {
                switch (item.type) {
                    case 'group':
                    case 'command': {
                        const label = this._label(item, group === selected);
                        const button = this._document.getElementById(item.identifier);
                        button.childNodes[1].innerHTML = label;
                        if (item.enabled) {
                            button.removeAttribute('disabled');
                            button.style.display = 'block';
                            visible = true;
                            block = true;
                            if (active) {
                                this._buttons.push(button);
                            }
                        } else {
                            button.setAttribute('disabled', '');
                            button.style.display = 'none';
                        }
                        break;
                    }
                    case 'separator': {
                        const element = this._document.getElementById(item.identifier);
                        element.style.display = block ? 'block' : 'none';
                        block = false;
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            for (let i = group.items.length - 1; i >= 0; i--) {
                const item = group.items[i];
                if ((item.type === 'group' || item.type === 'command') && item.enabled) {
                    break;
                } else if (item.type === 'separator') {
                    const element = this._document.getElementById(item.identifier);
                    element.style.display = 'none';
                }
            }
            if (!visible) {
                container.style.display = 'none';
            }
            container.style.opacity = active ? 1 : 0;
        }
        const button = this._element.ownerDocument.activeElement;
        const index = this._buttons.indexOf(button);
        if (index === -1 && this._buttons.length > 0) {
            this._buttons[0].focus();
        }
    }
};

view.Menu.Group = class {

    constructor(parent, label) {
        this.type = 'group';
        this.parent = parent;
        this.label = label;
        this.items = [];
    }

    get enabled() {
        return this.items.some((item) => item.enabled);
    }

    add(value) {
        const item = Object.keys(value).length > 0 ? new view.Menu.Command(value) : new view.Menu.Separator();
        item.identifier = `${this.identifier}-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.parent.register(item, item.accelerator);
    }

    group(label) {
        const item = new view.Menu.Group(this, label);
        item.identifier = `${this.identifier}-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.parent.register(item, item.accelerator);
        return item;
    }

    clear() {
        for (const item of this.items) {
            if (item.clear) {
                item.clear();
            }
            this.parent.unregister(item);
        }
        this.items = [];
    }

    register(item, accelerator) {
        return this.parent.register(item, accelerator);
    }

    unregister(item) {
        this.parent.unregister(item);
    }
};

view.Menu.Command = class {

    constructor(item) {
        this.type = 'command';
        this.accelerator = item.accelerator;
        this._label = item.label;
        this._enabled = item.enabled;
        this._execute = item.execute;
    }

    get label() {
        return typeof this._label === 'function' ? this._label() : this._label;
    }

    get enabled() {
        return this._enabled ? this._enabled() : true;
    }

    execute() {
        if (this._execute && this.enabled) {
            this._execute();
        }
    }
};

view.Menu.Separator = class {

    constructor() {
        this.type = 'separator';
        this.enabled = false;
    }
};

view.Worker = class {

    constructor(host) {
        this._host = host;
        this._timeout = -1;
        if (this._host.type !== 'Electron') {
            this._create();
        }
    }

    async request(message, delay, notification) {
        if (this._resolve) {
            const resolve = this._resolve;
            resolve({ type: 'terminate' });
            delete this._resolve;
            delete this._reject;
            this.cancel(true);
        } else {
            this.cancel(false);
        }
        return new Promise((resolve, reject) => {
            this._resolve = resolve;
            this._reject = reject;
            this._create();
            this._worker.postMessage(message);
            const window = this._host.window;
            this._timeout = window.setTimeout(async () => {
                await this._host.message(notification, null, 'Cancel');
                this.cancel(true);
                delete this._resolve;
                delete this._reject;
                resolve({ type: 'cancel' });
            }, delay);
        });
    }

    _create() {
        if (!this._worker) {
            this._worker = this._host.worker('./worker');
            this._worker.addEventListener('message', (e) => {
                this.cancel(false);
                const message = e.data;
                const resolve = this._resolve;
                const reject = this._reject;
                delete this._resolve;
                delete this._reject;
                if (reject && message.type === 'error') {
                    const error = new Error(`Worker: ${message.message}`);
                    reject(error);
                } else if (resolve) {
                    resolve(message);
                }
            });
            this._worker.addEventListener('error', (e) => {
                this.cancel(true);
                const reject = this._reject;
                delete this._resolve;
                delete this._reject;
                if (reject) {
                    reject(new Error(`Unknown worker error type '${e.type}'.`));
                }
            });
        }
    }

    cancel(terminate) {
        if (this._worker && terminate) {
            this._worker.terminate();
            this._worker = null;
        }
        if (this._timeout !== -1) {
            this._host.window.clearTimeout(this._timeout);
            this._timeout = -1;
            this._host.message();
        }
    }
};

view.Graph = class extends grapher.Graph {

    constructor(view, compound) {
        super(compound);
        this.view = view;
        this.counter = 0;
        this._nodeKey = 0;
        this._values = new Map();
        this._tensors = new Map();
        this._table = new Map();
        this._selection = new Set();
        this.blocks = new Set();
        this._zoom = 1;
        this._listeners = {};
        this._edgeBundles = new Map();
    }

    on(event, callback) {
        this._listeners[event] = this._listeners[event] || [];
        this._listeners[event].push(callback);
    }

    off(event, callback) {
        if (this._listeners[event]) {
            this._listeners[event] = this._listeners[event].filter((c) => c !== callback);
        }
    }

    emit(event, data) {
        if (this._listeners[event]) {
            for (const callback of this._listeners[event]) {
                callback(this, data);
            }
        }
    }

    get model() {
        return this.view.model;
    }

    get host() {
        return this.view.host;
    }

    get options() {
        return this.view.options;
    }

    get values() {
        return this._values;
    }

    get selection() {
        return this._selection;
    }

    alignFanoutSources() {
        const groups = new Map();
        for (const entry of this.edges.values()) {
            const edge = entry && entry.label;
            if (!edge) {
                continue;
            }
            edge.sourceAnchor = null;
            const value = edge.value && edge.value.value;
            if (!value || edge.hidden || edge._tunnel) {
                continue;
            }
            if (!groups.has(value)) {
                groups.set(value, []);
            }
            groups.get(value).push(edge);
        }
        const horizontal = this.options.direction === 'horizontal';
        const anchor = horizontal ? { x: 1, y: 0.5 } : { x: 0.5, y: 1 };
        for (const edges of groups.values()) {
            if (edges.length > 1) {
                for (const edge of edges) {
                    edge.sourceAnchor = anchor;
                }
            }
        }
        delete this._graphEditPorts;
    }

    _graphEditPortIndex() {
        if (!this._graphEditPorts) {
            const inputs = new WeakMap();
            const outputs = new WeakMap();
            const graphOutputs = new WeakMap();
            const intersect = (node, point) => {
                const dx = point.x - node.x;
                const dy = point.y - node.y;
                let h = node.height / 2;
                let w = node.width / 2;
                if (Math.abs(dy) * w > Math.abs(dx) * h) {
                    h = dy < 0 ? -h : h;
                    return {
                        x: (dy === 0 ? 0 : h * dx / dy) + (node.width / 2),
                        y: h + (node.height / 2)
                    };
                }
                w = dx < 0 ? -w : w;
                return {
                    x: w + (node.width / 2),
                    y: (dx === 0 ? 0 : w * dy / dx) + (node.height / 2)
                };
            };
            const set = (map, key, index, point) => {
                if (!map.has(key)) {
                    map.set(key, new Map());
                }
                map.get(key).set(index, point);
            };
            const local = (node, point) => ({
                x: point.x - node.x + node.width / 2,
                y: point.y - node.y + node.height / 2
            });
            for (const entry of this.edges.values()) {
                const edge = entry.label;
                if (!edge || !Array.isArray(edge.points) || edge.points.length < 3) {
                    continue;
                }
                if (edge.element && edge.element.classList.contains('graph-edit-edge-disconnected')) {
                    continue;
                }
                const route = Array.isArray(edge.graphEditRoute) && edge.graphEditRoute.length > 1 ?
                    edge.graphEditRoute : null;
                const target = edge.graphEditTarget;
                if (target && target.argument) {
                    let point = intersect(edge.to, edge.points[edge.points.length - 2]);
                    if (edge.targetPoint) {
                        point = local(edge.to, edge.targetPoint);
                    } else if (route) {
                        point = local(edge.to, route[route.length - 1]);
                    }
                    if (target.graphOutput === true) {
                        set(graphOutputs, target.argument, target.valueIndex, point);
                    } else {
                        set(inputs, target.argument, target.valueIndex, point);
                    }
                }
                const value = edge.value && edge.value.value;
                if (value && !edge.hidden) {
                    let point = intersect(edge.from, edge.points[1]);
                    if (edge.sourceAnchor) {
                        point = {
                            x: edge.sourceAnchor.x * edge.from.width,
                            y: edge.sourceAnchor.y * edge.from.height
                        };
                    } else if (edge.sourcePoint) {
                        point = local(edge.from, edge.sourcePoint);
                    } else if (route) {
                        point = local(edge.from, route[0]);
                    }
                    if (!outputs.has(value)) {
                        outputs.set(value, []);
                    }
                    const positions = outputs.get(value);
                    if (!positions.some((position) => Math.hypot(position.x - point.x, position.y - point.y) < 0.75)) {
                        positions.push(point);
                    }
                }
            }
            this._graphEditPorts = { inputs, outputs, graphOutputs };
        }
        return this._graphEditPorts;
    }

    graphEditInputPortPosition(argument, valueIndex) {
        const positions = this._graphEditPortIndex().inputs.get(argument);
        return positions ? positions.get(valueIndex) || null : null;
    }

    graphEditOutputPortPositions(value) {
        return this._graphEditPortIndex().outputs.get(value) || [];
    }

    graphEditOutputPortCount(value) {
        return value ? 1 : 0;
    }

    graphEditGraphOutputPortPosition(argument, valueIndex) {
        const positions = this._graphEditPortIndex().graphOutputs.get(argument);
        return positions ? positions.get(valueIndex) || null : null;
    }

    createNode(node) {
        const obj = new view.Node(this, node);
        obj.name = (this._nodeKey++).toString();
        this._table.set(node, obj);
        return obj;
    }

    createGraph(graph, type) {
        const obj = new view.Node(this, graph, type || 'graph');
        obj.name = (this._nodeKey++).toString();
        this._table.set(graph, obj);
        return obj;
    }

    createInput(input) {
        const obj = new view.Input(this, input);
        obj.name = (this._nodeKey++).toString();
        this._table.set(input, obj);
        return obj;
    }

    createOutput(output) {
        const obj = new view.Output(this, output);
        obj.name = (this._nodeKey++).toString();
        this._table.set(output, obj);
        return obj;
    }

    createValue(value) {
        const key = value && value.name && !value.initializer ? value.name : value;
        if (this._values.has(key)) {
            // duplicate argument name
            const obj = this._values.get(key);
            this._table.set(value, obj);
        } else {
            const obj = new view.Value(this, value);
            this._values.set(key, obj);
            this._table.set(value, obj);
        }
        return this._values.get(key);
    }

    createArgument(value) {
        if (Array.isArray(value.value) && value.value.length === 1 && value.value[0].initializer) {
            if (!this._tensors.has(value)) {
                const obj = new view.Argument(this, value);
                this._tensors.set(value, obj);
                this._table.set(value, obj);
            }
            return this._tensors.get(value);
        }
        return null;
    }

    edgeBundle(value) {
        return this._edgeBundles.get(value) || null;
    }

    _prepareEdgeBundles() {
        this._edgeBundles.clear();
        const nodeType = (node) => {
            const type = node && node.value ? node.value.type : null;
            if (!type) {
                return '';
            }
            return typeof type === 'string' ? type : type.name;
        };
        const dimensions = (value) => {
            const shape = value && value.value && value.value.type ? value.value.type.shape : null;
            return shape && Array.isArray(shape.dimensions) ? shape.dimensions : [];
        };
        const dimensionText = (dimension) => dimension !== null && dimension !== undefined &&
            dimension !== -1 && dimension !== -1n ? dimension.toString() : '?';
        const typeSignature = (value) => {
            const type = value && value.value ? value.value.type : null;
            const dataType = type && type.dataType ? type.dataType : '';
            return `${dataType}:${dimensions(value).map((dimension) => dimensionText(dimension)).join('x')}`;
        };
        for (const entry of this.nodes.values()) {
            const from = entry.label;
            if (!(from instanceof view.Node) || nodeType(from) !== 'Split') {
                continue;
            }
            for (const output of from.value.outputs || []) {
                const modelValues = (output.value || []).filter((value) => value && value.name && !value.initializer);
                if (modelValues.length < 3) {
                    continue;
                }
                const values = modelValues.map((value) => this._values.get(value.name)).filter((value) => value);
                if (values.length !== modelValues.length || values.some((value) => value.to.length === 0)) {
                    continue;
                }
                const to = values[0].to[0];
                if (!(to instanceof view.Node) || nodeType(to) !== 'Concat' ||
                    values.some((value) => value.to.some((consumer) => consumer !== to))) {
                    continue;
                }
                const names = new Set(modelValues.map((value) => value.name));
                const input = (to.value.inputs || []).find((argument) => {
                    const inputValues = (argument.value || []).filter((value) => value && !value.initializer);
                    const inputNames = new Set(inputValues.map((value) => value.name));
                    return inputValues.length >= modelValues.length && inputNames.size === names.size &&
                        inputValues.every((value) => names.has(value.name));
                });
                if (!input || new Set(values.map((value) => typeSignature(value))).size !== 1) {
                    continue;
                }
                const connectionCount = values.reduce((count, value) =>
                    count + value.to.filter((consumer) => consumer === to).length, 0);
                const members = [];
                for (const value of values) {
                    for (let toIndex = 0; toIndex < value.to.length; toIndex++) {
                        if (value.to[toIndex] === to) {
                            members.push({ value, toIndex });
                        }
                    }
                }
                const inputEntries = [];
                const bundledNames = new Set(values.map((value) => value.value.name));
                for (let valueIndex = 0; valueIndex < input.value.length; valueIndex++) {
                    const value = input.value[valueIndex];
                    if (value && bundledNames.has(value.name)) {
                        inputEntries.push({ argument: input, valueIndex, value });
                    }
                }
                const shape = dimensions(values[0]).map((dimension) => dimensionText(dimension)).join('\u00D7');
                const labels = new Set();
                for (const value of values) {
                    const encoding = this.model.attachment.quantization.precision(value.value);
                    if (encoding && encoding.label) {
                        labels.add(encoding.label);
                    }
                }
                const parts = [`\u00D7${connectionCount}`];
                if (shape) {
                    parts.push(shape);
                }
                if (labels.size > 0) {
                    parts.push(Array.from(labels).sort().join('/'));
                }
                const bundle = {
                    from,
                    to,
                    input,
                    output,
                    values,
                    members,
                    inputEntries,
                    connectionCount,
                    representativeIndex: Math.floor((connectionCount - 1) / 2),
                    label: parts.join(' \u00B7 '),
                    edge: null
                };
                for (const value of values) {
                    this._edgeBundles.set(value, bundle);
                }
            }
        }
    }

    find(value) {
        if (this._table.has(value)) {
            return this._table.get(value);
        }
        for (const obj of this._table.values()) {
            if (obj instanceof grapher.Node) {
                for (const block of obj.blocks) {
                    if (block instanceof view.Block) {
                        const found = block.target.find(value);
                        if (found) {
                            return found;
                        }
                    }
                }
            }
        }
        return null;
    }

    add(graph, signature) {
        this.target = graph;
        this.identifier = this.model.identifier;
        this.identifier += graph && graph.name ? `.${graph.name.replace(/\/|\\/g, '.')}` : '';
        const clusters = new Set();
        const clusterParentMap = new Map();
        const groups = graph.groups;
        if (groups) {
            for (const node of graph.nodes) {
                if (node.group) {
                    const path = node.group.split('/');
                    while (path.length > 0) {
                        const name = path.join('/');
                        path.pop();
                        clusterParentMap.set(name, path.join('/'));
                    }
                }
            }
        }
        const inputs = signature ? signature.inputs : graph.inputs;
        const outputs = signature ? signature.outputs : graph.outputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                if (argument.visible !== false) {
                    const viewInput = this.createInput(argument);
                    this.setNode(viewInput);
                    for (const value of argument.value) {
                        this.createValue(value).from = viewInput;
                    }
                }
            }
        }
        for (const node of graph.nodes) {
            const viewNode = this.createNode(node);
            this.setNode(viewNode);
            let outputs = node.outputs;
            if (node.chain && node.chain.length > 0) {
                const chainOutputs = node.chain[node.chain.length - 1].outputs;
                if (chainOutputs.length > 0) {
                    outputs = chainOutputs;
                }
            }
            if (Array.isArray(outputs)) {
                for (const argument of outputs) {
                    for (const value of argument.value) {
                        if (!value) {
                            throw new view.Error('Invalid null argument.');
                        }
                        if (value.name !== '') {
                            this.createValue(value).from = viewNode;
                        }
                    }
                }
            }
            if (Array.isArray(node.controlDependencies) && node.controlDependencies.length > 0) {
                for (const value of node.controlDependencies) {
                    this.createValue(value).controlDependency(viewNode);
                }
            }
            const createCluster = (name) => {
                if (!clusters.has(name)) {
                    this.setNode({ name, rx: 5, ry: 5 });
                    clusters.add(name);
                    const parent = clusterParentMap.get(name);
                    if (parent) {
                        createCluster(parent);
                        this.setParent(name, parent);
                    }
                }
            };
            if (groups) {
                let groupName = node.group;
                if (groupName && groupName.length > 0) {
                    if (!clusterParentMap.has(groupName)) {
                        const lastIndex = groupName.lastIndexOf('/');
                        if (lastIndex === -1) {
                            groupName = null;
                        } else {
                            groupName = groupName.substring(0, lastIndex);
                            if (!clusterParentMap.has(groupName)) {
                                groupName = null;
                            }
                        }
                    }
                    if (groupName) {
                        createCluster(`${groupName}\ngroup`);
                        this.setParent(viewNode.name, `${groupName}\ngroup`);
                    }
                }
            }
        }
        if (Array.isArray(outputs)) {
            for (const argument of outputs) {
                if (argument.visible !== false) {
                    const viewOutput = this.createOutput(argument);
                    this.setNode(viewOutput);
                    if (Array.isArray(argument.value)) {
                        for (const value of argument.value) {
                            this.createValue(value).to.push(viewOutput);
                        }
                    }
                }
            }
        }
    }

    addTunnels() {
        this._tunnels = [];
        const subgraphOuterRefs = (graph) => {
            const produced = new Set();
            if (Array.isArray(graph.inputs)) {
                for (const arg of graph.inputs) {
                    if (!Array.isArray(arg.value)) {
                        continue;
                    }
                    for (const val of arg.value) {
                        if (val.name) {
                            produced.add(val.name);
                        }
                    }
                }
            }
            for (const node of (graph.nodes || [])) {
                for (const arg of (node.outputs || [])) {
                    if (!Array.isArray(arg.value)) {
                        continue;
                    }
                    for (const val of arg.value) {
                        if (val.name) {
                            produced.add(val.name);
                        }
                    }
                }
            }
            const refs = new Set();
            for (const node of (graph.nodes || [])) {
                for (const arg of (node.inputs || [])) {
                    if (!Array.isArray(arg.value)) {
                        continue;
                    }
                    for (const val of arg.value) {
                        if (val.name && !val.initializer && !produced.has(val.name)) {
                            refs.add(val.name);
                        }
                    }
                }
            }
            return refs;
        };
        // Collect tunnel refs per (source, parent, attrName)
        const seen = new Set();
        for (const entry of this._nodes.values()) {
            const node = entry.label;
            if (!(node instanceof view.Node)) {
                continue;
            }
            const modelNode = node.value;
            const subgraphs = (modelNode.attributes || []).concat(modelNode.blocks || []);
            for (const attr of subgraphs) {
                if (attr.type !== 'graph' || !attr.value) {
                    continue;
                }
                const refs = subgraphOuterRefs(attr.value);
                for (const valueName of refs) {
                    const outerValue = this._values.get(valueName);
                    if (!outerValue || !outerValue.from) {
                        continue;
                    }
                    const sourceNode = outerValue.from;
                    const refKey = `${sourceNode.name}:${node.name}:${attr.name}`;
                    if (seen.has(refKey)) {
                        continue;
                    }
                    seen.add(refKey);
                    const edge = sourceNode.edge(node);
                    if (!edge._tunnel) {
                        edge._tunnel = true;
                    }
                    const edgeKey = `${edge.v}:${edge.w}`;
                    if (!this._edges.has(edgeKey)) {
                        this.setEdge(edge);
                    }
                    this._tunnels.push({
                        sourceNode,
                        parentNode: node,
                        attrName: attr.name,
                        valueName,
                        edge
                    });
                }
            }
        }
    }

    updateTunnels() {
        if (!this._tunnelGroup || !this._tunnels || !this._document) {
            return;
        }
        while (this._tunnelGroup.lastChild) {
            this._tunnelGroup.removeChild(this._tunnelGroup.lastChild);
        }
        if (this._tunnels.length === 0) {
            return;
        }
        const document = this._document;
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead-tunnel');
        marker.setAttribute('viewBox', '0 0 10 10');
        marker.setAttribute('refX', 9);
        marker.setAttribute('refY', 5);
        marker.setAttribute('markerUnits', 'strokeWidth');
        marker.setAttribute('markerWidth', 8);
        marker.setAttribute('markerHeight', 6);
        marker.setAttribute('orient', 'auto');
        const markerPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        markerPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z');
        markerPath.style.setProperty('stroke-width', 1);
        marker.appendChild(markerPath);
        defs.appendChild(marker);
        this._tunnelGroup.appendChild(defs);
        const intersectRect = (node, point) => {
            const dx = point.x - node.x;
            const dy = point.y - node.y;
            let h = node.height / 2;
            let w = node.width / 2;
            if (Math.abs(dy) * w > Math.abs(dx) * h) {
                if (dy < 0) {
                    h = -h;
                }
                return { x: node.x + (dy === 0 ? 0 : h * dx / dy), y: node.y + h };
            }
            if (dx < 0) {
                w = -w;
            }
            return { x: node.x + w, y: node.y + (dx === 0 ? 0 : w * dy / dx) };
        };
        const findTarget = (node, attrName, valueName) => {
            const nodeTop = node.y - node.height / 2;
            const nodeLeft = node.x - node.width / 2;
            for (const block of node.blocks) {
                if (!block._items) {
                    continue;
                }
                for (const item of block._items) {
                    if (item.name !== attrName) {
                        continue;
                    }
                    if (item.content && item.content.blocks) {
                        for (const innerBlock of item.content.blocks) {
                            if (innerBlock instanceof view.Block && innerBlock.target && innerBlock.target._values) {
                                const innerValue = innerBlock.target._values.get(valueName);
                                if (innerValue && innerValue.to.length > 0) {
                                    const innerNode = innerValue.to[0];
                                    if (innerNode.x !== undefined && innerNode.y !== undefined) {
                                        const padding = innerBlock._padding || 10;
                                        const originX = innerBlock.target.originX || 0;
                                        const originY = innerBlock.target.originY || 0;
                                        const contentNode = item.content;
                                        const cx = nodeLeft + block.x + (contentNode.x - contentNode.width / 2);
                                        const cy = nodeTop + block.y + (contentNode.y - contentNode.height / 2);
                                        return {
                                            x: cx + innerBlock.x + (padding - originX) + innerNode.x,
                                            y: cy + innerBlock.y + (padding - originY) + innerNode.y,
                                            width: innerNode.width || 0,
                                            height: innerNode.height || 0
                                        };
                                    }
                                }
                            }
                        }
                    }
                    if (item.content && item.content.x !== undefined) {
                        const contentNode = item.content;
                        return {
                            x: nodeLeft + block.x + contentNode.x,
                            y: nodeTop + block.y + contentNode.y,
                            width: contentNode.width,
                            height: contentNode.height
                        };
                    }
                    return {
                        x: nodeLeft + block.x + item.x + item.width / 2,
                        y: nodeTop + block.y + item.y + item.height / 2,
                        width: item.width,
                        height: item.height
                    };
                }
            }
            return { x: node.x, y: node.y, width: node.width || 0, height: node.height || 0 };
        };
        // Group tunnels by parent node to detect all overlaps into same target
        const groups = new Map();
        for (let i = 0; i < this._tunnels.length; i++) {
            const ref = this._tunnels[i];
            const key = ref.parentNode.name;
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push(i);
        }
        for (let i = 0; i < this._tunnels.length; i++) {
            const { parentNode, attrName, valueName, edge } = this._tunnels[i];
            if (!edge.points || edge.points.length < 2) {
                continue;
            }
            const target = findTarget(parentNode, attrName, valueName);
            const points = [];
            const inner = edge.points.slice(1, edge.points.length - 1);
            if (inner.length > 0) {
                points.push(intersectRect(edge.from, inner[0]));
                points.push(...inner);
            } else {
                points.push(intersectRect(edge.from, edge.points[edge.points.length - 1]));
            }
            points.push(intersectRect(target, points[points.length - 1]));
            // If multiple tunnels target the same parent, draw arcs to separate them
            const group = groups.get(parentNode.name);
            let pathData = '';
            if (group && group.length > 1) {
                const j = group.indexOf(i);
                const end = points[points.length - 1];
                const start = intersectRect(edge.from, end);
                const dy = Math.abs(end.y - start.y);
                const arcBase = Math.min(80, dy * 0.2);
                const arc = (j - (group.length - 1) / 2) * arcBase;
                const p = new grapher.Edge.Path();
                p.moveTo(start.x, start.y);
                p.bezierCurveTo(
                    start.x, start.y + (end.y - start.y) * 0.33,
                    end.x + arc, start.y + (end.y - start.y) * 0.67,
                    end.x, end.y
                );
                pathData = p.data;
            } else {
                pathData = new grapher.Edge.Curve(points).path.data;
            }
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('class', 'edge-path edge-path-tunnel');
            path.setAttribute('d', pathData);
            path.setAttribute('marker-end', 'url(#arrowhead-tunnel)');
            this._tunnelGroup.appendChild(path);
        }
    }

    build(document, origin) {
        if (!origin) {
            const element = document.getElementById('target');
            while (element.lastChild) {
                element.removeChild(element.lastChild);
            }
            const canvas = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            canvas.setAttribute('id', 'canvas');
            canvas.setAttribute('class', 'canvas');
            canvas.setAttribute('preserveAspectRatio', 'xMidYMid meet');
            canvas.setAttribute('width', '100%');
            canvas.setAttribute('height', '100%');
            element.appendChild(canvas);
            // Workaround for Safari background drag/zoom issue:
            // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
            const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            background.setAttribute('id', 'background');
            background.setAttribute('fill', 'none');
            background.setAttribute('pointer-events', 'all');
            canvas.appendChild(background);
            origin = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            origin.setAttribute('id', 'origin');
            canvas.appendChild(origin);
        }
        this._prepareEdgeBundles();
        for (const value of this._values.values()) {
            value.build();
        }
        super.build(document, origin);
    }

    async measure() {
        const document = this.host.document;
        const window = this.host.window;
        if (document.fonts && document.fonts.ready) {
            try {
                await document.fonts.ready;
            } catch {
                // continue regardless of error
            }
        }
        await new Promise((resolve) => {
            window.requestAnimationFrame(() => {
                window.requestAnimationFrame(() => {
                    window.requestAnimationFrame(resolve);
                });
            });
        });
        await super.measure();
    }

    clearSelection() {
        if (this._selection.size > 0) {
            for (const element of this._selection) {
                element.deselect();
            }
            this._selection.clear();
        }
        for (const entry of this._table.values()) {
            if (entry instanceof grapher.Node) {
                for (const block of entry.blocks) {
                    if (block.target && block.target.clearSelection) {
                        block.target.clearSelection();
                    }
                }
            }
        }
    }

    select(selection, source) {
        if (selection && this.view.target && this.view.target !== this) {
            this.view.target.clearSelection();
        } else {
            this.clearSelection();
        }
        if (selection) {
            let array = [];
            for (const value of selection) {
                if (this._table.has(value)) {
                    const element = this._table.get(value);
                    array = array.concat(element.select());
                    this._selection.add(element);
                }
            }
            this.emit('selectionchange', source);
            return array;
        }
        this.emit('selectionchange', source);
        return null;
    }

    activate(value, source) {
        if (this._table.has(value)) {
            this.select(null, source);
            const element = this._table.get(value);
            element.activate(source);
            return this.select([value], source);
        }
        return [];
    }

    focus(selection) {
        for (const value of selection) {
            const element = this._table.get(value);
            if (element && !this._selection.has(element)) {
                element.select();
            }
        }
    }

    blur(selection) {
        for (const value of selection) {
            const element = this._table.get(value);
            if (element && !this._selection.has(element)) {
                element.deselect();
            }
        }
    }

    restore(state) {
        const document = this.host.document;
        const canvas = document.getElementById('canvas');
        const origin = document.getElementById('origin');
        const background = document.getElementById('background');
        const elements = Array.from(canvas.getElementsByClassName('graph-input') || []);
        if (elements.length === 0) {
            const nodeElements = Array.from(canvas.getElementsByClassName('graph-node') || []);
            if (nodeElements.length > 0) {
                elements.push(nodeElements[0]);
            }
        }
        const size = canvas.getBBox();
        const margin = 100;
        const width = Math.ceil(margin + size.width + margin);
        const height = Math.ceil(margin + size.height + margin);
        origin.setAttribute('transform', `translate(${margin - size.x}, ${margin - size.y}) scale(1)`);
        background.setAttribute('width', width);
        background.setAttribute('height', height);
        this._width = width;
        this._height = height;
        delete this._scrollLeft;
        delete this._scrollRight;
        canvas.setAttribute('viewBox', `0 0 ${width} ${height}`);
        canvas.setAttribute('width', width);
        canvas.setAttribute('height', height);
        this._zoom = state ? state.zoom : 1;
        this._updateZoom(this._zoom);
        const container = document.getElementById('target');
        const context = state ? this.select([state.context]) : [];
        if (context.length > 0) {
            this.scrollTo(context, 'instant');
        } else if (elements && elements.length > 0) {
            // Center view based on input elements
            const bounds = container.getBoundingClientRect();
            const xs = [];
            const ys = [];
            for (let i = 0; i < elements.length; i++) {
                const element = elements[i];
                const rect = element.getBoundingClientRect();
                const width = Math.min(rect.width, bounds.width);
                const height = Math.min(rect.height, bounds.height);
                xs.push(rect.left + (width / 2));
                ys.push(rect.top + (height / 2));
            }
            let [x] = xs;
            const [y] = ys;
            if (ys.every((y) => y === ys[0])) {
                x = xs.reduce((a, b) => a + b, 0) / xs.length;
            }
            const left = (container.scrollLeft + x - bounds.left) - (bounds.width / 2);
            const top = (container.scrollTop + y - bounds.top) - (bounds.height / 2);
            container.scrollTo({ left, top, behavior: 'auto' });
        } else {
            const canvasRect = canvas.getBoundingClientRect();
            const graphRect = container.getBoundingClientRect();
            const left = (container.scrollLeft + (canvasRect.width / 2) - graphRect.left) - (graphRect.width / 2);
            const top = (container.scrollTop + (canvasRect.height / 2) - graphRect.top) - (graphRect.height / 2);
            container.scrollTo({ left, top, behavior: 'auto' });
        }
    }

    register() {
        if (!this._events) {
            this._events = {};
            this._events.scroll = (e) => this._scrollHandler(e);
            this._events.resize = () => this._updateScrollThumbs();
            this._events.wheel = (e) => this._wheelHandler(e);
            this._events.gesturestart = (e) => this._gestureStartHandler(e);
            this._events.pointerdown = (e) => this._pointerDownHandler(e);
            this._events.touchstart = (e) => this._touchStartHandler(e);
            const document = this.host.document;
            const element = document.getElementById('target');
            element.focus();
            element.addEventListener('scroll', this._events.scroll);
            element.addEventListener('wheel', this._events.wheel, { passive: false });
            element.addEventListener('pointerdown', this._events.pointerdown);
            const window = document.defaultView || this.host.window;
            if (window) {
                window.addEventListener('resize', this._events.resize);
            }
            this._registerScrollThumb('vertical');
            this._registerScrollThumb('horizontal');
            this._updateScrollThumbs();
            if (this.host.environment('agent') === 'safari') {
                element.addEventListener('gesturestart', this._events.gesturestart, false);
            } else {
                element.addEventListener('touchstart', this._events.touchstart, { passive: true });
            }
        }
    }

    unregister() {
        if (this._events) {
            const document = this.host.document;
            const window = document.defaultView || this.host.window;
            const element = document.getElementById('target');
            element.removeEventListener('scroll', this._events.scroll);
            element.removeEventListener('wheel', this._events.wheel);
            element.removeEventListener('pointerdown', this._events.pointerdown);
            if (window) {
                window.removeEventListener('resize', this._events.resize);
            }
            element.removeEventListener('gesturestart', this._events.gesturestart);
            element.removeEventListener('touchstart', this._events.touchstart);
            for (const direction of ['vertical', 'horizontal']) {
                const thumb = document.getElementById(`target-scroll-thumb-${direction}`);
                const handler = this._events[`thumb-${direction}`];
                if (thumb && handler) {
                    thumb.removeEventListener('pointerdown', handler);
                }
            }
            if (this._zoomFrame && window) {
                window.cancelAnimationFrame(this._zoomFrame);
            }
            delete this._zoomFrame;
            delete this._zoomTarget;
            delete this._zoomEvent;
            delete this._events;
        }
    }

    get zoom() {
        return this._zoom;
    }

    set zoom(value) {
        this._updateZoom(value);
    }

    _requestZoom(zoom, e) {
        const document = this.host.document;
        const window = document.defaultView;
        this._zoomTarget = zoom;
        this._zoomEvent = e ? { pageX: e.pageX, pageY: e.pageY } : null;
        if (!this._zoomFrame) {
            const update = () => {
                const zoom = this._zoomTarget;
                const event = this._zoomEvent;
                delete this._zoomFrame;
                delete this._zoomTarget;
                delete this._zoomEvent;
                this._updateZoom(zoom, event);
            };
            this._zoomFrame = window ? window.requestAnimationFrame(update) : globalThis.setTimeout(update, 0);
        }
    }

    _updateZoom(zoom, e) {
        const document = this.host.document;
        const container = document.getElementById('target');
        const canvas = document.getElementById('canvas');
        const limit = this.view.options.direction === 'vertical' ?
            container.clientHeight / this._height :
            container.clientWidth / this._width;
        const min = Math.min(Math.max(limit * 0.1, 0.01), 0.25);
        zoom = Math.max(min, Math.min(zoom, 3));
        const scrollLeft = this._scrollLeft || container.scrollLeft;
        const scrollTop = this._scrollTop || container.scrollTop;
        const x = (e ? e.pageX : (container.clientWidth / 2)) + scrollLeft;
        const y = (e ? e.pageY : (container.clientHeight / 2)) + scrollTop;
        const width = zoom * this._width;
        const height = zoom * this._height;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        this._scrollLeft = Math.max(0, ((x * zoom) / this._zoom) - (x - scrollLeft));
        this._scrollTop = Math.max(0, ((y * zoom) / this._zoom) - (y - scrollTop));
        container.scrollLeft = this._scrollLeft;
        container.scrollTop = this._scrollTop;
        this._zoom = zoom;
        this._updateScrollThumbs();
    }

    _pointerDownHandler(e) {
        if (e.pointerType === 'touch' || e.buttons !== 1) {
            return;
        }
        // Workaround for Firefox emitting 'pointerdown' event when scrollbar is pressed and interfering with dragging
        if (e.rangeParent === null) {
            return;
        }
        const document = this.host.document;
        const container = document.getElementById('target');
        const capture = e.target;
        capture.setPointerCapture(e.pointerId);
        this._mousePosition = {
            left: container.scrollLeft,
            top: container.scrollTop,
            x: e.clientX,
            y: e.clientY
        };
        e.target.style.cursor = 'grabbing';
        e.preventDefault();
        e.stopImmediatePropagation();
        let pointerUpHandler = null;
        const pointerMoveHandler = (e) => {
            if ((e.buttons & 1) === 0) {
                pointerUpHandler(e);
                return;
            }
            e.preventDefault();
            e.stopImmediatePropagation();
            if (this._mousePosition) {
                const dx = e.clientX - this._mousePosition.x;
                const dy = e.clientY - this._mousePosition.y;
                this._mousePosition.moved = dx * dx + dy * dy > 0;
                if (this._mousePosition.moved) {
                    const document = this.host.document;
                    const container = document.getElementById('target');
                    container.scrollTop = this._mousePosition.top - dy;
                    container.scrollLeft = this._mousePosition.left - dx;
                }
            }
        };
        const clickHandler = (e) => {
            e.stopPropagation();
            document.removeEventListener('click', clickHandler, true);
        };
        pointerUpHandler = (e) => {
            if (capture.hasPointerCapture(e.pointerId)) {
                capture.releasePointerCapture(e.pointerId);
            }
            capture.style.removeProperty('cursor');
            container.removeEventListener('pointerup', pointerUpHandler);
            container.removeEventListener('pointercancel', pointerUpHandler);
            container.removeEventListener('pointermove', pointerMoveHandler);
            const moved = Boolean(this._mousePosition && this._mousePosition.moved);
            delete this._mousePosition;
            if (moved) {
                e.preventDefault();
                e.stopImmediatePropagation();
                document.addEventListener('click', clickHandler, true);
            }
        };
        container.addEventListener('pointermove', pointerMoveHandler);
        container.addEventListener('pointerup', pointerUpHandler);
        container.addEventListener('pointercancel', pointerUpHandler);
    }

    _touchStartHandler(e) {
        if (e.touches.length === 2) {
            this._touchPoints = Array.from(e.touches);
            this._touchZoom = this._zoom;
        }
        const touchMoveHandler = (e) => {
            if (Array.isArray(this._touchPoints) && this._touchPoints.length === 2 && e.touches.length === 2) {
                const distance = (points) => {
                    const dx = (points[1].clientX - points[0].clientX);
                    const dy = (points[1].clientY - points[0].clientY);
                    return Math.sqrt(dx * dx + dy * dy);
                };
                const d1 = distance(Array.from(e.touches));
                const d2 = distance(this._touchPoints);
                if (d2 !== 0) {
                    const points = this._touchPoints;
                    const e = {
                        pageX: (points[1].pageX + points[0].pageX) / 2,
                        pageY: (points[1].pageY + points[0].pageY) / 2
                    };
                    const scale = d2 === 0 ? d1 : d1 / d2;
                    const zoom = this._touchZoom * Math.pow(scale, 0.25);
                    this._requestZoom(zoom, e);
                }
            }
        };
        const document = this.host.document;
        const container = document.getElementById('target');
        const touchEndHandler = () => {
            container.removeEventListener('touchmove', touchMoveHandler, { passive: true });
            container.removeEventListener('touchcancel', touchEndHandler, { passive: true });
            container.removeEventListener('touchend', touchEndHandler, { passive: true });
            delete this._touchPoints;
            delete this._touchZoom;
        };
        container.addEventListener('touchmove', touchMoveHandler, { passive: true });
        container.addEventListener('touchcancel', touchEndHandler, { passive: true });
        container.addEventListener('touchend', touchEndHandler, { passive: true });
    }

    _gestureStartHandler(e) {
        e.preventDefault();
        this._gestureZoom = this._zoom;
        const document = this.host.document;
        const container = document.getElementById('target');
        const gestureChangeHandler = (e) => {
            e.preventDefault();
            this._requestZoom(this._gestureZoom * Math.pow(e.scale, 0.25), e);
        };
        const gestureEndHandler = (e) => {
            container.removeEventListener('gesturechange', gestureChangeHandler, false);
            container.removeEventListener('gestureend', gestureEndHandler, false);
            e.preventDefault();
            if (this._gestureZoom) {
                this._requestZoom(this._gestureZoom * Math.pow(e.scale, 0.25), e);
                delete this._gestureZoom;
            }
        };
        container.addEventListener('gesturechange', gestureChangeHandler, false);
        container.addEventListener('gestureend', gestureEndHandler, false);
    }

    _scrollHandler(e) {
        if (this._scrollLeft && e.target.scrollLeft !== Math.floor(this._scrollLeft)) {
            delete this._scrollLeft;
        }
        if (this._scrollTop && e.target.scrollTop !== Math.floor(this._scrollTop)) {
            delete this._scrollTop;
        }
        this._updateScrollThumbs();
    }

    _registerScrollThumb(direction) {
        const document = this.host.document;
        if (!document.documentElement.classList.contains('vscode-webview')) {
            return;
        }
        const container = document.getElementById('target');
        const thumb = document.getElementById(`target-scroll-thumb-${direction}`);
        if (!thumb) {
            return;
        }
        const pointerDown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            thumb.setPointerCapture(event.pointerId);
            thumb.classList.add('dragging');
            const vertical = direction === 'vertical';
            const start = vertical ? event.clientY : event.clientX;
            const scrollStart = vertical ? container.scrollTop : container.scrollLeft;
            const clientSize = vertical ? container.clientHeight : container.clientWidth;
            const scrollSize = vertical ? container.scrollHeight : container.scrollWidth;
            const thumbSize = vertical ? thumb.offsetHeight : thumb.offsetWidth;
            const trackSize = Math.max(1, clientSize - 6 - thumbSize);
            const scrollRange = Math.max(0, scrollSize - clientSize);
            const move = (event) => {
                const position = vertical ? event.clientY : event.clientX;
                const scroll = scrollStart + ((position - start) * scrollRange / trackSize);
                if (vertical) {
                    container.scrollTop = scroll;
                } else {
                    container.scrollLeft = scroll;
                }
            };
            const end = (event) => {
                thumb.releasePointerCapture(event.pointerId);
                thumb.classList.remove('dragging');
                thumb.removeEventListener('pointermove', move);
                thumb.removeEventListener('pointerup', end);
                thumb.removeEventListener('pointercancel', end);
            };
            thumb.addEventListener('pointermove', move);
            thumb.addEventListener('pointerup', end);
            thumb.addEventListener('pointercancel', end);
        };
        this._events[`thumb-${direction}`] = pointerDown;
        thumb.addEventListener('pointerdown', pointerDown);
    }

    _updateScrollThumbs() {
        const document = this.host.document;
        if (!document.documentElement.classList.contains('vscode-webview')) {
            return;
        }
        const container = document.getElementById('target');
        if (!container) {
            return;
        }
        const update = (direction) => {
            const vertical = direction === 'vertical';
            const thumb = document.getElementById(`target-scroll-thumb-${direction}`);
            if (!thumb) {
                return;
            }
            const clientSize = vertical ? container.clientHeight : container.clientWidth;
            const scrollSize = vertical ? container.scrollHeight : container.scrollWidth;
            const scrollPosition = vertical ? container.scrollTop : container.scrollLeft;
            const scrollRange = scrollSize - clientSize;
            if (clientSize <= 0 || scrollRange <= 1) {
                thumb.classList.remove('visible');
                return;
            }
            const trackSize = Math.max(1, clientSize - 6);
            const thumbSize = Math.max(24, Math.round(trackSize * clientSize / scrollSize));
            const offset = Math.round((trackSize - thumbSize) * scrollPosition / scrollRange);
            thumb.classList.add('visible');
            if (vertical) {
                thumb.style.height = `${thumbSize}px`;
                thumb.style.transform = `translateY(${offset}px)`;
            } else {
                thumb.style.width = `${thumbSize}px`;
                thumb.style.transform = `translateX(${offset}px)`;
            }
        };
        update('vertical');
        update('horizontal');
    }

    _wheelHandler(e) {
        const command = this.host.environment('platform') === 'darwin' && e.metaKey;
        if (e.shiftKey || e.ctrlKey || command || this.view.options.mousewheel === 'zoom') {
            let factor = 1;
            if (e.deltaMode === 1) {
                factor = 0.05;
            } else if (e.deltaMode) {
                factor = 1;
            } else {
                factor = 0.002;
            }
            const delta = -e.deltaY * factor * (e.ctrlKey ? 10 : 1) * 0.25;
            const zoom = this._zoomTarget || this._zoom;
            this._requestZoom(zoom * Math.pow(2, delta), e);
            e.preventDefault();
        }
    }

    scrollTo(selection, behavior) {
        if (selection && selection.length > 0) {
            const document = this.host.document;
            const container = document.getElementById('target');
            const rect = container.getBoundingClientRect();
            // Exclude scrollbars
            const cw = container.clientWidth;
            const ch = container.clientHeight;
            // Shrink the test rectangle by 10%
            const bounds = {};
            bounds.left = (rect.x + cw / 2) - (cw * 0.45);
            bounds.width = cw * 0.9;
            bounds.right = bounds.left + bounds.width;
            bounds.top = (rect.y + ch / 2) - (ch * 0.45);
            bounds.height = ch * 0.9;
            bounds.bottom = bounds.top + bounds.height;
            let x = 0;
            let y = 0;
            let left = Number.POSITIVE_INFINITY;
            let right = Number.NEGATIVE_INFINITY;
            let top = Number.POSITIVE_INFINITY;
            let bottom = Number.NEGATIVE_INFINITY;
            for (const element of selection) {
                const rect = element.getBoundingClientRect();
                const width = Math.min(rect.width, bounds.width);
                const height = Math.min(rect.height, bounds.height);
                x += rect.left + (width / 2);
                y += rect.top + (height / 2);
                left = Math.min(left, rect.left);
                right = Math.max(right, rect.right);
                top = Math.min(top, rect.top);
                bottom = Math.max(bottom, rect.bottom);
            }
            // No need to scroll if new selection is in the safe area.
            if (right <= bounds.right && left >= bounds.left && bottom <= bounds.bottom && top >= bounds.top) {
                return;
            }
            // If new selection is completely out of the bounds, scroll to centerize it.
            if (bottom - top >= bounds.height || right - left >= bounds.width || right < rect.left || left > rect.right || bottom < rect.top || top > rect.bottom) {
                x /= selection.length;
                y /= selection.length;
                const options = {};
                options.left = (container.scrollLeft + x - bounds.left) - (bounds.width / 2);
                options.top = (container.scrollTop + y - bounds.top) - (bounds.height / 2);
                options.behavior = behavior || 'smooth';
                container.scrollTo(options);
                return;
            }
            const options = {};
            options.left = 0;
            options.top = 0;
            options.behavior = behavior || 'smooth';
            // similar to scrollIntoView block: "nearest"
            const dr = bounds.right - right;
            const dl = left - bounds.left;
            const db = bounds.bottom - bottom;
            const dt = top - bounds.top;
            if (right - left < bounds.width) {
                if (dl < 0) {
                    options.left = dl;
                } else if (dr < 0) {
                    options.left = -dr;
                }
            }
            if (bottom - top < bounds.height) {
                if (dt < 0) {
                    options.top = dt;
                } else if (db < 0) {
                    options.top = -db;
                }
            }
            container.scrollBy(options);
        }
    }
};

view.GraphEditInputPorts = class {

    constructor(owner) {
        this._owner = owner;
        this._ports = [];
        for (const argument of owner.inputs || []) {
            for (let valueIndex = 0; valueIndex < (Array.isArray(argument.value) ? argument.value.length : 0); valueIndex++) {
                const value = argument.value[valueIndex];
                if (value && !value.initializer) {
                    this._ports.push({ argument, value, valueIndex, element: null });
                }
            }
        }
    }

    build(document, parent) {
        const node = this._owner.value;
        const bundles = new Set();
        const ports = [];
        for (const port of this._ports) {
            const visualValue = port.value && port.value.name ?
                this._owner.context.values.get(port.value.name) : null;
            const candidateBundle = visualValue ? this._owner.context.edgeBundle(visualValue) : null;
            const bundle = candidateBundle && candidateBundle.to === this._owner &&
                candidateBundle.input === port.argument ? candidateBundle : null;
            if (!bundle) {
                const suffix = port.argument.value.length > 1 ? `[${port.valueIndex}]` : '';
                port.entries = [{
                    argument: port.argument,
                    valueIndex: port.valueIndex,
                    value: port.value,
                    label: `${port.argument.name}${suffix}`
                }];
                ports.push(port);
                continue;
            }
            if (bundles.has(bundle)) {
                continue;
            }
            bundles.add(bundle);
            const target = bundle.edge.graphEditTarget;
            const entries = bundle.inputEntries
                .map((candidate) => {
                    const value = candidate.value;
                    const suffix = candidate.argument.value.length > 1 ? `[${candidate.valueIndex}]` : '';
                    return {
                        argument: candidate.argument,
                        valueIndex: candidate.valueIndex,
                        value,
                        label: `${candidate.argument.name}${suffix}`
                    };
                });
            ports.push({
                argument: target.argument,
                value: target.argument.value[target.valueIndex],
                valueIndex: target.valueIndex,
                entries,
                bundle,
                bundleCount: bundle.connectionCount,
                element: null
            });
        }
        this._ports = ports;
        for (const port of this._ports) {
            const suffix = port.argument.value.length > 1 ? `[${port.valueIndex}]` : '';
            const name = `${node.name || node.type.name}.${port.argument.name}${suffix}`;
            const element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            const classes = ['graph-edit-input-port'];
            if (port.argument.option === 'optional') {
                classes.push('graph-edit-input-port-optional');
                if (port.value.name) {
                    classes.push('graph-edit-input-port-connected');
                }
            } else if (!port.value.name) {
                classes.push('graph-edit-input-port-required-missing');
            }
            element.setAttribute('class', classes.join(' '));
            element.setAttribute('role', 'button');
            element.setAttribute('tabindex', '0');
            element.setAttribute('aria-label', port.bundle ?
                `Choose bundled inputs ×${port.bundleCount} for ${node.name || node.type.name}` : `Connect to ${name}`);
            const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            marker.setAttribute('class', 'graph-edit-input-port-marker');
            marker.setAttribute('r', '6');
            element.appendChild(marker);
            if (port.bundle) {
                const count = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                count.setAttribute('class', 'graph-edit-port-bundle-count');
                count.setAttribute('x', '8');
                count.setAttribute('y', '-7');
                count.textContent = `×${port.bundleCount}`;
                element.appendChild(count);
            }
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            title.textContent = `Connect to input: ${name}`;
            element.appendChild(title);
            element.graphEditNode = node;
            element.graphEditArgument = port.argument;
            element.graphEditValueIndex = port.valueIndex;
            element.graphEditTitle = title;
            const activate = (event) => {
                event.preventDefault();
                event.stopPropagation();
                const result = this._owner.context.view.graphEditChooseInput(node, port.entries, element, event);
                if (result && typeof result.catch === 'function') {
                    result.catch((error) => this._owner.context.view.error(error, 'ONNX GraphSurgeon failed.', null));
                }
            };
            element.addEventListener('click', activate);
            element.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    activate(event);
                }
            });
            parent.appendChild(element);
            port.element = element;
        }
    }

    update() {
        const owner = this._owner;
        const horizontal = owner.context.options.direction === 'horizontal';
        for (let index = 0; index < this._ports.length; index++) {
            const port = this._ports[index];
            const offset = (index + 1) / (this._ports.length + 1);
            const position = owner.context.graphEditInputPortPosition(port.argument, port.valueIndex);
            let x = position ? position.x : 0;
            let y = position ? position.y : 0;
            if (!position) {
                x = horizontal ? 0 : owner.width * offset;
                y = horizontal ? owner.height * offset : 0;
            }
            port.element.setAttribute('transform', `translate(${x},${y})`);
        }
    }
};

view.GraphEditOutputPorts = class {

    constructor(owner) {
        this._owner = owner;
        this._ports = [];
        const seen = new Set();
        for (const argument of owner.outputs || []) {
            for (let valueIndex = 0; valueIndex < (argument && Array.isArray(argument.value) ? argument.value.length : 0); valueIndex++) {
                const value = argument.value[valueIndex];
                if (value && value.name && !value.initializer && !seen.has(value)) {
                    seen.add(value);
                    const suffix = argument.value.length > 1 ? `[${valueIndex}]` : '';
                    this._ports.push({
                        value,
                        entries: [{ value, label: `${argument.name}${suffix}` }],
                        argument,
                        valueIndex,
                        elements: []
                    });
                }
            }
        }
    }

    build(document, parent) {
        this._document = document;
        this._parent = parent;
        const bundled = new Set();
        const ports = [];
        const byName = new Map(this._ports.map((port) => [port.value.name, port]));
        for (const port of this._ports) {
            const visualValue = this._owner.context.values.get(port.value.name);
            const bundle = visualValue ? this._owner.context.edgeBundle(visualValue) : null;
            if (!bundle) {
                ports.push(port);
                continue;
            }
            if (bundled.has(bundle)) {
                continue;
            }
            bundled.add(bundle);
            const representative = bundle.edge && bundle.edge.value ? bundle.edge.value.value : port.value;
            const entries = bundle.values.map((value) => byName.get(value.value.name))
                .filter((entry) => entry)
                .map((entry) => ({ value: entry.value, label: entry.entries[0].label }));
            ports.push({
                value: representative,
                entries,
                bundle,
                bundleCount: bundle.connectionCount,
                elements: []
            });
        }
        this._ports = ports;
        for (const port of this._ports) {
            const count = port.bundle ? 1 : this._owner.context.graphEditOutputPortCount(port.value);
            for (let index = 0; index < count; index++) {
                this._create(port);
            }
        }
    }

    _create(port) {
        const element = this._document.createElementNS('http://www.w3.org/2000/svg', 'g');
        element.setAttribute('class', 'graph-edit-output-port');
        element.setAttribute('role', 'button');
        element.setAttribute('tabindex', '0');
        element.setAttribute('aria-label', port.bundle ?
            `Choose bundled outputs ×${port.bundleCount}` : `Use output ${port.value.name}`);
        const marker = this._document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('class', 'graph-edit-output-port-marker');
        marker.setAttribute('r', '6');
        element.appendChild(marker);
        if (port.bundle) {
            const count = this._document.createElementNS('http://www.w3.org/2000/svg', 'text');
            count.setAttribute('class', 'graph-edit-port-bundle-count');
            count.setAttribute('x', '8');
            count.setAttribute('y', '-7');
            count.textContent = `×${port.bundleCount}`;
            element.appendChild(count);
        }
        const title = this._document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = port.bundle ?
            `Click to choose one of ${port.entries.length} ONNX output tensors; drag to reconnect the representative bundled path` :
            `Create a new connection from ${port.value.name}`;
        element.appendChild(title);
        element.graphEditValue = port.value;
        element.graphEditTitle = title;
        const activate = (event) => {
            event.preventDefault();
            event.stopPropagation();
            const result = this._owner.context.view.graphEditChooseOutput(port.entries, element, event);
            if (result && typeof result.catch === 'function') {
                result.catch((error) => this._owner.context.view.error(error, 'ONNX GraphSurgeon failed.', null));
            }
        };
        element.addEventListener('pointerdown', (event) => {
            this._owner.context.view.beginGraphEditOutputDrag(
                port.value, element, event, port.bundle ? port.entries : null);
        });
        element.addEventListener('click', activate);
        element.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                activate(event);
            }
        });
        this._parent.appendChild(element);
        port.elements.push(element);
    }

    update() {
        const owner = this._owner;
        const horizontal = owner.context.options.direction === 'horizontal';
        for (let index = 0; index < this._ports.length; index++) {
            const port = this._ports[index];
            const offset = (index + 1) / (this._ports.length + 1);
            const count = port.bundle ? 1 : owner.context.graphEditOutputPortCount(port.value);
            while (port.elements.length < count) {
                this._create(port);
            }
            while (port.elements.length > count) {
                port.elements.pop().remove();
            }
            const positions = owner.context.graphEditOutputPortPositions(port.value);
            for (let positionIndex = 0; positionIndex < port.elements.length; positionIndex++) {
                const element = port.elements[positionIndex];
                const position = positions[positionIndex] || null;
                const visible = positionIndex < Math.max(1, positions.length);
                element.style.display = visible ? '' : 'none';
                let x = position ? position.x : 0;
                let y = position ? position.y : 0;
                if (!position) {
                    x = horizontal ? owner.width : owner.width * offset;
                    y = horizontal ? owner.height * offset : owner.height;
                }
                element.setAttribute('transform', `translate(${x},${y})`);
            }
        }
    }
};

view.GraphEditGraphOutputPorts = class {

    constructor(owner) {
        this._owner = owner;
        this._ports = [];
        for (let valueIndex = 0; valueIndex < (Array.isArray(owner.value.value) ? owner.value.value.length : 0); valueIndex++) {
            const value = owner.value.value[valueIndex];
            if (value && !value.initializer) {
                this._ports.push({ argument: owner.value, valueIndex, element: null });
            }
        }
    }

    build(document, parent) {
        for (const port of this._ports) {
            const argument = port.argument;
            const suffix = argument.value.length > 1 ? `[${port.valueIndex}]` : '';
            const name = `graph output ${argument.name}${suffix}`;
            const element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            const missing = !argument.value[port.valueIndex] || !argument.value[port.valueIndex].name;
            element.setAttribute('class',
                `graph-edit-input-port graph-edit-graph-output-port${missing ? ' graph-edit-graph-output-port-missing' : ''}`);
            element.setAttribute('role', 'button');
            element.setAttribute('tabindex', '0');
            element.setAttribute('aria-label', `Connect to ${name}`);
            const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            marker.setAttribute('class', 'graph-edit-input-port-marker');
            marker.setAttribute('r', '7');
            element.appendChild(marker);
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            title.textContent = `Connect to ${name}`;
            element.appendChild(title);
            element.graphEditGraphOutput = true;
            element.graphEditArgument = argument;
            element.graphEditValueIndex = port.valueIndex;
            element.graphEditTitle = title;
            const activate = (event) => {
                event.preventDefault();
                event.stopPropagation();
                const result = this._owner.context.view.graphEditGraphOutput(argument, port.valueIndex);
                if (result && typeof result.catch === 'function') {
                    result.catch((error) => this._owner.context.view.error(error, 'ONNX GraphSurgeon failed.', null));
                }
            };
            element.addEventListener('click', activate);
            element.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    activate(event);
                }
            });
            parent.appendChild(element);
            port.element = element;
        }
    }

    update() {
        const owner = this._owner;
        const horizontal = owner.context.options.direction === 'horizontal';
        for (let index = 0; index < this._ports.length; index++) {
            const port = this._ports[index];
            const offset = (index + 1) / (this._ports.length + 1);
            const position = owner.context.graphEditGraphOutputPortPosition(port.argument, port.valueIndex);
            let x = position ? position.x : 0;
            let y = position ? position.y : 0;
            if (!position) {
                x = horizontal ? 0 : owner.width * offset;
                y = horizontal ? owner.height * offset : 0;
            }
            port.element.setAttribute('transform', `translate(${x},${y})`);
        }
    }
};

view.Node = class extends grapher.Node {

    constructor(context, value, type) {
        super();
        this.context = context;
        this.value = value;
        this.id = `node-${value.name ? `name-${value.name}` : `id-${(context.counter++)}`}`;
        this._add(value, type);
        this._graphEditInputPorts = new view.GraphEditInputPorts(this);
        this._graphEditOutputPorts = new view.GraphEditOutputPorts(this);
        const inputs = value.inputs;
        if (type !== 'graph' && type !== 'function' && Array.isArray(inputs)) {
            for (const argument of inputs) {
                if (!argument.type || argument.type.endsWith('*')) {
                    if (Array.isArray(argument.value) && argument.value.length === 1 && argument.value[0].initializer) {
                        context.createArgument(argument);
                    } else {
                        for (const value of argument.value) {
                            if (value === null) {
                                // null argument
                            } else if (value.name !== '' && !value.initializer) {
                                context.createValue(value).to.push(this);
                            } else if (value.initializer) {
                                context.createValue(value);
                            }
                        }
                    }
                } else if (Array.isArray(argument.value) && argument.value.some((value) => value && value.constructor && value.constructor.name === 'Value' && typeof value.name === 'string' && value.name !== '' && !value.initializer)) {
                    for (const value of argument.value) {
                        if (value && value.constructor && value.constructor.name === 'Value' && typeof value.name === 'string' && value.name !== '' && !value.initializer) {
                            context.createValue(value).to.push(this);
                        }
                    }
                }
            }
        }
    }

    get class() {
        return 'graph-node';
    }

    get inputs() {
        return this.value.inputs;
    }

    get outputs() {
        return this.value.outputs;
    }

    build(document, parent) {
        super.build(document, parent);
        this.element.addEventListener('pointerdown', (event) => {
            this.context.view.beginGraphEditNodeDrag(this, event);
        });
        this.element.addEventListener('click', (event) => {
            this.context.view.graphEditNodeMenu(this.value, event);
        });
        this._graphEditInputPorts.build(document, this.element);
        this._graphEditOutputPorts.build(document, this.element);
    }

    update() {
        super.update();
        this._graphEditInputPorts.update();
        this._graphEditOutputPorts.update();
    }

    _add(value, type) {
        const node = (type === 'graph' || type === 'function') ? { type: value } : value;
        const options = this.context.options;
        const header =  this.header();
        const category = node.type && node.type.category ? node.type.category : '';
        if (node.type && typeof node.type.name !== 'string' || !node.type.name.split) { // #416
            const error = new view.Error(`Unsupported node type '${JSON.stringify(node.type.name)}'.`);
            if (this.context.model && this.context.model.identifier) {
                error.context = this.context.model.identifier;
            }
            throw error;
        }
        let content = options.names && (node.name || node.identifier) ? (node.name || node.identifier) : node.type.name.split('.').pop();
        let tooltip = options.names && (node.name || node.identifier) ? `[${node.type.name}]` : (node.name || node.identifier);
        if (content.length > 21) {
            tooltip = options.names ? `${content}` : `[${content}]`;
            const begin = content.substring(0, 10);
            const end = content.substring(content.length - 10, content.length);
            content = `${begin}\u2026${end}`;
        }
        const styles = category ? ['node-item-type', `node-item-type-${category.toLowerCase()}`] : ['node-item-type'];
        const title = header.add(null, styles);
        title.content = content;
        title.tooltip = tooltip;
        title.on('click', () => {
            this.context.activate(value, 'target');
        });
        const quantization = this.context.model.attachment.quantization;
        const explicit = quantization.node(node, false);
        const precision = quantization.node(node);
        const encodingBadge = aimet.EncodingFile.nodeBadge(explicit, precision);
        if (encodingBadge.labels.length > 0) {
            const heatmap = aimet.EncodingFile.precision(precision);
            const badge = header.add(null, ['node-item-quantization', `node-item-quantization-${heatmap}`]);
            badge.content = encodingBadge.labels.join(' ');
            badge.tooltip = encodingBadge.descriptions.join(', ');
            badge.padding = 5;
            badge.on('click', () => {
                this.context.activate(value, 'target');
            });
        }
        if (type === 'graph') {
            this.definition = header.add(null, styles);
            this.definition.content = '\u25CB';
            this.definition.tooltip = 'Show Graph';
            this.definition.padding = 4;
            this.definition.on('click', async () => await this.context.view.pushTarget(value, this.value));
            const expanded = this.context.blocks.has(value);
            const icon = expanded ? '\u2212' : '+';
            const tooltip = expanded ? 'Collapse Graph' : 'Expand Graph';
            this.expander = header.add(null, styles);
            this.expander.content = icon;
            this.expander.tooltip = tooltip;
            this.expander.padding = 6;
            this.expander.on('click', () => {
                const rect = this.expander.element.getBoundingClientRect();
                if (this.context.blocks.has(value)) {
                    this.context.blocks.delete(value);
                } else {
                    this.context.blocks.add(value);
                }
                this.context.view.refresh({ value: this.value, rect });
            });
        } else if (node.type.type || (Array.isArray(node.type.nodes) && node.type.nodes.length > 0)) {
            let icon = '\u0192';
            let tooltip = 'Show Function Definition';
            if (node.type.type === 'weights') {
                icon = '\u25CF';
                tooltip = 'Show Weights';
            }
            this.definition = header.add(null, styles);
            this.definition.content = icon;
            this.definition.tooltip = tooltip;
            this.definition.on('click', async () => await this.context.view.pushTarget(node.type, this.value));
        }
        let current = null;
        const list = () => {
            if (!current) {
                current = this.list();
                current.on('click', () => this.context.activate(node, 'target'));
            }
            return current;
        };
        let hiddenTensors = false;
        const objects = [];
        const attribute = (argument) => {
            let content = new view.Formatter(argument.value, argument.type).toString();
            if (content && content.length > 12) {
                content = `${content.substring(0, 12)}\u2026`;
            }
            const item = list().argument(argument.name, content);
            item.tooltip = argument.type;
            if (!content.startsWith('\u3008')) {
                item.separator = ' = ';
            }
            return item;
        };
        const isObject = (node) => {
            if (node.name || node.identifier || node.description ||
                (Array.isArray(node.inputs) && node.inputs.length > 0) ||
                (Array.isArray(node.outputs) && node.outputs.length > 0) ||
                (Array.isArray(node.attributes) && node.attributes.length > 0) ||
                (Array.isArray(node.blocks) && node.blocks.length > 0) ||
                (Array.isArray(node.chain) && node.chain.length > 0) ||
                (node.type && Array.isArray(node.type.nodes) && node.type.nodes.length > 0)) {
                return true;
            }

            return false;
        };
        const inputs = node.inputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                const type = argument.type;
                if (argument.visible !== false &&
                    ((type === 'graph' && argument.value) ||
                    (type === 'object' && isObject(argument.value)) ||
                    (type === 'object[]' && Array.isArray(argument.value) && argument.value.length > 0) ||
                    type === 'function' ||
                    (type === 'function[]' && Array.isArray(argument.value) && argument.value.length > 0))) {
                    objects.push(argument);
                } else if (options.weights && argument.visible !== false && argument.type !== 'attribute' && Array.isArray(argument.value) && argument.value.length === 1 && argument.value[0].initializer) {
                    const item = this.context.createArgument(argument);
                    list().add(item);
                } else if (options.weights && (argument.visible === false || Array.isArray(argument.value) && argument.value.length > 1) && (!argument.type || argument.type.endsWith('*')) && argument.value.some((value) => value !== null && value.initializer)) {
                    hiddenTensors = true;
                } else if (options.attributes && argument.visible !== false && argument.type && !argument.type.endsWith('*')) {
                    const item = attribute(argument);
                    list().add(item);
                }
            }
        }
        if (Array.isArray(node.attributes)) {
            const attributes = node.attributes.slice();
            attributes.sort((a, b) => a.name.toUpperCase().localeCompare(b.name.toUpperCase()));
            for (const argument of attributes) {
                const type = argument.type;
                if (argument.visible !== false &&
                    ((type === 'graph' && argument.value) ||
                    (type === 'object' && argument.value) ||
                    ((type === 'object[]' || type === 'function' || type === 'function[]') && Array.isArray(argument.value) && argument.value.length > 0))) {
                    objects.push(argument);
                } else if (options.attributes && argument.visible !== false) {
                    const item = attribute(argument);
                    list().add(item);
                }
            }
        }
        if (Array.isArray(node.blocks)) {
            for (const argument of node.blocks) {
                const type = argument.type;
                if (argument.visible !== false &&
                    ((type === 'graph' && argument.value) ||
                    (type === 'object' && isObject(argument.value)) ||
                    ((type === 'object[]' || type === 'function' || type === 'function[]') && Array.isArray(argument.value) && argument.value.length > 0))) {
                    objects.push(argument);
                }
            }
        }
        if (hiddenTensors) {
            const item = list().argument('\u3008\u2026\u3009', '');
            list().add(item);
        }
        for (const argument of objects) {
            const type = argument.type;
            let content = null;
            if (type === 'graph' && this.context.blocks.has(argument.value)) {
                content = this.context.createGraph(argument.value);
                content.blocks.push(new view.Block(this.context.view, argument.value, this.context.blocks));
                content.activate = () => this.context.view.showTargetProperties(argument.value);
                const item = list().argument(argument.name, content);
                list().add(item);
            } else if (type === 'graph' || type === 'function') {
                content = this.context.createGraph(argument.value, type);
                content.activate = () => this.context.view.showTargetProperties(argument.value);
                const item = list().argument(argument.name, content);
                list().add(item);
            } else if (type === 'graph[]') {
                content = argument.value.map((value) => this.context.createGraph(value));
                const item = list().argument(argument.name, content);
                list().add(item);
            } else {
                if (argument.type === 'object') {
                    content = this.context.createNode(argument.value);
                } else if (type === 'function[]' || argument.type === 'object[]') {
                    content = argument.value.map((value) => this.context.createNode(value));
                }
                const item = list().argument(argument.name, content);
                list().add(item);
            }
        }
        if (Array.isArray(node.chain) && node.chain.length > 0) {
            for (const innerNode of node.chain) {
                this.context.createNode(innerNode);
                this._add(innerNode);
            }
        }
        if (node.inner) {
            this.context.createNode(node.inner);
            this._add(node.inner);
        }
    }

    activate(source) {
        this.context.view.showNodeProperties(this.value, source);
    }

    edge(to, value) {
        this._edges = this._edges || new Map();
        if (value) {
            return new view.Edge(this, to);
        }
        if (!this._edges.has(to)) {
            this._edges.set(to, new view.Edge(this, to));
        }
        return this._edges.get(to);
    }
};

view.Block = class {

    constructor(viewRef, target, blocks) {
        this.target = new view.Graph(viewRef, false);
        if (blocks) {
            this.target.blocks = blocks;
        }
        this.target.add(target);
        this.x = 0;
        this.y = 0;
    }

    build(document, parent) {
        this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.element.setAttribute('class', 'node-block');
        parent.appendChild(this.element);
        if (!this.first) {
            this.line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            this.line.setAttribute('class', 'node');
            parent.appendChild(this.line);
        }
        this._background = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this._background.setAttribute('class', 'node-block-background');
        this.element.appendChild(this._background);
        this._origin = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.element.appendChild(this._origin);
        for (const value of this.target.values.values()) {
            value.build();
        }
        this.target.build(document, this._origin);
    }

    async measure() {
        for (const edge of this.target.edges.values()) {
            if (edge.label.labelElement) {
                const box = edge.label.labelElement.getBBox();
                edge.label.width = box.width;
                edge.label.height = box.height;
            }
        }
        await this.target.measure();
        await this.target.layout();
        const padding = 10;
        this._padding = padding;
        this.width = (this.target.width || 0) + 2 * padding;
        this.height = (this.target.height || 0) + 2 * padding;
    }

    async layout() {
    }

    update() {
        const offsetX = this._padding - (this.target.originX || 0);
        const offsetY = this._padding - (this.target.originY || 0);
        this.element.setAttribute('transform', `translate(0,${this.y})`);
        this._origin.setAttribute('transform', `translate(${offsetX},${offsetY})`);
        this._background.setAttribute('d', grapher.Node.roundedRect(0, 0, this.width, this.height, false, false, this.last, this.last));
        this.target.update();
        if (this.line) {
            this.line.setAttribute('x1', 0);
            this.line.setAttribute('x2', this.width);
            this.line.setAttribute('y1', this.y);
            this.line.setAttribute('y2', this.y);
        }
    }
};

view.Input = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        view.Input.counter = view.Input.counter || 0;
        const types = value.value.map((argument) => argument.type || '').join('\n');
        let name = value.name || '';
        if (name.length > 16) {
            name = name.split('/').pop();
        }
        const header = this.header();
        const title = header.add(null, ['graph-item-input']);
        title.content = name;
        title.tooltip = types;
        title.on('click', () => this.context.view.showTensorProperties(this.value, null, {
            title: 'Graph Input Properties',
            quantizationTitle: 'Graph Input QParam'
        }));
        const quantization = this.context.model.attachment.quantization;
        const producer = { inputs: [], outputs: [value] };
        const explicit = quantization.node(producer, false);
        const precision = quantization.node(producer);
        const encodingBadge = aimet.EncodingFile.nodeBadge(explicit, precision);
        if (encodingBadge.labels.length > 0) {
            const heatmap = aimet.EncodingFile.precision(precision);
            const badge = header.add(null, ['node-item-quantization', `node-item-quantization-${heatmap}`]);
            badge.content = encodingBadge.labels.join(' ');
            badge.tooltip = encodingBadge.descriptions
                .map((description) => description.replace('Output QParams', 'Graph Input QParam'))
                .join(', ');
            badge.padding = 5;
            badge.on('click', () => this.context.view.showTensorProperties(this.value, null, {
                title: 'Graph Input Properties',
                quantizationTitle: 'Graph Input QParam'
            }));
        }
        this.id = `input-${name ? `name-${name}` : `id-${(view.Input.counter++)}`}`;
        this._graphEditOutputPorts = new view.GraphEditOutputPorts(this);
    }

    get target() {
        return this.context.target === this.context.view.activeTarget ? null : this.context.target;
    }

    get class() {
        return 'graph-input';
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [this.value];
    }

    activate() {
        this.context.view.showTensorProperties(this.value, null, {
            title: 'Graph Input Properties',
            quantizationTitle: 'Graph Input QParam'
        });
    }

    build(document, parent) {
        super.build(document, parent);
        this.element.addEventListener('pointerdown', (event) => {
            this.context.view.beginGraphEditNodeDrag(this, event);
        });
        this.element.addEventListener('click', (event) => {
            this.context.view.graphEditGraphInputMenu(this.value, event);
        });
        this._graphEditOutputPorts.build(document, this.element);
    }

    update() {
        super.update();
        this._graphEditOutputPorts.update();
    }

    edge(to, value) {
        this._edges = this._edges || new Map();
        if (value) {
            return new view.Edge(this, to);
        }
        if (!this._edges.has(to)) {
            this._edges.set(to, new view.Edge(this, to));
        }
        return this._edges.get(to);
    }
};

view.Output = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        if (Array.isArray(value.value)) {
            const types = value.value.map((argument) => argument.type || '').join('\n');
            let name = value.name || '';
            if (name.length > 16) {
                name = name.split('/').pop();
            }
            const header = this.header();
            const title = header.add(null, ['graph-item-output']);
            title.content = name;
            title.tooltip = types;
            title.on('click', () => this.context.view.showTargetProperties(this.target));
        }
        this._graphEditInputPorts = new view.GraphEditGraphOutputPorts(this);
    }

    get target() {
        return this.context.target === this.context.view.activeTarget ? null : this.context.target;
    }

    get class() {
        return 'graph-output';
    }

    get inputs() {
        return [this.value];
    }

    get outputs() {
        return [];
    }

    activate() {
        this.context.view.showTargetProperties(this.target);
    }

    build(document, parent) {
        super.build(document, parent);
        this.element.addEventListener('pointerdown', (event) => {
            this.context.view.beginGraphEditNodeDrag(this, event);
        });
        this.element.addEventListener('click', (event) => {
            this.context.view.graphEditGraphOutputMenu(this.value, event);
        });
        this._graphEditInputPorts.build(document, this.element);
    }

    update() {
        super.update();
        this._graphEditInputPorts.update();
    }
};

view.Value = class {

    constructor(context, value) {
        this.context = context;
        this.value = value;
        this.from = null;
        this.to = [];
    }

    controlDependency(node) {
        this._controlDependencies = this._controlDependencies || new Set();
        this._controlDependencies.add(this.to.length);
        this.to.push(node);
    }

    build() {
        this._edges = this._edges || [];
        const bundle = this.context.edgeBundle(this);
        if (this.from && Array.isArray(this.to)) {
            for (let i = 0; i < this.to.length; i++) {
                const to = this.to[i];
                const bundleIndex = bundle ? bundle.members.findIndex((member) =>
                    member.value === this && member.toIndex === i) : -1;
                const bundleLeader = bundle && bundleIndex === bundle.representativeIndex;
                // A bundle is one visual/layout edge backed by many logical
                // ONNX connections. Keep those connections in bundle.members
                // instead of creating hidden SVG and Dagre edges for them.
                if (bundle && !bundleLeader) {
                    continue;
                }
                let content = '';
                const type = this.value.type;
                if (type &&
                    type.shape &&
                    type.shape.dimensions &&
                    type.shape.dimensions.length > 0 &&
                    type.shape.dimensions.every((dim) => !dim || Number.isInteger(dim) || typeof dim === 'bigint' || (typeof dim === 'string'))) {
                    content = type.shape.dimensions.map((dim) => (dim !== null && dim !== undefined && dim !== -1 && dim !== -1n) ? dim : '?').join('\u00D7');
                    content = content.length > 16 ? '' : content;
                }
                if (this.context.options.names) {
                    content = this.value.name.split('\n').shift(); // custom argument id
                }
                const encoding = this.context.model.attachment.quantization.precision(this.value);
                if (encoding) {
                    const label = encoding.inferred ? `~${encoding.label}` : encoding.label;
                    content = content ? `${content} · ${label}` : label;
                }
                const edge = this.from.edge(to, this);
                if (!edge.value) {
                    edge.value = this;
                    if (to instanceof view.Node) {
                        const occurrence = this.to.slice(0, i).filter((item) => item === to).length;
                        const targets = [];
                        for (const argument of to.value.inputs || []) {
                            for (let valueIndex = 0; valueIndex < (argument.value || []).length; valueIndex++) {
                                if (argument.value[valueIndex] === this.value) {
                                    targets.push({
                                        node: to.value,
                                        argument,
                                        valueIndex,
                                        oldValue: this.value,
                                        port: null
                                    });
                                }
                            }
                        }
                        edge.graphEditTarget = targets[occurrence] || null;
                    } else if (to instanceof view.Output) {
                        const argument = to.value;
                        const valueIndex = (argument.value || []).findIndex((value) =>
                            value === this.value || value && value.name === this.value.name);
                        edge.graphEditTarget = valueIndex >= 0 ? {
                            graphOutput: true,
                            argument,
                            valueIndex,
                            oldValue: this.value,
                            port: null
                        } : null;
                    }
                    if (content) {
                        edge.label = content;
                    }
                    edge.id = i === 0 ? `edge-${this.value.name}` : `edge-${this.value.name}-${i}`;
                    const classes = [];
                    if (this._controlDependencies && this._controlDependencies.has(i)) {
                        classes.push('edge-path-control-dependency');
                    }
                    if (encoding && encoding.inferred) {
                        classes.push('edge-path-precision-inferred');
                        edge.labelClass = 'edge-label-precision-inferred';
                    }
                    if (bundle) {
                        edge.bundle = bundle;
                        bundle.edge = edge;
                        edge.label = bundle.label;
                        edge.labelClass = 'edge-label-bundle';
                        classes.push('edge-path-bundle');
                    }
                    if (classes.length > 0) {
                        edge.class = classes.join(' ');
                    }
                }
                this.context.setEdge(edge);
                this._edges.push(edge);
            }
        }
    }

    select() {
        let array = [];
        if (Array.isArray(this._edges)) {
            for (const edge of this._edges) {
                array = array.concat(edge.select());
            }
        }
        return array;
    }

    deselect() {
        if (Array.isArray(this._edges)) {
            for (const edge of this._edges) {
                edge.deselect();
            }
        }
    }

    activate(source) {
        if (this.value && this.from && Array.isArray(this.to) && !this.value.initializer) {
            const from = this.from.value;
            const to = this.to.map((node) => node.value);
            this.context.view.showConnectionProperties(this.value, from, to, source);
        } else if (this.value && this.value.initializer) {
            this.context.view.showTensorProperties({ value: [this.value] }, source);
        }
    }
};

view.Argument = class extends grapher.Argument {

    constructor(context, value) {
        const name = value.name;
        let content = '';
        let separator = '';
        let tooltip = '';
        if (Array.isArray(value.value) && value.value.length === 1 && value.value[0].initializer) {
            const tensor = value.value[0].initializer;
            const type = value.value[0].type;
            tooltip = type.toString();
            content = view.Formatter.tensor(tensor);
            const encoding = context.model.attachment.quantization.value(value.value[0]);
            if (encoding) {
                content = `${content} · ${encoding.label}`;
            }
            if (!content.startsWith('\u3008')) {
                separator = ' = ';
            }
        }
        super(name, content);
        this.context = context;
        this.value = value;
        this.separator = separator;
        this.tooltip = tooltip;
    }

    focus() {
        this.context.focus([this.value]);
    }

    blur() {
        this.context.blur([this.value]);
    }

    activate(source) {
        this.context.view.showTensorProperties(this.value, source);
    }
};

view.Edge = class extends grapher.Edge {

    constructor(from, to) {
        super(from, to);
        this.v = from.name;
        this.w = to.name;
        view.Edge.counter = (view.Edge.counter || 0) + 1;
        this.name = view.Edge.counter.toString();
    }

    get minlen() {
        if (this.from.inputs.every((argument) => (!argument.type || argument.type.endsWith('*')) && argument.value.every((value) => value.initializer))) {
            return 2;
        }
        return 1;
    }

    focus() {
        if (this.bundle) {
            this.value.context.focus(this.bundle.values.map((value) => value.value));
        } else {
            this.value.context.focus([this.value.value]);
        }
    }

    blur() {
        if (this.bundle) {
            this.value.context.blur(this.bundle.values.map((value) => value.value));
        } else {
            this.value.context.blur([this.value.value]);
        }
    }

    activate() {
        const graphEdit = this.value.context.view.graphEditConnection(this);
        if (!this.value.context.view.graphEditEnabled) {
            this.value.context.activate(this.value.value, 'target');
        } else if (graphEdit && typeof graphEdit.catch === 'function') {
            graphEdit.catch((error) => this.value.context.view.error(error, 'ONNX GraphSurgeon failed.', null));
        }
    }
};

view.Sidebar = class {

    constructor(host) {
        this._host = host;
        this._stack = [];
        this._width = this._host.get('sidebar-width');
        this._width = Number.isFinite(this._width) ? this._width : null;
        this._closeSidebarHandler = () => this.close();
        this._closeSidebarKeyDownHandler = (e) => {
            if (e.keyCode === 27) { // Escape
                e.stopPropagation();
                e.preventDefault();
                this.close();
            }
        };
        const sidebar = this._element('sidebar');
        const resize = this._element('sidebar-resize');
        this._resizeSidebarPointerDownHandler = (event) => {
            if (event.target !== resize) {
                return;
            }
            event.preventDefault();
            const startX = event.clientX;
            const startWidth = sidebar.getBoundingClientRect().width;
            resize.classList.add('active');
            sidebar.classList.add('resizing');
            this._resizeSidebarPointerMoveHandler = (event) => {
                const width = startWidth + startX - event.clientX;
                this._setWidth(width);
            };
            this._resizeSidebarPointerUpHandler = () => {
                resize.classList.remove('active');
                sidebar.classList.remove('resizing');
                this._host.document.removeEventListener('pointermove', this._resizeSidebarPointerMoveHandler, true);
                this._host.document.removeEventListener('pointerup', this._resizeSidebarPointerUpHandler, true);
                this._host.document.removeEventListener('pointercancel', this._resizeSidebarPointerUpHandler, true);
                this._host.set('sidebar-width', this._width);
            };
            this._host.document.addEventListener('pointermove', this._resizeSidebarPointerMoveHandler, true);
            this._host.document.addEventListener('pointerup', this._resizeSidebarPointerUpHandler, true);
            this._host.document.addEventListener('pointercancel', this._resizeSidebarPointerUpHandler, true);
        };
        this._host.document.addEventListener('pointerdown', this._resizeSidebarPointerDownHandler, true);
        const window = this._host.document.defaultView;
        if (window) {
            window.addEventListener('resize', () => {
                if (this._stack.length > 0) {
                    this._setWidth(this._width);
                }
            });
        }
        sidebar.addEventListener('transitionend', (event) => {
            if (event.propertyName === 'opacity' && sidebar.style.opacity === '0') {
                const content = this._element('sidebar-content');
                content.replaceChildren();
            }
        });
    }

    _element(id) {
        return this._host.document.getElementById(id);
    }

    open(content, title, source) {
        const element = this._render(content);
        const entry = { title, element, content };
        if (source === 'sidebar') {
            const depth = 10;
            this._update(this._stack.concat(entry).slice(-depth));
        } else {
            this._update([entry]);
        }
    }

    close() {
        this._update([]);
    }

    get identifier() {
        if (this._stack.length > 0) {
            const content = this._stack[this._stack.length - 1].content;
            if (content.identifier) {
                return content.identifier;
            }
        }
        return '';
    }

    _render(content) {
        try {
            content.render();
        } catch (error) {
            content.error(error, false);
        }
        const element = content.element;
        return Array.isArray(element) ? element : [element];
    }

    _setWidth(width) {
        const document = this._host.document;
        const window = document.defaultView;
        const viewport = window ? window.innerWidth : document.documentElement.clientWidth;
        const minimum = Math.min(280, viewport);
        const maximum = Math.max(minimum, Math.floor(viewport * 0.85));
        const fallback = Math.min(viewport * 0.6, 504);
        this._width = Math.round(Math.min(maximum, Math.max(minimum, Number.isFinite(width) ? width : fallback)));
        this._element('sidebar').style.width = `${this._width}px`;
        this._element('target').style.width = `${Math.max(0, viewport - this._width)}px`;
    }

    _update(stack) {
        const sidebar = this._element('sidebar');
        const element = this._element('sidebar-content');
        const container = this._element('target');
        const closeButton = this._element('sidebar-closebutton');
        closeButton.removeEventListener('click', this._closeSidebarHandler);
        this._host.document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
        if (this._stack.length > 0) {
            const entry = this._stack.pop();
            const content = entry.content;
            if (content && content.deactivate) {
                content.deactivate();
            }
        }
        if (stack) {
            this._stack = stack;
        }
        if (this._stack.length > 0) {
            const entry = this._stack[this._stack.length - 1];
            this._element('sidebar-title').innerHTML = entry.title || '';
            closeButton.addEventListener('click', this._closeSidebarHandler);
            if (typeof entry.content === 'string') {
                element.innerHTML = entry.element;
            } else if (entry.element instanceof Array) {
                element.replaceChildren(...entry.element);
            } else {
                element.replaceChildren(entry.element);
            }
            this._setWidth(this._width);
            sidebar.style.right = 0;
            sidebar.style.opacity = 1;
            this._host.document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
            const content = entry.content;
            if (content && content.activate) {
                content.activate();
            }
        } else {
            sidebar.style.right = `${0 - (this._width || sidebar.getBoundingClientRect().width)}px`;
            sidebar.style.opacity = 0;
            const clone = element.cloneNode(true);
            element.parentNode.replaceChild(clone, element);
            container.style.width = '100%';
            container.focus();
        }
    }
};

view.Control = class {

    constructor(context) {
        this._view = context;
        this._host = context.host;
    }

    createElement(tagName, className) {
        const element = this._host.document.createElement(tagName);
        if (className) {
            element.setAttribute('class', className);
        }
        return element;
    }

    createTextNode(data) {
        const node = this._host.document.createTextNode(data);
        return node;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        try {
            if (this._events && this._events[event]) {
                for (const callback of this._events[event]) {
                    callback(this, data);
                }
            }
        } catch (error) {
            this.error(error, false);
        }
    }

    error(error, fatal) {
        this._view.exception(error, fatal || false);
    }

    escape(value) {
        return value.toString().split('&').join('&amp;').split('<').join('&lt;').split('>').join('&gt;');
    }
};

view.Expander = class extends view.Control {

    constructor(context) {
        super(context);
        this.element = this.createElement('div', 'sidebar-item-value');
        this._count = -1;
    }

    render() {
        return [this.element];
    }

    expandable() {
        if (!this._expander) {
            this._expander = this.createElement('div', 'sidebar-item-value-expander');
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => this.toggle());
            this.add(this._expander);
        }
    }

    add(element) {
        this.element.appendChild(element);
    }

    control(element) {
        this.add(element);
    }

    toggle() {
        this._count = this._count === -1 ? this.element.childElementCount : this._count;
        if (this._expander) {
            while (this.element.childElementCount > this._count) {
                this.element.removeChild(this.element.lastChild);
            }
            if (this._expander.innerText === '+') {
                this._expander.innerText = '-';
                this.expand();
            } else {
                this._expander.innerText = '+';
                this.collapse();
            }
        }
    }

    expand() {
    }

    collapse() {
    }
};

view.TargetSelector = class extends view.Control {

    constructor(context, element) {
        super(context);
        this._element = element;
        [this._select] = element.getElementsByTagName('select');
        this._select.addEventListener('change', (e) => {
            const target = this._targets[e.target.selectedIndex];
            this.emit('change', target);
        });
        this._targets = [];
    }

    update(model, stack) {
        while (this._select.firstChild) {
            this._select.removeChild(this._select.firstChild);
        }
        this._targets = [];
        const current = stack.length > 0 ? stack[stack.length - 1] : null;
        const section = (title, targets) => {
            if (targets.length > 0) {
                const group = this.createElement('optgroup');
                group.setAttribute('label', title);
                this._select.appendChild(group);
                for (let i = 0; i < targets.length; i++) {
                    const target = targets[i];
                    const option = this.createElement('option');
                    option.innerText = target.name;
                    group.appendChild(option);
                    if (current && current.target === target.target && current.signature === target.signature) {
                        option.setAttribute('selected', 'true');
                        this._select.setAttribute('title', target.name);
                    }
                    this._targets.push(target);
                }
            }
        };
        const modules = [];
        const signatures = [];
        const functions = [];
        if (model && Array.isArray(model.modules)) {
            for (const graph of model.modules) {
                const name = graph.name || '-';
                modules.push({ name, target: graph, signature: null });
                if (Array.isArray(graph.functions)) {
                    for (const func of graph.functions) {
                        functions.push({ name: `${name}.${func.name}`, target: func, signature: null });
                    }
                }
                if (Array.isArray(graph.signatures)) {
                    for (const signature of graph.signatures) {
                        signatures.push({ name: `${name}.${signature.name}`, target: graph, signature });
                    }
                }
            }
        }
        if (model && Array.isArray(model.functions)) {
            for (const func of model.functions) {
                functions.push({ name: func.name, target: func, signature: null });
            }
        }
        section('Modules', modules);
        section('Signatures', signatures);
        section('Functions', functions);
        const visible = functions.length > 0 || signatures.length > 0 || modules.length > 1;
        this._element.style.display = visible ? 'inline' : 'none';
    }
};

view.ObjectSidebar = class extends view.Control {

    constructor(context) {
        super(context);
        this.element = this.createElement('div', 'sidebar-object');
    }

    addSection(title) {
        const element = this.createElement('div', 'sidebar-section');
        element.innerText = title;
        this.element.appendChild(element);
    }

    addEntry(name, item) {
        const entry = new view.NameValueView(this._view, name, item);
        const element = entry.render();
        this.element.appendChild(element);
    }

    addProperty(name, value, style) {
        const item = new view.TextView(this._view, value, style);
        this.addEntry(name, item);
        return item;
    }

    addArgument(name, argument, source) {
        const value = new view.ArgumentView(this._view, argument, source);
        value.on('focus', (sender, value) => {
            this.emit('focus', value);
            this._focused = this._focused || new Set();
            this._focused.add(value);
        });
        value.on('blur', (sender, value) => {
            this.emit('blur', value);
            this._focused = this._focused || new Set();
            this._focused.delete(value);
        });
        value.on('select', (sender, value) => this.emit('select', value));
        value.on('activate', (sender, value) => this.emit('activate', value));
        value.on('deactivate', (sender, value) => this.emit('deactivate', value));
        this.addEntry(name, value);
        return value;
    }

    addQuantization(entry, title, includeName, section) {
        if (!entry) {
            return;
        }
        if (section !== false) {
            this.addSection(title || 'Quantization');
        }
        if (includeName) {
            this.addProperty('tensor', entry.name, 'nowrap');
        }
        this.addProperty('encoding', entry.label);
        this.addProperty('data type', entry.dataType, 'code');
        this.addProperty('granularity', entry.granularity);
        if (entry.axis !== null) {
            this.addProperty('axis', entry.axis, 'code');
        }
        if (entry.blockSize !== null) {
            this.addProperty('block size', entry.blockSize, 'code');
        }
        if (entry.symmetric !== null) {
            this.addProperty('symmetric', entry.symmetric ? 'true' : 'false');
        }
        const fields = [
            ['scale', entry.scale],
            [entry.zeroPointLabel, entry.zeroPoint],
            ['min', entry.min],
            ['max', entry.max]
        ];
        for (const [name, value] of fields) {
            const content = aimet.Utility.format(value);
            if (content) {
                this.addProperty(name, content, 'code');
            }
        }
    }

    addPrecision(entry, title, includeName, section) {
        if (!entry) {
            return;
        }
        if (section !== false) {
            this.addSection(title || 'Tensor Precision');
        }
        if (includeName) {
            this.addProperty('tensor', entry.name, 'nowrap');
        }
        this.addProperty('precision', entry.label);
        this.addProperty('data type', entry.dataType, 'code');
        const source = entry.inferred ? `propagated from ${entry.inferredFrom}` : 'explicit tensor encoding';
        this.addProperty('source', source, 'nowrap');
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('span');
        const title = this.createElement('b');
        title.textContent = 'ERROR: ';
        element.appendChild(title);
        const message = this.createTextNode(` ${error.message}`);
        element.appendChild(message);
        this.element.appendChild(element);
    }
};

view.NodeSidebar = class extends view.ObjectSidebar {

    constructor(context, node) {
        super(context);
        this._node = node;
    }

    get identifier() {
        return 'node';
    }

    render() {
        const node = this._node;
        if (node.type) {
            const type = node.type;
            const item = this.addProperty('type', node.type.identifier || node.type.name);
            if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
                let icon = '?';
                let tooltip = 'Show Definition';
                if (type.type === 'weights') {
                    icon = '\u25CF';
                    tooltip = 'Show Weights';
                } else if (Array.isArray(type.nodes)) {
                    icon = '\u0192';
                }
                item.action(icon, tooltip, () => {
                    this.emit('show-definition', null);
                });
            }
            const module = node.type.module;
            const version = node.type.version;
            const status = node.type.status;
            if (module || version || status) {
                const list = [module, version ? `v${version}` : '', status];
                const value = list.filter((value) => value).join(' ');
                this.addProperty('module', value, 'nowrap');
            }
        }
        if (node.name) {
            this.addProperty('name', node.name, 'nowrap');
        }
        if (node.identifier) {
            this.addProperty('identifier', node.identifier, 'nowrap');
        }
        if (node.description) {
            this.addProperty('description', node.description);
        }
        if (node.device) {
            this.addProperty('device', node.device);
        }
        const quantization = this._view.model.attachment.quantization;
        const precision = quantization.node(node);
        for (let i = 0; i < precision.inputs.length; i++) {
            this.addPrecision(precision.inputs[i], 'Input Tensor Precision', true, i === 0);
        }
        const encodings = quantization.node(node, false);
        const qparams = [
            ['Parameter QParams', encodings.parameters],
            ['Output QParams', encodings.outputs]
        ];
        for (const [title, entries] of qparams) {
            for (let i = 0; i < entries.length; i++) {
                this.addQuantization(entries[i], title, true, i === 0);
            }
        }
        const attributes = node.attributes;
        if (Array.isArray(attributes) && attributes.length > 0) {
            this.addSection('Attributes');
            attributes.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
            for (const attribute of attributes) {
                this.addArgument(attribute.name, attribute, 'attribute');
            }
        }
        const inputs = node.inputs;
        if (Array.isArray(inputs) && inputs.length > 0) {
            this.addSection('Inputs');
            for (const input of inputs) {
                const name = input.name;
                this.addArgument(name, input);
            }
        }
        const outputs = node.outputs;
        if (Array.isArray(outputs) && outputs.length > 0) {
            this.addSection('Outputs');
            for (const output of outputs) {
                const name = output.name;
                this.addArgument(name, output);
            }
        }
        const blocks = node.blocks;
        if (Array.isArray(blocks) && blocks.length > 0) {
            this.addSection('Blocks');
            for (const block of blocks) {
                const name = block.name;
                this.addArgument(name, block);
            }
        }
        const metadata = this._view.model.attachment.metadata.node(node);
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addSection('Metadata');
            for (const argument of metadata) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
        const metrics = this._view.model.attachment.metrics.node(node);
        if (Array.isArray(metrics) && metrics.length > 0) {
            this.addSection('Metrics');
            for (const argument of metrics) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
    }

    activate() {
        this.emit('select', this._node);
    }

    deactivate() {
        this.emit('select', null);
        if (this._focused) {
            for (const value of this._focused) {
                this.emit('blur', value);
            }
            this._focused.clear();
        }
    }
};

view.NameValueView = class extends view.Control {

    constructor(context, name, value) {
        super(context);
        this._name = name;
        this._value = value;
        const nameElement = this.createElement('div', 'sidebar-item-name');
        const input = this.createElement('input');
        input.setAttribute('type', 'text');
        input.setAttribute('value', name);
        input.setAttribute('title', name);
        input.setAttribute('readonly', 'true');
        nameElement.appendChild(input);
        const valueElement = this.createElement('div', 'sidebar-item-value-list');
        for (const element of value.render()) {
            valueElement.appendChild(element);
        }
        this.element = this.createElement('div', 'sidebar-item');
        this.element.appendChild(nameElement);
        this.element.appendChild(valueElement);
    }

    get name() {
        return this._name;
    }

    render() {
        return this.element;
    }

    toggle() {
        this._value.toggle();
    }
};

view.TextView = class extends view.Control {

    constructor(context, value, style) {
        super(context);
        this.element = this.createElement('div', 'sidebar-item-value');
        let className = 'sidebar-item-value-line';
        if (value !== null && value !== undefined) {
            const list = Array.isArray(value) ? value : [value];
            for (const item of list) {
                const line = this.createElement('div', className);
                switch (style) {
                    case 'code': {
                        const element = this.createElement('code');
                        element.textContent = item;
                        line.appendChild(element);
                        break;
                    }
                    case 'bold': {
                        const element = this.createElement('b');
                        element.textContent = item;
                        line.appendChild(element);
                        break;
                    }
                    case 'nowrap': {
                        line.innerText = item;
                        line.style.whiteSpace = style;
                        break;
                    }
                    default: {
                        line.innerText = item;
                        break;
                    }
                }
                this.element.appendChild(line);
                className = 'sidebar-item-value-line-border';
            }
        } else {
            const line = this.createElement('div', className);
            line.classList.add('sidebar-item-disable-select');
            line.innerHTML = '&nbsp';
            this.element.appendChild(line);
        }
    }

    action(text, description, callback) {
        const action = this.createElement('div', 'sidebar-item-value-expander');
        action.setAttribute('title', description);
        action.addEventListener('click', () => callback());
        action.innerHTML = text;
        this.element.insertBefore(action, this.element.childNodes[0]);
    }

    render() {
        return [this.element];
    }

    toggle() {
    }
};

view.ArgumentView = class extends view.Control {

    constructor(context, argument, source) {
        super(context);
        this._argument = argument;
        this._source = source;
        this._elements = [];
        this._items = [];
        const type = argument.type === 'attribute' ? null : argument.type;
        let value = argument.value;
        if (argument.type === 'attribute') {
            this._source = 'attribute';
        }
        if (argument.type === 'tensor' || argument.type === 'tensor?') {
            if (value === null || (value && value.constructor && value.constructor.name === 'Value')) {
                value = [value];
            } else {
                value = [{ type: value.type, initializer: value }];
            }
        } else if (argument.type === 'tensor[]' || argument.type === 'tensor?[]') {
            value = value.map((value) => {
                if (value === null || (value && value.constructor && value.constructor.name === 'Value')) {
                    return value;
                }
                return { type: value.type, initializer: value };
            });
        }
        this._source = typeof type === 'string' && !type.endsWith('*') ? 'attribute' : this._source;
        const primitive = typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint';
        if (primitive) {
            const item = new view.PrimitiveView(context, argument);
            this._items.push(item);
        } else if (this._source === 'attribute' && type !== 'tensor' && type !== 'tensor?' && type !== 'tensor[]' && type !== 'tensor?[]') {
            this._source = 'attribute';
            const item = new view.PrimitiveView(context, argument);
            this._items.push(item);
        } else if (Array.isArray(value) && value.length === 0) {
            const item = new view.TextView(this._view, null);
            this._items.push(item);
        } else {
            const values = value;
            for (const value of values) {
                const emit = values.length === 1 && value && value.initializer;
                const target = emit ? argument : value;
                if (value === null) {
                    const item = new view.TextView(this._view, null);
                    this._items.push(item);
                } else {
                    const item = new view.ValueView(context, value, this._source);
                    item.on('focus', () => this.emit('focus', target));
                    item.on('blur', () => this.emit('blur', target));
                    item.on('activate', () => this.emit('activate', target));
                    item.on('select', () => this.emit('select', target));
                    this._items.push(item);
                }
            }
        }
        for (const item of this._items) {
            this._elements.push(...item.render());
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
        for (const item of this._items) {
            item.toggle();
        }
    }
};

view.PrimitiveView = class extends view.Expander {

    constructor(context, argument) {
        super(context);
        try {
            this._argument = argument;
            const type = argument.type === 'attribute' ? null : argument.type;
            const value = argument.value;
            if (type) {
                this.expandable();
            }
            switch (type) {
                case 'graph': {
                    const line = this.createElement('div', 'sidebar-item-value-line-link');
                    line.textContent = value ? value.name || '\u00A0' : '(null)';
                    if (value) {
                        line.addEventListener('click', () => this.emit('activate', value));
                    }
                    this.add(line);
                    break;
                }
                case 'function': {
                    const line = this.createElement('div', 'sidebar-item-value-line-link');
                    line.textContent = value.name;
                    line.addEventListener('click', () => this.emit('activate', value));
                    this.add(line);
                    break;
                }
                case 'object[]': {
                    for (const obj of argument.value) {
                        const line = this.createElement('div', 'sidebar-item-value-line');
                        line.textContent = obj.type ? obj.type.name : '?';
                        this.add(line);
                    }
                    break;
                }
                default: {
                    const formatter = new view.Formatter(value, type);
                    let content = formatter.toString();
                    if (content) {
                        if (content.length > 2000) {
                            content = `${content.substring(0, 2000)}\u2026`;
                        }
                        const multiline = content.includes('\n');
                        if (!multiline && content.length > 80) {
                            this.expandable();
                        }
                        content = this.escape(content);
                        if (multiline) {
                            content = content.split('\n').join('<br>');
                        }
                    }
                    this._line = this.createElement('div', 'sidebar-item-value-line');
                    this._line.innerHTML = content ? content : '&nbsp;';
                    this.add(this._line);
                }
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    expand() {
        try {
            if (this._line) {
                this._line.classList.add('sidebar-item-value-line-wrap');
            }
            const type = this._argument.type;
            const value = this._argument.value;
            let content = type === 'tensor' && value && value.type ? value.type.toString() : this._argument.type;
            if (content) {
                content = this.escape(content);
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.innerHTML = `type: <code><b>${content}</b></code>`;
                this.add(line);
            }
            const description = this._argument.description;
            if (description) {
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.innerHTML = this.escape(description);
                this.add(line);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    collapse() {
        if (this._line) {
            this._line.classList.remove('sidebar-item-value-line-wrap');
        }
    }

    _info(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `<b>${name}:</b> ${this.escape(value)}`;
        this._add(line);
    }

    _add(child) {
        child.className = this._first === false ? 'sidebar-item-value-line-border' : 'sidebar-item-value-line';
        this.add(child);
        this._first = false;
    }
};

view.ValueView = class extends view.Expander {

    constructor(context, value, source) {
        super(context);
        this._value = value;
        try {
            if (value && value.constructor && value.constructor.name === 'Value' && source === 'attribute') {
                source = '';
            }
            const type = this._value.type;
            const initializer = this._value.initializer;
            const quantization = this._value.quantization;
            const location = this._value.location !== undefined;
            if (initializer) {
                this.element.classList.add('sidebar-item-value-content');
            }
            if (type || initializer || quantization || location || source === 'attribute') {
                this.expandable();
            }
            if (initializer && source !== 'attribute') {
                const element = this.createElement('div', 'sidebar-item-value-button');
                element.classList.add('sidebar-item-value-button-tool');
                element.setAttribute('title', 'Show Tensor');
                element.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-weight"></use></svg>`;
                element.addEventListener('pointerenter', () => this.emit('focus', this._value));
                element.addEventListener('pointerleave', () => this.emit('blur', this._value));
                element.style.cursor = 'pointer';
                element.addEventListener('click', () => this.emit('activate', this._value));
                this.control(element);
            }
            const name = this._value.name ? this._value.name.split('\n').shift() : ''; // custom argument id
            this._hasId = name && source !== 'attribute' ? true : false;
            this._hasCategory = initializer && initializer.category && source !== 'attribute' ? true : false;
            if (this._hasId || (!this._hasCategory && !type && source !== 'attribute')) {
                this._hasId = true;
                const element = this.createElement('div', 'sidebar-item-value-line');
                if (typeof name !== 'string') {
                    throw new Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
                }
                const text = this.createElement('b');
                text.innerText = name || ' ';
                const line = this.createElement('span', 'sidebar-item-value-line-content');
                line.innerText = 'name: ';
                line.appendChild(text);
                element.appendChild(line);
                element.addEventListener('pointerenter', () => this.emit('focus', this._value));
                element.addEventListener('pointerleave', () => this.emit('blur', this._value));
                element.style.cursor = 'pointer';
                element.addEventListener('click', () => this.emit('activate', this._value));
                this._add(element);
            } else if (this._hasCategory) {
                this._bold('category', initializer.category);
            } else if (type) {
                this._code('tensor', type);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    render() {
        return [this.element];
    }

    expand() {
        try {
            const initializer = this._value.initializer;
            if (this._hasId && this._hasCategory) {
                this._bold('category', initializer.category);
            }
            let type = null;
            let denotation = null;
            if (this._value.type) {
                type = this._value.type.toString();
                denotation = this._value.type.denotation || null;
            }
            if (type && (this._hasId || this._hasCategory)) {
                this._code('tensor', type);
            }
            if (denotation) {
                this._code('denotation', denotation);
            }
            const description = this._value.description;
            if (description) {
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.innerHTML = this.escape(description);
                this.add(line);
            }
            const identifier = this._value.identifier;
            if (identifier !== undefined) {
                this._bold('identifier', identifier);
            }
            const layout = this._value.type ? this._value.type.layout : null;
            if (layout) {
                this._bold('layout', layout.replace('.', ' '));
            }
            const quantization = this._value.quantization;
            if (quantization) {
                if (typeof quantization.type !== 'string') {
                    throw new view.Error('Unsupported quantization value.');
                }
                const value = new view.Quantization(quantization).toString();
                if (quantization.type && (quantization.type !== 'linear' || (value && value !== 'q'))) {
                    const line = this.createElement('div', 'sidebar-item-value-line-border');
                    const content = [
                        `<span class='sidebar-item-value-line-content'>quantization: <b>${quantization.type}</b></span>`
                    ];
                    if (value) {
                        content.push(`<pre style='margin: 4px 0 2px 0'>${value}</pre>`);
                    }
                    line.innerHTML = content.join('');
                    this._add(line);
                }
            }
            if (initializer) {
                if (initializer.location) {
                    this._bold('location', initializer.location);
                }
                const stride = initializer.stride;
                if (Array.isArray(stride) && stride.length > 0) {
                    this._code('stride', stride.join(','));
                }
                const tensor = new view.TensorView(this._view, initializer);
                const content = tensor.content;
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.appendChild(content);
                this._add(line);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    _bold(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `${name}: <b>${this.escape(value)}</b>`;
        this._add(line);
    }

    _code(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `${name}: <code><b>${this.escape(value)}</b></code>`;
        this._add(line);
    }

    _info(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `<b>${name}:</b> ${this.escape(value)}`;
        this._add(line);
    }

    _add(child) {
        child.className = this._first === false ? 'sidebar-item-value-line-border' : 'sidebar-item-value-line';
        this.add(child);
        this._first = false;
    }
};

view.TensorView = class extends view.Expander {

    constructor(context, value, tensor) {
        super(context);
        this._value = value;
        this._tensor = tensor || new base.Tensor(value);
    }

    render() {
        if (!this._button) {
            this.expandable();
            this._button = this.createElement('div', 'sidebar-item-value-button');
            this._button.setAttribute('style', 'float: left;');
            this._button.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-weight"></use></svg>`;
            this._button.addEventListener('click', () => this.toggle());
            this.control(this._button);
            const line = this.createElement('div', 'sidebar-item-value-line');
            line.classList.add('sidebar-item-disable-select');
            line.innerHTML = '&nbsp';
            this.element.appendChild(line);
        }
        return super.render();
    }

    expand() {
        try {
            const content = this.content;
            const container = this.createElement('div', 'sidebar-item-value-line-border');
            container.appendChild(content);
            this.element.appendChild(container);
        } catch (error) {
            this.error(error, false);
        }
    }

    get content() {
        const content = this.createElement('pre');
        const value = this._value;
        const tensor = this._tensor;
        if (tensor.encoding !== '<' && tensor.encoding !== '>' && tensor.encoding !== '|') {
            content.innerHTML = `Tensor encoding '${tensor.layout}' is not implemented.`;
        } else if (tensor.layout && (tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo')) {
            content.innerHTML = `Tensor layout '${tensor.layout}' is not implemented.`;
        } else if (tensor.type && tensor.type.dataType === '?') {
            content.innerHTML = 'Tensor data type is not defined.';
        } else if (tensor.type && !tensor.type.shape) {
            content.innerHTML = 'Tensor shape is not defined.';
        } else {
            content.innerHTML = '&#x23F3';
            const promise = value.peek && !value.peek() ? value.read() : Promise.resolve();
            promise.then(() => {
                if (tensor.empty) {
                    content.innerHTML = 'Tensor data is empty.';
                } else {
                    content.innerHTML = tensor.toString();
                    if (this._host.save && value.type.shape && value.type.shape.dimensions && value.type.shape.dimensions.length > 0) {
                        this._saveButton = this.createElement('div', 'sidebar-item-value-button');
                        this._saveButton.classList.add('sidebar-item-value-button-context');
                        this._saveButton.setAttribute('style', 'float: right;');
                        this._saveButton.innerHTML = '&#x1F4BE;';
                        this._saveButton.addEventListener('click', async () => {
                            await this.export();
                        });
                        content.insertBefore(this._saveButton, content.firstChild);
                    }
                }
            }).catch((error) => {
                content.innerHTML = error.message;
            });
        }
        return content;
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('div', 'sidebar-item-value-line');
        const title = this.createElement('b');
        title.textContent = 'ERROR: ';
        element.appendChild(title);
        const message = this.createTextNode(error.message);
        element.appendChild(message);
        this.element.appendChild(element);
    }

    async export() {
        const window = this._host.window;
        const tensor = this._tensor;
        const defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
        const file = await this._host.save('NumPy Array', 'npy', defaultPath);
        if (file) {
            try {
                let data_type = '?';
                switch (tensor.type.dataType) {
                    case 'boolean': data_type = 'bool'; break;
                    case 'bfloat16': data_type = 'float32'; break;
                    case 'float4e2m1fn': data_type = 'float16'; break;
                    case 'float6e2m3fn': data_type = 'float16'; break;
                    case 'float6e3m2fn': data_type = 'float16'; break;
                    case 'float8e3m4': data_type = 'float16'; break;
                    case 'float8e4m3': data_type = 'float16'; break;
                    case 'float8e4m3b11fnuz': data_type = 'float16'; break;
                    case 'float8e4m3fn': data_type = 'float16'; break;
                    case 'float8e4m3fnuz': data_type = 'float16'; break;
                    case 'float8e5m2': data_type = 'float16'; break;
                    case 'float8e5m2fnuz': data_type = 'float16'; break;
                    case 'float8e8m0fnu': data_type = 'float16'; break;
                    case 'float8e8m0': data_type = 'float16'; break;
                    case 'int4': data_type = 'int8'; break;
                    case 'int48': data_type = 'int64'; break;
                    case 'quint8': data_type = 'uint8'; break;
                    default: data_type = tensor.type.dataType; break;
                }
                const python = await import('./python.js');
                const execution = new python.Execution();
                const io = execution.__import__('io');
                const numpy = execution.register('numpy');
                const bytes = new io.BytesIO();
                const dtype = new numpy.dtype(data_type);
                const array = numpy.asarray(tensor.value, dtype);
                numpy.save(bytes, array);
                bytes.seek(0);
                const blob = new window.Blob([bytes.read()], { type: 'application/octet-stream' });
                await this._host.export(file, blob);
            } catch (error) {
                this._view.error(error, 'Error saving NumPy tensor.', null);
            }
        }
    }
};

view.NodeView = class extends view.Expander {

    constructor(context, node) {
        super(context);
        this._node = node;
        const name = node.name;
        const type = node.type ? node.type.name : '';
        if (name && type) {
            this.expandable();
        }
        if (type) {
            const type = node.type.name;
            const element = this.createElement('div', 'sidebar-item-value-line');
            element.innerHTML = `<span class='sidebar-item-value-line-content'>node: <b>${this.escape(type || ' ')}</b></span>`;
            element.addEventListener('pointerenter', () => this.emit('focus', this._node));
            element.addEventListener('pointerleave', () => this.emit('blur', this._node));
            element.addEventListener('click', () => this.emit('activate', this._node));
            element.style.cursor = 'pointer';
            this.element.appendChild(element);
        } else {
            const element = this.createElement('div', 'sidebar-item-value-line');
            element.innerHTML = `<span class='sidebar-item-value-line-content'>name: <b>${this.escape(name || ' ')}</b></span>`;
            element.addEventListener('pointerenter', () => this.emit('focus', this._node));
            element.addEventListener('pointerleave', () => this.emit('blur', this._node));
            element.addEventListener('click', () => this.emit('activate', this._node));
            element.style.cursor = 'pointer';
            this.element.appendChild(element);
        }
    }

    expand() {
        const name = this._node.name;
        const element = this.createElement('div', 'sidebar-item-value-line-border');
        element.innerHTML = `<span class='sidebar-item-value-line-content'>name: <b>${this.escape(name)}</b></span>`;
        element.addEventListener('pointerenter', () => this.emit('focus', this._node));
        element.addEventListener('pointerleave', () => this.emit('blur', this._node));
        element.addEventListener('click', () => this.emit('activate', this._node));
        element.style.cursor = 'pointer';
        this.element.appendChild(element);
    }
};

view.NodeListView = class extends view.Control {

    constructor(context, list) {
        super(context);
        this._elements = [];
        for (const node of list) {
            const item = new view.NodeView(this._view, node);
            item.on('focus', (sender, value) => this.emit('focus', value));
            item.on('blur', (sender, value) => this.emit('blur', value));
            item.on('activate', (sender, value) => this.emit('activate', value));
            item.on('deactivate', (sender, value) => this.emit('deactivate', value));
            item.on('select', (sender, value) => this.emit('select', value));
            item.toggle();
            for (const element of item.render()) {
                this._elements.push(element);
            }
        }
    }

    render() {
        return this._elements;
    }
};

view.ConnectionSidebar = class extends view.ObjectSidebar {

    constructor(context, value, from, to) {
        super(context);
        this._value = value;
        this._from = from;
        this._to = to;
    }

    get identifier() {
        return 'connection';
    }

    render() {
        const value = this._value;
        const from = this._from;
        const to = this._to;
        const [name] = value.name.split('\n');
        this.addProperty('name', name);
        if (value.type) {
            const item = new view.ValueView(this._view, value);
            this.addEntry('type', item);
            item.toggle();
        }
        if (from) {
            this.addSection('Inputs');
            this.addNodeList('from', [from]);
        }
        if (Array.isArray(to) && to.length > 0) {
            this.addSection('Outputs');
            this.addNodeList('to', to);
        }
        this.addQuantization(this._view.model.attachment.quantization.value(value));
        const metadata = this._view.model.attachment.metadata.value(value);
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addSection('Metadata');
            for (const argument of metadata) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
        const metrics = this._view.model.attachment.metrics.value(value);
        if (Array.isArray(metrics) && metrics.length > 0) {
            this.addSection('Metrics');
            for (const argument of metrics) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
    }

    addNodeList(name, list) {
        const entry = new view.NodeListView(this._view, list);
        entry.on('focus', (sender, value) => {
            this.emit('focus', value);
            this._focused = this._focused || new Set();
            this._focused.add(value);
        });
        entry.on('blur', (sender, value) => {
            this.emit('blur', value);
            this._focused = this._focused || new Set();
            this._focused.delete(value);
        });
        entry.on('select', (sender, value) => this.emit('select', value));
        entry.on('activate', (sender, value) => this.emit('activate', value));
        this.addEntry(name, entry);
    }

    activate() {
        this.emit('select', this._value);
    }

    deactivate() {
        this.emit('select', null);
        if (this._focused) {
            for (const value of this._focused) {
                this.emit('blur', value);
            }
            this._focused.clear();
        }
    }
};

view.TensorSidebar = class extends view.ObjectSidebar {

    constructor(context, value, quantizationTitle) {
        super(context);
        this._value = value;
        this._quantizationTitle = quantizationTitle;
    }

    get identifier() {
        return 'tensor';
    }

    render() {
        const [value] = this._value.value;
        const tensor = value.initializer;
        const name = tensor && tensor.name ? tensor.name : value.name.split('\n')[0];
        if (name) {
            this.addProperty('name', name);
        }
        if (!tensor && value.type) {
            const item = new view.ValueView(this._view, value);
            this.addEntry('type', item);
            item.toggle();
        }
        if (tensor) {
            const category = tensor.category;
            if (category) {
                this.addProperty('category', category);
            }
            const description = tensor.description;
            if (description) {
                this.addProperty('description', description);
            }
            const type = tensor.type;
            if (type) {
                const dataType = type.dataType;
                this.addProperty('type', `${dataType}`, 'code');
                const shape = type.shape && Array.isArray(type.shape.dimensions) ? type.shape.dimensions.toString(', ') : '?';
                if (shape) {
                    this.addProperty('shape', shape, 'code');
                }
                const denotation = type.denotation;
                if (denotation) {
                    this.addProperty('denotation', denotation, 'code');
                }
                const layout = type.layout;
                if (layout) {
                    this.addProperty('layout', layout.replace('.', ' '));
                }
            }
            const location = tensor.location;
            if (location) {
                this.addProperty('location', tensor.location);
            }
            const stride = tensor.stride;
            if (Array.isArray(stride) && stride.length > 0) {
                this.addProperty('stride', stride.join(','), 'code');
            }
            const tensorView = new view.TensorView(this._view, tensor, this._tensor);
            this.addEntry('value', tensorView);
            const attributes = tensor.attributes;
            if (Array.isArray(attributes) && attributes.length > 0) {
                this.addSection('Attributes');
                for (const attribute of attributes) {
                    this.addArgument(attribute.name, attribute, 'attribute');
                }
            }
            const metadata = this._view.model.attachment.metadata.tensor(tensor);
            if (Array.isArray(metadata) && metadata.length > 0) {
                this.addSection('Metadata');
                for (const argument of metadata) {
                    this.addArgument(argument.name, argument, 'attribute');
                }
            }
        }
        this.addQuantization(
            this._view.model.attachment.quantization.tensor(tensor, value),
            this._quantizationTitle
        );
        // Metrics
        if (value.initializer) {
            const tensor = value.initializer;
            const promise = tensor.peek && !tensor.peek() ? tensor.read() : Promise.resolve();
            promise.then(() => {
                this._tensor = new base.Tensor(tensor);
                if (!this._tensor.empty) {
                    if (!this._metrics) {
                        const tensor = new metrics.Tensor(this._tensor);
                        this._metrics = this._view.model.attachment.metrics.tensor(tensor);
                    }
                    if (this._metrics.length > 0) {
                        this.addSection('Metrics');
                        for (const metric of this._metrics) {
                            const value = metric.type === 'percentage' ? `${(metric.value * 100).toFixed(1)}%` : metric.value;
                            const argument = new metadata.Argument(metric.name, value, metric.type);
                            this.addArgument(metric.name, argument, 'attribute');
                        }
                    }
                }
            });
        }
    }

    activate() {
        this.emit('select', this._value);
    }

    deactivate() {
        this.emit('select', null);
    }
};

view.ModelSidebar = class extends view.ObjectSidebar {

    constructor(context, model) {
        super(context);
        this._model = model;
    }

    get identifier() {
        return 'model';
    }

    render() {
        const model = this._model;
        if (model.format) {
            this.addProperty('format', model.format);
        }
        if (model.producer) {
            this.addProperty('producer', model.producer);
        }
        if (model.name) {
            this.addProperty('name', model.name);
        }
        if (model.version) {
            this.addProperty('version', model.version);
        }
        if (model.description) {
            this.addProperty('description', model.description);
        }
        if (model.domain) {
            this.addProperty('domain', model.domain);
        }
        if (model.imports) {
            this.addProperty('imports', model.imports);
        }
        if (model.runtime) {
            this.addProperty('runtime', model.runtime);
        }
        if (model.source) {
            this.addProperty('source', model.source);
        }
        const quantization = this._view.model.attachment.quantization;
        if (!quantization.empty) {
            const summary = quantization.summary;
            const profile = quantization.profile;
            this.addSection('Quantization');
            this.addProperty('format', `AIMET encodings ${quantization.version}`);
            this.addProperty('activations', summary.activations, 'code');
            this.addProperty('parameters', summary.parameters, 'code');
            for (const [label, count] of profile.encodings) {
                this.addProperty(label, count, 'code');
            }
            if (profile.cache.length > 0) {
                this.addProperty('KV cache', profile.cache.map(([label, count]) => `${label}: ${count}`).join(', '), 'code');
            }
            this.addProperty('matched', `${summary.matched} / ${summary.total}`, 'code');
            if (summary.unmatched > 0) {
                this.addProperty('unmatched', summary.unmatched, 'code');
            }
            if (summary.unencoded > 0) {
                this.addProperty('model tensors without encoding', summary.unencoded, 'code');
            }
            if (summary.inferred > 0) {
                this.addProperty('inferred activation precision', summary.inferred, 'code');
            }
            if (quantization.issues.length > 0) {
                this.addSection('Validation');
                const limit = 20;
                for (const issue of quantization.issues.slice(0, limit)) {
                    this.addProperty(issue.severity, issue.message);
                }
                if (quantization.issues.length > limit) {
                    this.addProperty('more', `${quantization.issues.length - limit} additional issues`);
                }
            }
        }
        const metadata = this._view.model.attachment.metadata.model(model);
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addSection('Metadata');
            for (const argument of metadata) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
        const metrics = this.metrics;
        if (Array.isArray(metrics) && metrics.length > 0) {
            this.addSection('Metrics');
            for (const argument of metrics) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
    }

    get metrics() {
        const model = new metrics.Model(this._model);
        return this._view.model.attachment.metrics.model(model);
    }
};

view.StatisticsSidebar = class extends view.ObjectSidebar {

    constructor(context, model, target) {
        super(context);
        this._model = model;
        this._target = target;
    }

    get identifier() {
        return 'statistics';
    }

    render() {
        const quantization = this._view.model.attachment.quantization;
        const targets = new Set([...(this._model.modules || []), ...(this._model.functions || [])]);
        const nodes = new Set();
        const visit = (target) => {
            for (const node of target.nodes || []) {
                nodes.add(node);
                for (const block of node.blocks || []) {
                    if (block && block.type === 'graph' && block.value) {
                        visit(block.value);
                    }
                }
            }
        };
        for (const target of targets) {
            visit(target);
        }
        this.addSection('Model');
        this.addProperty('graphs', targets.size, 'code');
        this.addProperty('nodes', nodes.size, 'code');
        if (this._target && Array.isArray(this._target.nodes)) {
            this.addProperty('current graph nodes', this._target.nodes.length, 'code');
        }
        if (!quantization.empty) {
            const summary = quantization.summary;
            const profile = quantization.profile;
            const nodeCounts = new Map();
            let encodedNodes = 0;
            for (const node of nodes) {
                const explicit = quantization.node(node, false);
                const group = quantization.node(node);
                if (explicit.parameters.length > 0 || explicit.outputs.length > 0) {
                    encodedNodes++;
                }
                if (group.entries.length > 0) {
                    const precision = aimet.EncodingFile.precision(group).toUpperCase();
                    nodeCounts.set(precision, (nodeCounts.get(precision) || 0) + 1);
                }
            }
            this.addSection('Node Precision');
            this.addProperty('with QParam', encodedNodes, 'code');
            this.addProperty('without QParam', nodes.size - encodedNodes, 'code');
            for (const [label, count] of Array.from(nodeCounts).sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))) {
                this.addProperty(label, count, 'code');
            }
            this.addSection('Encodings');
            for (const [label, count] of profile.encodings) {
                this.addProperty(label, count, 'code');
            }
            if (profile.cache.length > 0) {
                this.addProperty('KV cache', profile.cache.map(([label, count]) => `${label}: ${count}`).join(', '), 'code');
            }
            this.addProperty('matched QParams', `${summary.matched} / ${summary.total}`, 'code');
            this.addProperty('mismatched QParams', summary.unmatched, 'code');
            this.addProperty('model tensors without QParam', summary.unencoded, 'code');
            this.addProperty('inferred activation precision', summary.inferred, 'code');
            const errors = quantization.issues.filter((issue) => issue.severity === 'error').length;
            const warnings = quantization.issues.filter((issue) => issue.severity === 'warning').length;
            this.addSection('Validation');
            this.addProperty('errors', errors, 'code');
            this.addProperty('warnings', warnings, 'code');
        }
    }
};

view.TargetSidebar = class extends view.ObjectSidebar {

    constructor(context, target, signature) {
        super(context);
        this._target = target;
        this._signature = signature;
    }

    render() {
        const target = this._target;
        const signature = this._signature;
        if (target.name) {
            const item = this.addProperty('name', target.name);
            if (target.type === 'function') {
                item.action('\u0192', 'Show Function Documentation', () => {
                    this.emit('show-definition', null);
                });
            }
        }
        if (signature && signature.name) {
            this.addProperty('signature', signature.name);
        }
        if (target.version) {
            this.addProperty('version', target.version);
        }
        if (target.description) {
            this.addProperty('description', target.description);
        }
        const attributes = signature ? signature.attributes : target.attributes;
        const inputs = signature ? signature.inputs : target.inputs;
        const outputs = signature ? signature.outputs : target.outputs;
        if (Array.isArray(attributes) && attributes.length > 0) {
            this.addSection('Attributes');
            for (const attribute of attributes) {
                this.addProperty(attribute.name, attribute.value);
            }
        }
        if (Array.isArray(inputs) && inputs.length > 0) {
            this.addSection('Inputs');
            for (const input of inputs) {
                const value = this.addArgument(input.name, input);
                value.toggle();
            }
        }
        if (Array.isArray(outputs) && outputs.length > 0) {
            this.addSection('Outputs');
            for (const output of outputs) {
                const value = this.addArgument(output.name, output);
                value.toggle();
            }
        }
        const metadata = this._view.model.attachment.metadata.graph(target);
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addSection('Metadata');
            for (const argument of metadata) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
        const metrics = this.metrics;
        if (Array.isArray(metrics) && metrics.length > 0) {
            this.addSection('Metrics');
            for (const argument of metrics) {
                this.addArgument(argument.name, argument, 'attribute');
            }
        }
    }

    get metrics() {
        const target = new metrics.Target(this._target);
        return this._view.model.attachment.metrics.graph(target);
    }

    get identifier() {
        return 'target';
    }
};

view.DocumentationSidebar = class extends view.Control {

    constructor(context, type) {
        super(context);
        this._type = type;
        this._escapeReplacementsMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
        this._escapeTestNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/;
        this._escapeReplaceNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/g;
    }

    get identifier() {
        return 'documentation';
    }

    render() {
        if (!this.element) {
            this.element = this.createElement('div', 'sidebar-documentation');
            const type = view.Documentation.open(this._type);
            this._append(this.element, 'h1', type.name);
            if (type.summary) {
                this._append(this.element, 'p', type.summary);
            }
            if (type.description) {
                this._append(this.element, 'p', type.description);
            }
            if (Array.isArray(type.attributes) && type.attributes.length > 0) {
                this._append(this.element, 'h2', 'Attributes');
                const attributes = this._append(this.element, 'dl');
                for (const attribute of type.attributes) {
                    this._append(attributes, 'dt', attribute.name + (attribute.type ? `: <tt>${this._escape(attribute.type)}</tt>` : ''));
                    this._append(attributes, 'dd', attribute.description);
                }
                this.element.appendChild(attributes);
            }
            if (Array.isArray(type.inputs) && type.inputs.length > 0) {
                this._append(this.element, 'h2', `Inputs${type.inputs_range ? ` (${type.inputs_range})` : ''}`);
                const inputs = this._append(this.element, 'dl');
                for (const input of type.inputs) {
                    this._append(inputs, 'dt', input.name + (input.type ? `: <tt>${this._escape(input.type)}</tt>` : '') + (input.option ? ` (${input.option})` : ''));
                    this._append(inputs, 'dd', input.description);
                }
            }
            if (Array.isArray(type.outputs) && type.outputs.length > 0) {
                this._append(this.element, 'h2', `Outputs${type.outputs_range ? ` (${type.outputs_range})` : ''}`);
                const outputs = this._append(this.element, 'dl');
                for (const output of type.outputs) {
                    this._append(outputs, 'dt', output.name + (output.type ? `: <tt>${this._escape(output.type)}</tt>` : '') + (output.option ? ` (${output.option})` : ''));
                    this._append(outputs, 'dd', output.description);
                }
            }
            if (Array.isArray(type.type_constraints) && type.type_constraints.length > 0) {
                this._append(this.element, 'h2', 'Type Constraints');
                const type_constraints = this._append(this.element, 'dl');
                for (const type_constraint of type.type_constraints) {
                    this._append(type_constraints, 'dt', `${type_constraint.type_param_str}: ${type_constraint.allowed_type_strs.map((item) => `<tt>${item}</tt>`).join(', ')}`);
                    this._append(type_constraints, 'dd', type_constraint.description);
                }
            }
            if (Array.isArray(type.examples) && type.examples.length > 0) {
                this._append(this.element, 'h2', 'Examples');
                for (const example of type.examples) {
                    this._append(this.element, 'h3', example.summary);
                    this._append(this.element, 'pre', example.code);
                }
            }
            if (Array.isArray(type.references) && type.references.length > 0) {
                this._append(this.element, 'h2', 'References');
                const references = this._append(this.element, 'ul');
                for (const reference of type.references) {
                    this._append(references, 'li', reference.description);
                }
            }
            if (this._host.type === 'Electron') {
                this.element.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        const url = e.target.href;
                        if (url.startsWith('http://') || url.startsWith('https://')) {
                            e.preventDefault();
                            this.emit('navigate', { link: url });
                        }
                    }
                });
            }
        }
    }

    _append(parent, type, content) {
        const element = this.createElement(type);
        if (content) {
            element.innerHTML = content;
        }
        parent.appendChild(element);
        return element;
    }

    _escape(content) {
        if (this._escapeTestNoEncodeRegExp.test(content)) {
            return content.replace(this._escapeReplaceNoEncodeRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('span');
        const title = this.createElement('b');
        title.textContent = 'ERROR: ';
        element.appendChild(title);
        const message = this.createTextNode(error.message);
        element.appendChild(message);
        this.element.appendChild(element);
    }
};

view.FindSidebar = class extends view.Control {

    constructor(context, state, graph, signature) {
        super(context);
        this._target = graph;
        this._signature = signature;
        this._state = state || {
            query: '',
            node: true,
            connection: true,
            weight: true
        };
        this._toggles = {
            node: { hide: 'Hide Nodes', show: 'Show Nodes' },
            connection: { hide: 'Hide Connections', show: 'Show Connections' },
            weight: { hide: 'Hide Weights', show: 'Show Weights' }
        };
    }

    get identifier() {
        return 'find';
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        try {
            if (this._events && this._events[event]) {
                for (const callback of this._events[event]) {
                    callback(this, data);
                }
            }
        } catch (error) {
            this.error(error, false);
        }
    }

    _reset() {
        this._focus(null);
        this._table.clear();
        this._content.replaceChildren();
        this._edges.clear();
        for (const value of Object.values(this._toggles)) {
            delete value.template;
        }
        const unquote = this._state.query.match(new RegExp(/^'(.*)'|"(.*)"$/));
        if (unquote) {
            this._exact = true;
            const term = unquote[1] || unquote[2];
            this._terms = [term];
        } else {
            this._exact = false;
            this._terms = this._state.query.trim().toLowerCase().split(' ').map((term) => term.trim()).filter((term) => term.length > 0);
        }
    }

    _term(value) {
        if (this._exact) {
            return value === this._terms[0];
        }
        value = value.toLowerCase();
        return this._terms.every((term) => value.indexOf(term) !== -1);
    }

    _value(value) {
        if (this._terms.length === 0) {
            return true;
        }
        if (value.name && this._term(value.name.split('\n').shift())) {
            return true;
        }
        if (value.identifier && this._term(value.identifier)) {
            return true;
        }
        if (value.type && !this._exact) {
            for (const term of this._terms) {
                if (value.type.dataType && term === value.type.dataType.toLowerCase()) {
                    return true;
                }
                if (value.type.shape) {
                    if (term === value.type.shape.toString().toLowerCase()) {
                        return true;
                    }
                    if (value.type.shape && Array.isArray(value.type.shape.dimensions)) {
                        const dimensions = value.type.shape.dimensions.map((dimension) => dimension ? dimension.toString().toLowerCase() : '');
                        if (term === dimensions.join(',')) {
                            return true;
                        }
                        if (dimensions.some((dimension) => term === dimension)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    _edge(value) {
        if (value.name && !this._edges.has(value.name) && this._value(value)) {
            const content = `${value.name.split('\n').shift()}`;
            this._add(value, content, 'connection'); // split custom argument id
            this._edges.add(value.name);
        }
    }

    _node(node) {
        if (this._state.connection) {
            const inputs = node.inputs;
            if (Array.isArray(inputs)) {
                for (const input of node.inputs) {
                    if (!input.type || input.type.endsWith('*')) {
                        for (const value of input.value) {
                            if (value !== null && !value.initializer) {
                                this._edge(value);
                            }
                        }
                    }
                }
            }
        }
        if (this._state.node) {
            const name = node.name;
            const type = node.type.name;
            const identifier = node.identifier;
            if ((name && this._term(name)) || (type && this._term(type)) || (identifier && this._term(identifier))) {
                const content = `${name || `[${type}]`}`;
                this._add(node, content, 'node');
            }
        }
        if (this._state.weight) {
            const inputs = node.inputs;
            if (Array.isArray(inputs)) {
                for (const argument of node.inputs) {
                    if (!argument.type || argument.type.endsWith('*')) {
                        for (const value of argument.value) {
                            if (value !== null && value.initializer && this._value(value)) {
                                let content = null;
                                if (value.name) {
                                    content = `${value.name.split('\n').shift()}`; // split custom argument id
                                } else if (Array.isArray(argument.value) && argument.value.length === 1 && argument.name.indexOf('.') !== -1) {
                                    content = argument.name;
                                } else if (value.type && value.type.shape && Array.isArray(value.type.shape.dimensions) && value.type.shape.dimensions.length > 0) {
                                    content = `${value.type.shape.dimensions.map((d) => (d !== null && d !== undefined) ? d : '?').join('\u00D7')}`;
                                }
                                if (content) {
                                    const target = argument.value.length === 1 ? argument : node;
                                    this._add(target, content, 'weight');
                                }
                            }
                        }
                    } else if (argument.type === 'object') {
                        this._node(argument.value);
                    } else if (argument.type === 'object[]') {
                        for (const value of argument.value) {
                            this._node(value);
                        }
                    }
                }
            }
        }
    }

    _add(value, content, type) {
        if (!this._toggles[type].template) {
            const element = this.createElement('li');
            element.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-${type}"></use></svg>`;
            this._toggles[type].template = element;
        }
        const element = this._toggles[type].template.cloneNode(true);
        const text = this._host.document.createTextNode(content);
        element.appendChild(text);
        this._table.set(element, value);
        this._content.appendChild(element);
    }

    _focus(element, event) {
        if (this._cursor === element) {
            return;
        }
        if (this._cursor) {
            this._cursor.classList.remove('focus');
            this.emit('blur', this._table.get(this._cursor));
        }
        this._cursor = element;
        if (element) {
            element.classList.add('focus');
            this.emit(event || 'focus', this._table.get(element));
            element.scrollIntoView({ block: 'nearest' });
        }
    }

    _move(delta) {
        const forward = delta > 0;
        let count = Math.abs(delta);
        let current = this._cursor;
        let target = null;
        while (count > 0) {
            let sibling = null;
            if (current) {
                sibling = forward ? current.nextElementSibling : current.previousElementSibling;
            } else {
                sibling = forward ? this._content.firstElementChild : this._content.lastElementChild;
            }
            if (!sibling) {
                break;
            }
            current = sibling;
            target = sibling;
            count--;
        }
        if (target) {
            this._keyboard = true;
            this._focus(target, 'select');
        }
    }

    _page() {
        const child = this._content.firstElementChild;
        const height = child ? child.offsetHeight : 0;
        return height > 0 ? Math.max(1, Math.floor(this._content.clientHeight / height)) : 1;
    }

    _update() {
        try {
            this._reset();
            const inputs = this._signature ? this._signature.inputs : this._target.inputs;
            if (this._state.connection) {
                for (const input of inputs) {
                    for (const value of input.value) {
                        this._edge(value);
                    }
                }
            }
            for (const node of this._target.nodes) {
                this._node(node);
            }
            if (this._state.connection) {
                const outputs = this._signature ? this._signature.outputs : this._target.outputs;
                for (const output of outputs) {
                    if (!output.type || output.type.endsWith('*')) {
                        for (const value of output.value) {
                            this._edge(value);
                        }
                    }
                }
            }
        } catch (error) {
            this.error(error, false);
        }
    }

    render() {
        this._table = new Map();
        this._cursor = null;
        this._active = null;
        this._edges = new Set();
        this._keyboard = false;
        this._search = this.createElement('div', 'sidebar-find-search');
        this._query = this.createElement('input', 'sidebar-find-query');
        this._search.appendChild(this._query);
        this._content = this.createElement('ol', 'sidebar-find-content');
        this._content.setAttribute('tabindex', '-1');
        this._elements = [this._search, this._content];
        this._query.setAttribute('id', 'search');
        this._query.setAttribute('type', 'text');
        this._query.setAttribute('spellcheck', 'false');
        this._query.setAttribute('placeholder', 'Search');
        this._query.addEventListener('input', (e) => {
            this._state.query = e.target.value;
            this.emit('state-changed', this._state);
            this._update();
        });
        this._query.addEventListener('keydown', (e) => {
            if (e.keyCode === 0x08 && !e.altKey && !e.ctrlKey && !e.shiftKey && !e.metaKey) {
                e.stopPropagation();
            } else if (e.keyCode === 0x28 && this._content.children.length > 0) { // Down
                e.preventDefault();
                e.stopPropagation();
                this._focus(null);
                this._content.focus();
                this._move(1);
            }
        });
        for (const [name, toggle] of Object.entries(this._toggles)) {
            toggle.element = this.createElement('label', 'sidebar-find-toggle');
            toggle.element.innerHTML = `<svg class='sidebar-find-toggle-icon'><use href="#sidebar-icon-${name}"></use></svg>`;
            toggle.element.setAttribute('title', this._state[name] ? toggle.hide : toggle.show);
            toggle.checkbox = this.createElement('input');
            toggle.checkbox.setAttribute('type', 'checkbox');
            toggle.checkbox.setAttribute('data', name);
            toggle.checkbox.addEventListener('change', (e) => {
                const name = e.target.getAttribute('data');
                this._state[name] = e.target.checked;
                const toggle = this._toggles[name];
                toggle.element.setAttribute('title', e.target.checked ? toggle.hide : toggle.show);
                this.emit('state-changed', this._state);
                this._update();
            });
            toggle.element.insertBefore(toggle.checkbox, toggle.element.firstChild);
            this._search.appendChild(toggle.element);
        }
        this._content.addEventListener('keydown', (e) => {
            if (e.keyCode === 0x28) { // Down
                e.preventDefault();
                e.stopPropagation();
                this._move(1);
            } else if (e.keyCode === 0x26) { // Up
                e.preventDefault();
                e.stopPropagation();
                if (!this._cursor || this._cursor === this._content.firstChild) {
                    this._focus(null);
                    this._query.focus();
                } else {
                    this._move(-1);
                }
            } else if (e.keyCode === 0x22) { // Page Down
                e.preventDefault();
                e.stopPropagation();
                this._move(this._page());
            } else if (e.keyCode === 0x21) { // Page Up
                e.preventDefault();
                e.stopPropagation();
                this._move(-this._page());
            } else if (e.keyCode === 0x0D && this._cursor) { // Enter
                e.preventDefault();
                e.stopPropagation();
                this._active = this._table.get(this._cursor);
                this.emit('activate', this._active);
            }
        });
        this._content.addEventListener('click', (e) => {
            if (this._table.has(e.target)) {
                this._keyboard = false;
                this.emit('select', this._table.get(e.target));
            }
        });
        this._content.addEventListener('dblclick', (e) => {
            if (this._table.has(e.target)) {
                this._keyboard = false;
                this._active = this._table.get(e.target);
                this.emit('activate', this._active);
            }
        });
        this._content.addEventListener('pointermove', (e) => {
            this._keyboard = false;
            if (this._table.has(e.target)) {
                this._focus(e.target);
            }
        });
        this._content.addEventListener('pointerover', (e) => {
            if (!this._keyboard && this._table.has(e.target)) {
                this._focus(e.target);
            }
        });
        this._content.addEventListener('pointerleave', () => {
            this._keyboard = false;
            if (this._host.document.activeElement !== this._content) {
                this._focus(null);
            }
        });
    }

    get element() {
        return [this._search, this._content];
    }

    activate() {
        this._query.value = '';
        this._query.value = this._state.query;
        for (const [name, toggle] of Object.entries(this._toggles)) {
            toggle.checkbox.checked = this._state[name];
            toggle.element.setAttribute('title', this._state[name] ? toggle.hide : toggle.show);
        }
        this._update();
        let cursor = null;
        if (this._active) {
            for (const [element, value] of this._table) {
                if (value === this._active) {
                    cursor = element;
                    break;
                }
            }
        }
        if (cursor) {
            this._keyboard = true;
            this._content.focus();
            this._focus(cursor, 'select');
        } else {
            this._query.focus();
        }
        this._host.event('open_sidebar', {
            sidebar_identifier: this.identifier,
            sidebar_size: this._table.size
        });
    }

    deactivate() {
        this._reset();
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('li');
        const title = this.createElement('b');
        title.textContent = 'ERROR: ';
        element.appendChild(title);
        const message = this.createTextNode(` ${error.message}`);
        element.appendChild(message);
        this._content.appendChild(element);
    }
};

view.Quantization = class {

    constructor(quantization) {
        Object.assign(this, quantization);
    }

    toString() {
        if (this.type === 'linear' || /^quant\d\d?_.*$/.test(this.type)) {
            const content = [];
            const scale = this.scale || [];
            const offset = this.offset || [];
            const bias = this.bias || [];
            const max = this.max || [];
            const min = this.min || [];
            const length = Math.max(scale.length, offset.length, bias.length, min.length, max.length);
            const size = length.toString().length;
            for (let i = 0; i < length; i++) {
                let s = 'q';
                let bracket = false;
                if (i < offset.length && offset[i] !== undefined && offset[i] !== 0 && offset[i] !== 0n) {
                    const value = offset[i];
                    s = value > 0 ? `${s} - ${value}` : `${s} + ${-value}`;
                    bracket = true;
                }
                if (i < scale.length && scale[i] !== undefined && scale[i] !== 1 && scale[i] !== 1n) {
                    const value = scale[i];
                    s = bracket ? `(${s})` : s;
                    s = `${value} * ${s}`;
                    bracket = true;
                }
                if (i < bias.length && bias[i] !== undefined && bias[i] !== 0 && bias[i] !== 0n) {
                    const value = bias[i];
                    s = bracket ? `(${s})` : s;
                    s = value < 0 ? `${s} - ${-value}` : `${s} + ${value}`;
                }
                if (i < min.length && min[i] !== undefined && min[i] !== 0 && min[i] !== 0n) {
                    s = `${min[i]} \u2264 ${s}`;
                }
                if (i < max.length && max[i] !== undefined && max[i] !== 0 && max[i] !== 0n) {
                    s = `${s} \u2264 ${max[i]}`;
                }
                content.push(length > 1 ? `${i.toString().padStart(size, ' ')}: ${s}` : `${s}`);
            }
            return content.join('\n');
        } else if (this.type === 'lookup') {
            const size = this.value.length.toString().length;
            return this.value.map((value, index) => `${index.toString().padStart(size, ' ')}: ${value}`).join('\n');
        } else if (this.type === 'annotation') {
            return Array.from(this.value).map(([name, value]) => `${name} = ${value}`).join('\n');
        } else if (/^q\d_[01k]$/.test(this.type) || /^iq\d_[xsnlm]+$/.test(this.type) || this.type === 'mxfp4') {
            return '';
        }
        throw new view.Error(`Unknown quantization type '${this.type}'.`);
    }
};

view.Documentation = class {

    static open(source) {
        if (source) {
            const generator = markdown.Generator.open();
            const target = {};
            if (source.name) {
                target.name = source.name;
            }
            if (source.module) {
                target.module = source.module;
            }
            if (source.category) {
                target.category = source.category;
            }
            if (source.summary) {
                target.summary = generator.html(source.summary);
            }
            if (source.description) {
                target.description = generator.html(source.description);
            }
            if (Array.isArray(source.attributes)) {
                target.attributes = source.attributes.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type === null || typeof source.type === 'string' ? source.type : source.type.toString();
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.required !== undefined) {
                        target.required = source.required;
                    }
                    if (source.minimum !== undefined) {
                        target.minimum = source.minimum;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.src_type !== undefined) {
                        target.src_type = source.src_type;
                    }
                    if (source.description) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.inputs)) {
                target.inputs = source.inputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type === null || typeof source.type === 'string' ? source.type : source.type.toString();
                    }
                    if (source.description) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeListAttr;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.outputs)) {
                target.outputs = source.outputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type) {
                        target.type = source.type === null || typeof source.type === 'string' ? source.type : source.type.toString();
                    }
                    if (source.description) {
                        target.description = generator.html(source.description);
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeListAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.references)) {
                target.references = source.references.map((source) => {
                    if (source) {
                        target.description = generator.html(source.description);
                    }
                    return target;
                });
            }
            if (source.version !== undefined) {
                target.version = source.version;
            }
            if (source.operator !== undefined) {
                target.operator = source.operator;
            }
            if (source.identifier !== undefined) {
                target.identifier = source.identifier;
            }
            if (source.package !== undefined) {
                target.package = source.package;
            }
            if (source.status !== undefined) {
                target.status = source.status;
            }
            if (source.min_input !== undefined) {
                target.min_input = source.min_input;
            }
            if (source.max_input !== undefined) {
                target.max_input = source.max_input;
            }
            if (source.min_output !== undefined) {
                target.min_output = source.min_output;
            }
            if (source.max_output !== undefined) {
                target.max_output = source.max_output;
            }
            if (source.inputs_range !== undefined) {
                target.inputs_range = source.inputs_range;
            }
            if (source.outputs_range !== undefined) {
                target.outputs_range = source.outputs_range;
            }
            if (source.examples !== undefined) {
                target.examples = source.examples;
            }
            if (source.constants !== undefined) {
                target.constants = source.constants;
            }
            if (source.type_constraints !== undefined) {
                target.type_constraints = source.type_constraints;
            }
            return target;
        }
        return null;
    }
};

view.Formatter = class {

    constructor(value, type, quote) {
        this._value = value;
        this._type = type;
        this._quote = quote;
        this._values = new Set();
    }

    toString() {
        return this._format(this._value, this._type, this._quote);
    }

    _format(value, type, quote) {
        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'type') {
            return `${value.__module__}.${value.__name__}`;
        }
        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'function') {
            return `${value.__module__}.${value.__name__}`;
        }
        if (typeof value === 'function') {
            return value();
        }
        if (value !== null && value !== undefined && (typeof value === 'bigint' || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')) {
            return value.toString();
        }
        if (Number.isNaN(value)) {
            return 'NaN';
        }
        switch (type) {
            case 'shape':
                return value ? value.toString() : '(null)';
            case 'shape[]':
                if (value && !Array.isArray(value)) {
                    throw new Error(`Invalid shape '${JSON.stringify(value)}'.`);
                }
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'graph':
                return value ? value.name : '(null)';
            case 'graph[]':
                return value ? value.map((graph) => graph.name).join(', ') : '(null)';
            case 'tensor': {
                if (value === null) {
                    return '(null)';
                }
                return view.Formatter.tensor(value);
            }
            case 'object':
                return value.type.name;
            case 'function':
                return value.name;
            case 'object[]':
            case 'function[]':
                return value ? value.map((item) => item.type.name).join(', ') : '(null)';
            case 'type':
                return value ? value.toString() : '(null)';
            case 'type[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'complex':
                return value ? value.toString() : '(null)';
            default:
                break;
        }
        if (typeof value === 'string' && (!type || type !== 'string')) {
            if (quote) {
                return `"${value}"`;
            }
            if (value.trim().length === 0) {
                return value;
            }
            return value;
        }
        if (Array.isArray(value)) {
            if (value.length === 0) {
                return quote ? '[]' : '';
            }
            let ellipsis = false;
            if (value.length > 2000) {
                value = value.slice(0, 2000);
                ellipsis = true;
            }
            const itemType = (type && type.endsWith('[]')) ? type.substring(0, type.length - 2) : null;
            const array = value.map((value) => {
                if (value && typeof value === 'bigint') {
                    return value.toString();
                }
                if (Number.isNaN(value)) {
                    return 'NaN';
                }
                if (value && value.constructor && value.constructor.name === 'Value' && value.name) {
                    return `{${value.name}}`;
                }
                const quote = !itemType || itemType === 'string';
                return this._format(value, itemType, quote);
            });
            if (ellipsis) {
                array.push('\u2026');
            }
            return quote ? ['[', array.join(', '), ']'].join(' ') : array.join(', ');
        }
        if (value === null) {
            return quote ? 'null' : '';
        }
        if (value === undefined) {
            return 'undefined';
        }
        if (value !== Object(value)) {
            return value.toString();
        }
        if (this._values.has(value)) {
            return '\u2026';
        }
        this._values.add(value);
        let list = null;
        const map = value instanceof Map ? Array.from(value) : Object.entries(value);
        const entries = map.filter(([name]) => typeof name === 'string' && !name.startsWith('__') && !name.endsWith('__'));
        if (entries.length === 1) {
            list = [this._format(entries[0][1], null, true)];
        } else {
            list = entries.map(([name, value]) => `${name}: ${this._format(value, null, true)}`);
        }
        let objectType = value.__type__;
        if (!objectType && value.constructor.name && value.constructor.name !== 'Object') {
            objectType = value.constructor.name;
        }
        if (objectType) {
            return objectType + (list.length === 0 ? '()' : ['(', list.join(', '), ')'].join(''));
        }
        switch (list.length) {
            case 0:
                return quote ? '()' : '';
            case 1:
                return list[0];
            default:
                return quote ? ['(', list.join(', '), ')'].join(' ') : list.join(', ');
        }
    }

    static tensor(value) {
        const type = value.type;
        if (type && type.shape && type.shape.dimensions && Array.isArray(type.shape.dimensions)) {
            if (type.shape.dimensions.length === 0 && (!value.peek || value.peek() === true)) {
                const tensor = new base.Tensor(value);
                const encoding = tensor.encoding;
                if ((encoding === '<' || encoding === '>' || encoding === '|') && !tensor.empty && tensor.type.dataType !== '?') {
                    let content = tensor.toString();
                    if (content && content.length > 10) {
                        content = `${content.substring(0, 10)}\u2026`;
                    }
                    return content;
                }
            }
            const content = type.shape.dimensions.map((d) => (d !== null && d !== undefined) ? d : '?').join('\u00D7');
            return `\u3008${content}\u3009`;
        }
        return '\u3008\u2026\u3009';
    }
};

markdown.Generator = class {

    static open() {
        if (!markdown.Generator.generator) {
            markdown.Generator.generator = new markdown.Generator();
        }
        return markdown.Generator.generator;
    }

    constructor() {
        this._newlineRegExp = /^\n+/;
        this._codeRegExp = /^( {4}[^\n]+\n*)+/;
        this._fencesRegExp = /^ {0,3}(`{3,}(?=[^`\n]*\n)|~{3,})([^\n]*)\n(?:|([\s\S]*?)\n)(?: {0,3}\1[~`]* *(?:\n+|$)|$)/;
        this._hrRegExp = /^ {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)/;
        this._headingRegExp = /^ {0,3}(#{1,6}) +([^\n]*?)(?: +#+)? *(?:\n+|$)/;
        this._blockquoteRegExp = /^( {0,3}> ?(([^\n]+(?:\n(?! {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--))[^\n]+)*)|[^\n]*)(?:\n|$))+/;
        this._listRegExp = /^( {0,3})((?:[*+-]|\d{1,9}[.)])) [\s\S]+?(?:\n+(?=\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$))|\n+(?= {0,3}\[((?!\s*\])(?:\\[[\]]|[^[\]])+)\]: *\n? *<?([^\s>]+)>?(?:(?: +\n? *| *\n *)((?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))))? *(?:\n+|$))|\n{2,}(?! )(?!\1(?:[*+-]|\d{1,9}[.)]) )\n*|\s*$)/;
        this._htmlRegExp = /^ {0,3}(?:<(script|pre|style)[\s>][\s\S]*?(?:<\/\1>[^\n]*\n+|$)|<!--(?!-?>)[\s\S]*?(?:-->|$)[^\n]*(\n+|$)|<\?[\s\S]*?(?:\?>\n*|$)|<![A-Z][\s\S]*?(?:>\n*|$)|<!\[CDATA\[[\s\S]*?(?:\]\]>\n*|$)|<\/?(address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)[\s\S]*?(?:\n{2,}|$)|<(?!script|pre|style)([a-z][\w-]*)(?: +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?)*? *\/?>(?=[ \t]*(?:\n|$))[\s\S]*?(?:\n{2,}|$)|<\/(?!script|pre|style)[a-z][\w-]*\s*>(?=[ \t]*(?:\n|$))[\s\S]*?(?:\n{2,}|$))/i;
        this._defRegExp = /^ {0,3}\[((?!\s*\])(?:\\[[\]]|[^[\]])+)\]: *\n? *<?([^\s>]+)>?(?:(?: +\n? *| *\n *)((?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))))? *(?:\n+|$)/;
        this._nptableRegExp = /^ *([^|\n ].*\|.*)\n {0,3}([-:]+ *\|[-| :]*)(?:\n((?:(?!\n| {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {4}[^\n]| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--)).*(?:\n|$))*)\n*|$)/;
        this._tableRegExp = /^ *\|(.+)\n {0,3}\|?( *[-:]+[-| :]*)(?:\n *((?:(?!\n| {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {4}[^\n]| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--)).*(?:\n|$))*)\n*|$)/;
        this._lheadingRegExp = /^([^\n]+)\n {0,3}(=+|-+) *(?:\n+|$)/;
        this._textRegExp = /^[^\n]+/;
        this._bulletRegExp = /(?:[*+-]|\d{1,9}[.)])/;
        this._itemRegExp = /^( *)((?:[*+-]|\d{1,9}[.)])) ?[^\n]*(?:\n(?!\1(?:[*+-]|\d{1,9}[.)]) ?)[^\n]*)*/gm;
        this._paragraphRegExp = /^([^\n]+(?:\n(?! {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--))[^\n]+)*)/;
        this._backpedalRegExp = /(?:[^?!.,:;*_~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_~)]+(?!$))+/;
        this._escapeRegExp = /^\\([!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~~|])/;
        this._escapesRegExp = /\\([!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~])/g;
        /* eslint-disable no-control-regex */
        this._autolinkRegExp = /^<([a-zA-Z][a-zA-Z0-9+.-]{1,31}:[^\s\x00-\x1f<>]*|[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_]))>/;
        this._linkRegExp = /^!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\(\s*(<(?:\\[<>]?|[^\s<>\\])*>|[^\s\x00-\x1f]*)(?:\s+("(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)))?\s*\)/;
        /* eslint-enable no-control-regex */
        this._urlRegExp = /^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9-]+\.?)+[^\s<]*|^[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/i;
        this._tagRegExp = /^<!--(?!-?>)[\s\S]*?-->|^<\/[a-zA-Z][\w:-]*\s*>|^<[a-zA-Z][\w-]*(?:\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?)*?\s*\/?>|^<\?[\s\S]*?\?>|^<![a-zA-Z]+\s[\s\S]*?>|^<!\[CDATA\[[\s\S]*?\]\]>/;
        this._reflinkRegExp = /^!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\[(?!\s*\])((?:\\[[\]]?|[^[\]\\])+)\]/;
        this._nolinkRegExp = /^!?\[(?!\s*\])((?:\[[^[\]]*\]|\\[[\]]|[^[\]])*)\](?:\[\])?/;
        this._reflinkSearchRegExp = /!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\[(?!\s*\])((?:\\[[\]]?|[^[\]\\])+)\]|!?\[(?!\s*\])((?:\[[^[\]]*\]|\\[[\]]|[^[\]])*)\](?:\[\])?(?!\()/g;
        this._strongStartRegExp = /^(?:(\*\*(?=[*!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]))|\*\*)(?![\s])|__/;
        this._strongMiddleRegExp = /^\*\*(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|\*(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?\*)+?\*\*$|^__(?![\s])((?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|_(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?_)+?)__$/;
        this._strongEndAstRegExp = /[^!"#$%&'()+\-.,/:;<=>?@[\]`{|}~\s]\*\*(?!\*)|[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]\*\*(?!\*)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~_\s]|$))/g;
        this._strongEndUndRegExp = /[^\s]__(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g;
        this._emStartRegExp = /^(?:(\*(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]))|\*)(?![*\s])|_/;
        this._emMiddleRegExp = /^\*(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|\*(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?\*)+?\*$|^_(?![_\s])(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|_(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?_)+?_$/;
        this._emEndAstRegExp = /[^!"#$%&'()+\-.,/:;<=>?@[\]`{|}~\s]\*(?!\*)|[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]\*(?!\*)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~_\s]|$))/g;
        this._emEndUndRegExp = /[^\s]_(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g;
        this._codespanRegExp = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/;
        this._brRegExp = /^( {2,}|\\)\n(?!\s*$)/;
        this._delRegExp = /^~+(?=\S)([\s\S]*?\S)~+/;
        this._textspanRegExp = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<![`*~]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+/=?_`{|}~-](?=[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+@))|(?=[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+@))/;
        this._punctuationRegExp = /^([\s*!"#$%&'()+\-.,/:;<=>?@[\]`{|}~])/;
        this._blockSkipRegExp = /\[[^\]]*?\]\([^)]*?\)|`[^`]*?`|<[^>]*?>/g;
        this._escapeTestRegExp = /[&<>"']/;
        this._escapeReplaceRegExp = /[&<>"']/g;
        this._escapeTestNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/;
        this._escapeReplaceNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/g;
        this._escapeReplacementsMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
        this._cache = new Map();
    }

    html(source) {
        if (this._cache.has(source)) {
            return this._cache.get(source);
        }
        const tokens = [];
        const links = new Map();
        source = source.replace(/\r\n|\r/g, '\n').replace(/\t/g, '    ');
        this._tokenize(source, tokens, links, true);
        this._tokenizeBlock(tokens, links);
        const target = this._render(tokens, true);
        if (this._cache.size > 256) {
            this._cache.delete(this._cache.keys().next().value);
        }
        this._cache.set(source, target);
        return target;
    }

    _tokenize(source, tokens, links, top) {
        source = source.replace(/^ +$/gm, '');
        while (source) {
            let match = this._newlineRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                if (match[0].length > 1) {
                    tokens.push({ type: 'space' });
                }
                continue;
            }
            match = this._codeRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const lastToken = tokens[tokens.length - 1];
                if (lastToken && lastToken.type === 'paragraph') {
                    lastToken.text += `\n${match[0].trimRight()}`;
                } else {
                    const text = match[0].replace(/^ {4}/gm, '').replace(/\n*$/, '');
                    tokens.push({ type: 'code', text });
                }
                continue;
            }
            match = this._fencesRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const language = match[2] ? match[2].trim() : match[2];
                let content = match[3] || '';
                const matchIndent = match[0].match(/^(\s+)(?:```)/);
                if (matchIndent !== null) {
                    const [, indent] = matchIndent;
                    content = content.split('\n').map((node) => {
                        const match = node.match(/^\s+/);
                        return (match !== null && match[0].length >= indent.length) ? node.slice(indent.length) : node;
                    }).join('\n');
                }
                tokens.push({ type: 'code', language, text: content });
                continue;
            }
            match = this._headingRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'heading', depth: match[1].length, text: match[2] });
                continue;
            }
            match = this._nptableRegExp.exec(source);
            if (match) {
                const header = this._splitCells(match[1].replace(/^ *| *\| *$/g, ''));
                const align = match[2].replace(/^ *|\| *$/g, '').split(/ *\| */);
                if (header.length === align.length) {
                    const cells = match[3] ? match[3].replace(/\n$/, '').split('\n') : [];
                    const token = { type: 'table', header, align, cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        } else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        } else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        } else {
                            token.align[i] = null;
                        }
                    }
                    token.cells = token.cells.map((cell) => this._splitCells(cell, token.header.length));
                    source = source.substring(token.raw.length);
                    tokens.push(token);
                    continue;
                }
            }
            match = this._hrRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'hr' });
                continue;
            }
            match = this._blockquoteRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = match[0].replace(/^ *> ?/gm, '');
                tokens.push({ type: 'blockquote', text, tokens: this._tokenize(text, [], links, top) });
                continue;
            }
            match = this._listRegExp.exec(source);
            if (match) {
                const [value, , bull] = match;
                const ordered = bull.length > 1;
                const parent = bull[bull.length - 1] === ')';
                let raw = value;
                const list = { type: 'list', raw, ordered, start: ordered ? Number(bull.slice(0, -1)) : '', loose: false, items: [] };
                const itemMatch = value.match(this._itemRegExp);
                let next = false;
                const length = itemMatch.length;
                for (let i = 0; i < length; i++) {
                    let item = itemMatch[i];
                    raw = item;
                    let space = item.length;
                    item = item.replace(/^ *([*+-]|\d+[.)]) ?/, '');
                    if (item.indexOf('\n ') !== -1) {
                        space -= item.length;
                        item = item.replace(new RegExp(`^ {1,${space}}`, 'gm'), '');
                    }
                    if (i !== length - 1) {
                        const [bullet] = this._bulletRegExp.exec(itemMatch[i + 1]);
                        if (ordered ? bullet.length === 1 || (!parent && bullet[bullet.length - 1] === ')') : (bullet.length > 1)) {
                            const addBack = itemMatch.slice(i + 1).join('\n');
                            list.raw = list.raw.substring(0, list.raw.length - addBack.length);
                            i = length - 1;
                        }
                    }
                    let loose = next || /\n\n(?!\s*$)/.test(item);
                    if (i !== length - 1) {
                        next = item.charAt(item.length - 1) === '\n';
                        if (!loose) {
                            loose = next;
                        }
                    }
                    if (loose) {
                        list.loose = true;
                    }
                    const task = /^\[[ xX]\] /.test(item);
                    let checked = false;
                    if (task) {
                        checked = item[1] !== ' ';
                        item = item.replace(/^\[[ xX]\] +/, '');
                    }
                    list.items.push({ type: 'list_item', raw, task, checked, loose, text: item });
                }
                source = source.substring(list.raw.length);
                for (const item of list.items) {
                    item.tokens = this._tokenize(item.text, [], links, false);
                }
                tokens.push(list);
                continue;
            }
            match = this._htmlRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'html', pre: (match[1] === 'pre' || match[1] === 'script' || match[1] === 'style'), text: match[0] });
                continue;
            }
            if (top) {
                match = this._defRegExp.exec(source);
                if (match) {
                    source = source.substring(match[0].length);
                    match[3] = match[3] ? match[3].substring(1, match[3].length - 1) : match[3];
                    const tag = match[1].toLowerCase().replace(/\s+/g, ' ');
                    if (!links.has(tag)) {
                        links.set(tag, { href: match[2], title: match[3] });
                    }
                    continue;
                }
            }
            match = this._tableRegExp.exec(source);
            if (match) {
                const header = this._splitCells(match[1].replace(/^ *| *\| *$/g, ''));
                const align = match[2].replace(/^ *|\| *$/g, '').split(/ *\| */);
                if (header.length === align.length) {
                    const cells = match[3] ? match[3].replace(/\n$/, '').split('\n') : [];
                    const token = { type: 'table', header, align, cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        } else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        } else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        } else {
                            token.align[i] = null;
                        }
                    }
                    token.cells = token.cells.map((cell) => this._splitCells(cell.replace(/^ *\| *| *\| *$/g, ''), token.header.length));
                    source = source.substring(token.raw.length);
                    tokens.push(token);
                    continue;
                }
            }
            match = this._lheadingRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'heading', depth: match[2].charAt(0) === '=' ? 1 : 2, text: match[1] });
                continue;
            }
            if (top) {
                match = this._paragraphRegExp.exec(source);
                if (match) {
                    source = source.substring(match[0].length);
                    tokens.push({ type: 'paragraph', text: match[1].charAt(match[1].length - 1) === '\n' ? match[1].slice(0, -1) : match[1] });
                    continue;
                }
            }
            match = this._textRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const lastToken = tokens[tokens.length - 1];
                if (lastToken && lastToken.type === 'text') {
                    lastToken.text += `\n${match[0]}`;
                } else {
                    tokens.push({ type: 'text', text: match[0] });
                }
                continue;
            }
            throw new Error(`Unexpected '${source.charCodeAt(0)}'.`);
        }
        return tokens;
    }

    _tokenizeInline(source, links, inLink, inRawBlock, prevChar) {
        const tokens = [];
        let maskedSource = source;
        if (links.size > 0) {
            while (maskedSource) {
                const match = this._reflinkSearchRegExp.exec(maskedSource);
                if (match) {
                    if (links.has(match[0].slice(match[0].lastIndexOf('[') + 1, -1))) {
                        maskedSource = `${maskedSource.slice(0, match.index)}[${'a'.repeat(match[0].length - 2)}]${maskedSource.slice(this._reflinkSearchRegExp.lastIndex)}`;
                    }
                    continue;
                }
                break;
            }
        }
        while (maskedSource) {
            const match = this._blockSkipRegExp.exec(maskedSource);
            if (match) {
                maskedSource = `${maskedSource.slice(0, match.index)}[${'a'.repeat(match[0].length - 2)}]${maskedSource.slice(this._blockSkipRegExp.lastIndex)}`;
                continue;
            }
            break;
        }
        while (source) {
            let match = this._escapeRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'escape', text: this._escape(match[1]) });
                continue;
            }
            match = this._tagRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                if (!inLink && /^<a /i.test(match[0])) {
                    inLink = true;
                } else if (inLink && /^<\/a>/i.test(match[0])) {
                    inLink = false;
                }
                if (!inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = true;
                } else if (inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = false;
                }
                tokens.push({ type: 'html', raw: match[0], text: match[0] });
                continue;
            }
            match = this._linkRegExp.exec(source);
            if (match) {
                let index = -1;
                const [, , ref] = match;
                if (ref.indexOf(')') !== -1) {
                    let level = 0;
                    for (let i = 0; i < ref.length; i++) {
                        switch (ref[i]) {
                            case '\\':
                                i++;
                                break;
                            case '(':
                                level++;
                                break;
                            case ')':
                                level--;
                                if (level < 0) {
                                    index = i;
                                    i = ref.length;
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
                if (index > -1) {
                    const length = (match[0].indexOf('!') === 0 ? 5 : 4) + match[1].length + index;
                    match[2] = match[2].substring(0, index);
                    match[0] = match[0].substring(0, length).trim();
                    match[3] = '';
                }
                const title = (match[3] ? match[3].slice(1, -1) : '').replace(this._escapesRegExp, '$1');
                const href = match[2].trim().replace(/^<([\s\S]*)>$/, '$1').replace(this._escapesRegExp, '$1');
                const token = this._outputLink(match, href, title);
                source = source.substring(match[0].length);
                if (token.type === 'link') {
                    token.tokens = this._tokenizeInline(token.text, links, true, inRawBlock, '');
                }
                tokens.push(token);
                continue;
            }
            match = this._reflinkRegExp.exec(source) || this._nolinkRegExp.exec(source);
            if (match) {
                let link = (match[2] || match[1]).replace(/\s+/g, ' ');
                link = links.get(link.toLowerCase());
                if (!link || !link.href) {
                    const text = match[0].charAt(0);
                    source = source.substring(text.length);
                    tokens.push({ type: 'text', text });
                } else {
                    source = source.substring(match[0].length);
                    const token = this._outputLink(match, link);
                    if (token.type === 'link') {
                        token.tokens = this._tokenizeInline(token.text, links, true, inRawBlock, '');
                    }
                    tokens.push(token);
                }
                continue;
            }
            match = this._strongStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '**' ? this._strongEndAstRegExp : this._strongEndUndRegExp;
                endReg.lastIndex = 0;
                let cap = '';
                while ((match = endReg.exec(masked)) !== null) {
                    cap = this._strongMiddleRegExp.exec(masked.slice(0, match.index + 3));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.substring(2, cap[0].length - 2);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'strong', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._emStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '*' ? this._emEndAstRegExp : this._emEndUndRegExp;
                endReg.lastIndex = 0;
                let cap = '';
                while ((match = endReg.exec(masked)) !== null) {
                    cap = this._emMiddleRegExp.exec(masked.slice(0, match.index + 2));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.slice(1, cap[0].length - 1);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'em', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._codespanRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                let content = match[2].replace(/\n/g, ' ');
                if (/[^ ]/.test(content) && content.startsWith(' ') && content.endsWith(' ')) {
                    content = content.substring(1, content.length - 1);
                }
                tokens.push({ type: 'codespan', text: this._encode(content) });
                continue;
            }
            match = this._brRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'br' });
                continue;
            }
            match = this._delRegExp.exec(source);
            if (match) {
                const [value, text] = match;
                source = source.substring(value.length);
                tokens.push({ type: 'del', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                continue;
            }
            match = this._autolinkRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = this._escape(match[1]);
                const href = match[2] === '@' ? `mailto:${text}` : text;
                tokens.push({ type: 'link', text, href, tokens: [{ type: 'text', raw: text, text }] });
                continue;
            }
            if (!inLink) {
                match = this._urlRegExp.exec(source);
                if (match) {
                    const email = match[2] === '@';
                    let [value] = match;
                    if (!email) {
                        let prevCapZero = '';
                        do {
                            prevCapZero = value;
                            [value] = this._backpedalRegExp.exec(value);
                        } while (prevCapZero !== value);
                    }
                    const text = this._escape(value);
                    let href = text;
                    if (email) {
                        href = `mailto:${text}`;
                    } else if (text.startsWith('www.')) {
                        href = `http://${text}`;
                    }
                    source = source.substring(value.length);
                    tokens.push({ type: 'link', text, href, tokens: [{ type: 'text', text }] });
                    continue;
                }
            }
            match = this._textspanRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                prevChar = match[0].slice(-1);
                tokens.push({ type: 'text' , text: inRawBlock ? match[0] : this._escape(match[0]) });
                continue;
            }
            throw new Error(`Unexpected '${source.charCodeAt(0)}'.`);
        }
        return tokens;
    }

    _tokenizeBlock(tokens, links) {
        for (const token of tokens) {
            switch (token.type) {
                case 'paragraph':
                case 'text':
                case 'heading': {
                    token.tokens  = this._tokenizeInline(token.text, links, false, false, '');
                    break;
                }
                case 'table': {
                    token.tokens = {};
                    token.tokens.header = token.header.map((header) => this._tokenizeInline(header, links, false, false, ''));
                    token.tokens.cells = token.cells.map((cell) => cell.map((row) => this._tokenizeInline(row, links, false, false, '')));
                    break;
                }
                case 'blockquote': {
                    this._tokenizeBlock(token.tokens, links);
                    break;
                }
                case 'list': {
                    for (const item of token.items) {
                        this._tokenizeBlock(item.tokens, links);
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }

    _render(tokens, top) {
        let html = '';
        while (tokens.length > 0) {
            const token = tokens.shift();
            switch (token.type) {
                case 'space': {
                    continue;
                }
                case 'hr': {
                    html += '<hr>\n';
                    continue;
                }
                case 'heading': {
                    const level = token.depth;
                    html += `<h${level}>${this._renderInline(token.tokens)}</h${level}>\n`;
                    continue;
                }
                case 'code': {
                    const code = token.text;
                    const [language] = (token.language || '').match(/\S*/);
                    html += `<pre><code${language ? ` class="language-${this._encode(language)}"` : ''}>${token.escaped ? code : this._encode(code)}</code></pre>\n`;
                    continue;
                }
                case 'table': {
                    let header = '';
                    let cell = '';
                    for (let j = 0; j < token.header.length; j++) {
                        const content = this._renderInline(token.tokens.header[j]);
                        const align = token.align[j];
                        cell += `<th${align ? ` align="${align}"` : ''}>${content}</th>\n`;
                    }
                    header += `<tr>\n${cell}</tr>\n`;
                    let body = '';
                    for (let j = 0; j < token.cells.length; j++) {
                        const row = token.tokens.cells[j];
                        cell = '';
                        for (let k = 0; k < row.length; k++) {
                            const content = this._renderInline(row[k]);
                            const align = token.align[k];
                            cell += `<td${align ? ` align="${align}"` : ''}>${content}</td>\n`;
                        }
                        body += `<tr>\n${cell}</tr>\n`;
                    }
                    html += `<table>\n<thead>\n${header}</thead>\n${body ? `<tbody>${body}</tbody>` : body}</table>\n`;
                    continue;
                }
                case 'blockquote': {
                    html += `<blockquote>\n${this._render(token.tokens, true)}</blockquote>\n`;
                    continue;
                }
                case 'list': {
                    const ordered = token.ordered;
                    const start = token.start;
                    const loose = token.loose;
                    let body = '';
                    for (const item of token.items) {
                        let itemBody = '';
                        if (item.task) {
                            const checkbox = `<input ${item.checked ? 'checked="" ' : ''}disabled="" type="checkbox"> `;
                            if (loose) {
                                if (item.tokens.length > 0 && item.tokens[0].type === 'text') {
                                    item.tokens[0].text = `${checkbox} ${item.tokens[0].text}`;
                                    if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === 'text') {
                                        item.tokens[0].tokens[0].text = `${checkbox} ${item.tokens[0].tokens[0].text}`;
                                    }
                                } else {
                                    item.tokens.unshift({ type: 'text', text: checkbox });
                                }
                            } else {
                                itemBody += checkbox;
                            }
                        }
                        itemBody += this._render(item.tokens, loose);
                        body += `<li>${itemBody}</li>\n`;
                    }
                    const type = (ordered ? 'ol' : 'ul');
                    html += `<${type}${ordered && start !== 1 ? (` start="${start}"`) : ''}>\n${body}</${type}>\n`;
                    continue;
                }
                case 'html': {
                    html += token.text;
                    continue;
                }
                case 'paragraph': {
                    html += `<p>${this._renderInline(token.tokens)}</p>\n`;
                    continue;
                }
                case 'text': {
                    html += top ? '<p>' : '';
                    html += token.tokens ? this._renderInline(token.tokens) : token.text;
                    while (tokens.length > 0 && tokens[0].type === 'text') {
                        const token = tokens.shift();
                        html += `\n${token.tokens ? this._renderInline(token.tokens) : token.text}`;
                    }
                    html += top ? '</p>\n' : '';
                    continue;
                }
                default: {
                    throw new Error(`Unexpected token type '${token.type}'.`);
                }
            }
        }
        return html;
    }

    _renderInline(tokens) {
        let html = '';
        for (const token of tokens) {
            switch (token.type) {
                case 'escape':
                case 'html':
                case 'text': {
                    html += token.text;
                    break;
                }
                case 'link': {
                    const text = this._renderInline(token.tokens);
                    html += `<a href="${token.href}"${token.title ? ` title="${token.title}"` : ''} target="_blank">${text}</a>`;
                    break;
                }
                case 'image': {
                    html += `<img src="${token.href}" alt="${token.text}"${token.title ? ` title="${token.title}"` : ''}>`;
                    break;
                }
                case 'strong': {
                    const text = this._renderInline(token.tokens);
                    html += `<strong>${text}</strong>`;
                    break;
                }
                case 'em': {
                    const text = this._renderInline(token.tokens);
                    html += `<em>${text}</em>`;
                    break;
                }
                case 'codespan': {
                    html += `<code>${token.text}</code>`;
                    break;
                }
                case 'br': {
                    html += '<br>';
                    break;
                }
                case 'del': {
                    const text = this._renderInline(token.tokens);
                    html += `<del>${text}</del>`;
                    break;
                }
                default: {
                    throw new Error(`Unexpected token type '${token.type}'.`);
                }
            }
        }
        return html;
    }

    _outputLink(match, href, title) {
        title = title ? this._escape(title) : null;
        const text = match[1].replace(/\\([[\]])/g, '$1');
        return match[0].charAt(0) === '!' ?
            { type: 'image', href, title, text: this._escape(text) } :
            { type: 'link', href, title, text };
    }

    _splitCells(tableRow, count) {
        const row = tableRow.replace(/\|/g, (match, offset, str) => {
            let escaped = false;
            let position = offset;
            while (--position >= 0 && str[position] === '\\') {
                escaped = !escaped;
            }
            return escaped ? '|' : ' |';
        });
        const cells = row.split(/ \|/);
        if (cells.length > count) {
            cells.splice(count);
        } else {
            while (cells.length < count) {
                cells.push('');
            }
        }
        return cells.map((cell) => cell.trim().replace(/\\\|/g, '|'));
    }

    _encode(content) {
        if (this._escapeTestRegExp.test(content)) {
            return content.replace(this._escapeReplaceRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }

    _escape(content) {
        if (this._escapeTestNoEncodeRegExp.test(content)) {
            return content.replace(this._escapeReplaceNoEncodeRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }
};

png.Encoder = class {

    constructor(window, width, height) {
        this.width = width;
        this.height = height;
        const compressor = new window.CompressionStream('deflate');
        this.writer = compressor.writable.getWriter();
        this.response = new window.Response(compressor.readable).blob();
    }

    async write(data, rows) {
        const bytesPerRow = this.width * 4;
        const filtered = new Uint8Array(rows * (1 + bytesPerRow));
        let offset = 0;
        let dataOffset = 0;
        for (let i = 0; i < rows; i++) {
            filtered[offset++] = 0;
            filtered.set(data.subarray(dataOffset, dataOffset + bytesPerRow), offset);
            offset += bytesPerRow;
            dataOffset += bytesPerRow;
        }
        await this.writer.write(filtered);
    }

    async toBuffer() {
        await this.writer.close();
        const blob = await this.response;
        const arrayBuffer = await blob.arrayBuffer();
        const compressed = new Uint8Array(arrayBuffer);
        const crc32Table = new Uint32Array(256);
        for (let i = 0; i < 256; i++) {
            let c = i;
            for (let j = 0; j < 8; j++) {
                c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
            }
            crc32Table[i] = c;
        }
        const crc32 = (buffer, offset, length) => {
            let crc = 0xFFFFFFFF;
            for (let i = 0; i < length; i++) {
                crc = crc32Table[(crc ^ buffer[offset + i]) & 0xFF] ^ (crc >>> 8);
            }
            return (crc ^ 0xFFFFFFFF) >>> 0;
        };
        const buffer = new Uint8Array(57 + compressed.length);
        const view = new DataView(buffer.buffer);
        // Signature
        buffer.set([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], 0);
        // IHDR
        view.setUint32(8, 13, false);
        buffer.set([0x49, 0x48, 0x44, 0x52], 12);
        view.setUint32(16, this.width, false);
        view.setUint32(20, this.height, false);
        buffer.set([8, 6, 0, 0, 0], 24);
        view.setUint32(29, crc32(buffer, 12, 17), false);
        // IDAT
        view.setUint32(33, compressed.length, false);
        buffer.set([0x49, 0x44, 0x41, 0x54], 37);
        buffer.set(compressed, 41);
        view.setUint32(41 + compressed.length, crc32(buffer, 37, 4 + compressed.length), false);
        // IEND
        buffer.set([0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82], 45 + compressed.length);
        return buffer;
    }
};

metadata.Attachment = class {

    constructor() {
        this.metadata = new metadata.Attachment.Container('metadata');
        this.metrics = new metadata.Attachment.Container('metrics');
        this.quantization = new aimet.EncodingFile();
    }

    async open(context) {
        context = new view.Context(context);
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.json') || identifier.endsWith('.encodings')) {
            const data = await context.peek('json');
            if (data && data.signature === 'netron:attachment') {
                const containers = [this.metadata, this.metrics];
                for (const container of containers) {
                    container.open(data[container.name]);
                }
                return true;
            }
            if (this.quantization.open(data)) {
                return true;
            }
        }
        return false;
    }

    bind(model) {
        if (!this.quantization.empty) {
            this.quantization.bind(model);
        }
    }

};

metadata.Attachment.Container = class {

    constructor(name) {
        this._name = name;
        this._entries = new Map();
    }

    get name() {
        return this._name;
    }

    open(data) {
        this._entries.clear();
        if (Array.isArray(data)) {
            for (const item of data) {
                if (item.kind && ('target' in item || 'identifier' in item)) {
                    const key = 'target' in item ? `${item.kind}::${item.target}` : `${item.kind}[${item.identifier}]`;
                    if (!this._entries.has(key)) {
                        this._entries.set(key, new Map());
                    }
                    const entries = this._entries.get(key);
                    entries.set(item.name, { value: item.value, type: item.type });
                }
            }
        }
    }

    model(value) {
        return this._list(value, 'model');
    }

    graph(value) {
        return this._list(value, 'graph');
    }

    node(value) {
        return this._list(value, 'node');
    }

    value(value) {
        return this._list(value, 'value');
    }

    tensor(value) {
        return this._list(value, 'tensor');
    }

    _list(value, kind) {
        const category = this._name;
        const entries = value[category] || [];
        const result = new Map(entries.map((entry) => [entry.name, entry]));
        if (value.name || kind === 'model' || kind === 'graph') {
            const key = `${kind}::${(value.name || '').split('\n').shift()}`;
            if (this._entries.has(key)) {
                for (const [name, entry] of this._entries.get(key)) {
                    const argument = new metadata.Argument(name, entry.value, entry.type || 'attribute');
                    result.set(name, argument);
                }
            }
        }
        if (value.identifier) {
            const key = `${kind}[${value.identifier}]`;
            if (this._entries.has(key)) {
                for (const [name, entry] of this._entries.get(key)) {
                    const argument = new metadata.Argument(name, entry.value, entry.type || 'attribute');
                    result.set(name, argument);
                }
            }
        }
        return Array.from(result.values());
    }
};

metadata.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

metrics.Model = class {

    constructor(model) {
        this._model = model;
        this._metrics = null;
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._model.metrics || []);
            const keys = new Set(this._metrics.map((metric) => metric.name));
            if (!keys.has('parameters')) {
                let parameters = 0;
                for (const graph of this._model.graphs || []) {
                    const map = new Map((new metrics.Target(graph).metrics || []).map((metric) => [metric.name, metric]));
                    parameters = map.has('parameters') ? parameters + map.get('parameters').value : NaN;
                }
                for (const func of this._model.functions || []) {
                    const map = new Map((new metrics.Target(func).metrics || []).map((metric) => [metric.name, metric]));
                    parameters = map.has('parameters') ? parameters + map.get('parameters').value : NaN;
                }
                if (!Number.isNaN(parameters) && parameters > 0) {
                    this._metrics.push(new metadata.Argument('parameters', parameters, 'attribute'));
                }
            }
        }
        return this._metrics;
    }
};

metrics.Target = class {

    constructor(target) {
        this._target = target;
        this._metrics = null;
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._target.metrics || []);
            const keys = new Set(this._metrics.map((metrics) => metrics.name));
            if (!keys.has('parameters')) {
                let parameters = 0;
                const initializers = new Set();
                if (this._target && Array.isArray(this._target.nodes)) {
                    for (const node of this._target.nodes) {
                        for (const argument of node.inputs || []) {
                            if (argument && Array.isArray(argument.value)) {
                                for (const value of argument.value) {
                                    if (value && value.initializer) {
                                        initializers.add(value.initializer);
                                    }
                                }
                            }
                        }
                    }
                }
                for (const tensor of initializers) {
                    const shape = tensor && tensor.type && tensor.type.shape && Array.isArray(tensor.type.shape.dimensions) ? tensor.type.shape.dimensions : [];
                    if (!shape.every((dim) => typeof dim === 'number' || typeof dim === 'bigint')) {
                        parameters = 0;
                        break;
                    }
                    parameters += shape.reduce((a, b) => BigInt(a) * BigInt(b), 1n).toNumber();
                }
                if (parameters > 0) {
                    this._metrics.push(new metadata.Argument('parameters', parameters, 'attribute'));
                }
            }
        }
        return this._metrics;
    }
};

metrics.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this._metrics = null;
    }

    get name() {
        return this._tensor.name || '';
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._tensor.metrics || []);
            const keys = new Set(this._metrics.map((metrics) => metrics.name));
            const type = this._tensor.type;
            const shape = type.shape.dimensions;
            const size = shape.reduce((a, b) => BigInt(a) * BigInt(b), 1n).toNumber();
            if (size < 0x800000 &&
                (type.dataType.startsWith('float') || type.dataType.startsWith('bfloat')) &&
                (!keys.has('sparsity') || !keys.has('min') || !keys.has('max') && !keys.has('mean') || !keys.has('max') || !keys.has('std'))) {
                const data = this._tensor.value;
                let zeros = 0;
                let min = null;
                let max = null;
                let sum = 0;
                let count = 0;
                const stack = [data];
                while (stack.length > 0) {
                    const data = stack.pop();
                    if (Array.isArray(data)) {
                        for (const element of data) {
                            stack.push(element);
                        }
                    } else {
                        zeros += data === 0 || data === 0n || data === '';
                        min = Math.min(data, min === null ? data : min);
                        max = Math.max(data, max === null ? data : max);
                        sum += data;
                        count += 1;
                    }
                }
                const mean = sum / count;
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metadata.Argument('min', min, type.dataType));
                }
                if (!keys.has('max')) {
                    this._metrics.push(new metadata.Argument('max', max, type.dataType));
                }
                if (!keys.has('mean')) {
                    this._metrics.push(new metadata.Argument('mean', mean, type.dataType));
                }
                if (!keys.has('std')) {
                    let variance = 0;
                    const stack = [data];
                    while (stack.length > 0) {
                        const data = stack.pop();
                        if (Array.isArray(data)) {
                            for (const element of data) {
                                stack.push(element);
                            }
                        } else {
                            variance += Math.pow(data - mean, 2);
                        }
                    }
                    this._metrics.push(new metadata.Argument('std', Math.sqrt(variance / count)));
                }
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metadata.Argument('sparsity', count > 0 ? zeros / count : 0, 'percentage'));
                }
            }
        }
        return this._metrics;
    }
};

view.Context = class {

    constructor(context, identifier, stream) {
        this._context = context;
        this._tags = new Map();
        this._content = new Map();
        this._stream = stream || context.stream;
        identifier = typeof identifier === 'string' ? identifier : context.identifier;
        const index = Math.max(identifier.lastIndexOf('/'), identifier.lastIndexOf('\\'));
        this._base = index === -1 ? undefined : identifier.substring(0, index);
        this._identifier = index === -1 ? identifier : identifier.substring(index + 1);
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    get container() {
        if (this._context instanceof view.Container) {
            return this._context;
        }
        return null;
    }

    async asset(file) {
        return this._context.asset(file);
    }

    async fetch(file) {
        const stream = await this._context.fetch(file, null, this._base);
        return new view.Context(this._context, file, stream);
    }

    context(identifier, stream, entries) {
        if (stream instanceof Uint8Array) {
            stream = new base.BinaryStream(stream);
        }
        const context = entries instanceof Map ? new view.Container(this._context, this._identifier, entries) : this._context;
        return new view.Context(context, identifier, stream);
    }

    async require(id) {
        return this._context.require(id);
    }

    error(error, fatal) {
        if (error && this.identifier) {
            error.context = this.container ? this.container.identifier : this.identifier;
        }
        this._context.error(error, fatal);
    }

    set(type, value) {
        this.type = type;
        this.value = value;
        return type;
    }

    async peek(type) {
        if (!this._content.has(type)) {
            this._content.set(type, undefined);
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const match = (buffer, signature) => {
                    return signature.length <= buffer.length && buffer.every((value, index) => signature[index] === undefined || signature[index] === value);
                };
                const buffer = stream.peek(Math.min(stream.length, 16));
                const skip =
                    match(buffer, [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19]) || // PyTorch
                    (type !== 'npz' && type !== 'zip' && match(buffer, [0x50, 0x4B, 0x03, 0x04])) || // ZIP
                    (type !== 'hdf5' && match(buffer, [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A])) || // \x89HDF\r\n\x1A\n
                    Array.from(this._tags).some(([key, value]) => key !== 'flatbuffers' && key !== 'xml' && value.size > 0) ||
                    Array.from(this._content.values()).some((obj) => obj !== undefined);
                if (!skip) {
                    switch (type) {
                        case 'json': {
                            try {
                                const buffer = stream.peek(Math.min(stream.length, 0x1000));
                                if (stream.length < 0x7ffff000 &&
                                    (buffer.length < 8 || String.fromCharCode.apply(null, buffer.slice(0, 8)) !== '\x89HDF\r\n\x1A\n') &&
                                    (buffer.some((v) => v === 0x22 || v === 0x5b || v === 0x5d || v === 0x7b || v === 0x7d))) {
                                    const json = await import('./json.js');
                                    const reader = json.TextReader.open(stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'json.gz': {
                            try {
                                const entries = await this.peek('gzip');
                                if (entries && entries.size === 1) {
                                    const stream = entries.values().next().value;
                                    const json = await import('./json.js');
                                    const reader = json.TextReader.open(stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'xml': {
                            try {
                                const buffer = stream.peek(Math.min(this.stream.length, 0x1000));
                                const content = String.fromCharCode.apply(null, buffer);
                                if (stream.length < 0x7ffff000 && content.indexOf('<') !== -1 && content.indexOf('</') !== -1) {
                                    const xml = await import('./xml.js');
                                    const reader = xml.TextReader.open(this._stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'pkl': {
                            let unpickler = null;
                            const types = new Set();
                            try {
                                const zip = await import('./zip.js');
                                const archive = zip.Archive.open(stream, 'zlib');
                                const data = archive ? archive.entries.get('') : stream;
                                let condition = false;
                                if (data.length > 4) {
                                    const head = data.peek(4);
                                    condition = head[0] === 0x80 && head[1] < 7;
                                    if (!condition) {
                                        data.seek(-1);
                                        const tail = data.peek(1);
                                        data.seek(0);
                                        if (tail[0] === 0x2e) {
                                            const size = Math.min(data.length, 256);
                                            const buffer = data.peek(size);
                                            condition =
                                                (buffer[0] === 0x28 && buffer[1] === 0x64 && buffer[2] === 0x70) ||
                                                (buffer[0] === 0x28 && buffer[1] === 0x63 && buffer.indexOf(0x0a) !== -1);
                                            if (!condition) {
                                                const content = String.fromCharCode.apply(null, buffer);
                                                const list = ['ccopy_reg', 'cnumpy.core.multiarray', '(dp0'];
                                                condition = list.some((value) => content.indexOf(value) !== -1);
                                            }
                                        }
                                    }
                                }
                                if (condition) {
                                    const python = await import('./python.js');
                                    const execution = new python.Execution();
                                    execution.on('resolve', (sender, name) => types.add(name));
                                    const pickle = execution.__import__('pickle');
                                    unpickler = new pickle.Unpickler(data);
                                }
                            } catch {
                                // continue regardless of error
                            }
                            if (unpickler) {
                                const storages = new Map();
                                unpickler.persistent_load = (saved_id) => {
                                    if (Array.isArray(saved_id) && saved_id.length > 3) {
                                        switch (saved_id[0]) {
                                            case 'storage': {
                                                const [, storage_type, key, , size] = saved_id;
                                                if (!storages.has(key)) {
                                                    const storage = new storage_type(size);
                                                    storages.set(key, storage);
                                                }
                                                return storages.get(key);
                                            }
                                            default: {
                                                throw new view.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
                                            }
                                        }
                                    }
                                    throw new view.Error("Unsupported 'persistent_load'.");
                                };
                                try {
                                    const obj = unpickler.load();
                                    this._content.set(type, obj);
                                } catch (error) {
                                    this._content.set(type, error);
                                }
                                if (Array.from(types).every((name) => !name.startsWith('__torch__.'))) {
                                    for (const name of types) {
                                        this.error(new view.Error(`Unknown type name '${name}'.`));
                                    }
                                }
                            }
                            break;
                        }
                        case 'hdf5': {
                            const hdf5 = await import('./hdf5.js');
                            const file = hdf5.File.open(stream);
                            if (file) {
                                try {
                                    this._content.set(type, file.read());
                                } catch (error) {
                                    this._content.set(type, error);
                                }
                            }
                            break;
                        }
                        case 'zip':
                        case 'tar':
                        case 'gzip': {
                            this._content.set('zip', undefined);
                            this._content.set('tar', undefined);
                            this._content.set('gzip', undefined);
                            let stream = this._stream;
                            try {
                                const zip = await import('./zip.js');
                                const archive = zip.Archive.open(this._stream, 'gzip');
                                if (archive) {
                                    let entries = archive.entries;
                                    if (entries.size === 1) {
                                        const key = entries.keys().next().value;
                                        stream = entries.values().next().value;
                                        const name = key === '' ? this.identifier.replace(/\.gz$/, '') : key;
                                        entries = new Map([[name, stream]]);
                                    }
                                    this._content.set('gzip', entries);
                                }
                            } catch (error) {
                                this._content.set('gzip', error);
                            }
                            let skipTar = false;
                            try {
                                const zip = await import('./zip.js');
                                const archive = zip.Archive.open(stream, 'zip');
                                if (archive) {
                                    this._content.set('zip', archive.entries);
                                    skipTar = true;
                                }
                            } catch (error) {
                                this._content.set('zip', error);
                            }
                            if (!skipTar) {
                                try {
                                    const tar = await import('./tar.js');
                                    const archive = tar.Archive.open(stream);
                                    if (archive) {
                                        this._content.set('tar', archive.entries);
                                    }
                                } catch (error) {
                                    this._content.set('tar', error);
                                }
                            }
                            break;
                        }
                        case 'flatbuffers.binary': {
                            try {
                                const flatbuffers = await import('./flatbuffers.js');
                                const reader = flatbuffers.BinaryReader.open(this._stream);
                                if (reader) {
                                    this._content.set('flatbuffers.binary', reader);
                                }
                            } catch (error) {
                                this._content.set('flatbuffers.binary', error);
                            }
                            break;
                        }
                        case 'npz': {
                            const content = new Map();
                            const entries = await this.peek('zip');
                            if (entries instanceof Map && entries.size > 0 &&
                                Array.from(entries.keys()).every((name) => name.endsWith('.npy'))) {
                                const python = await import('./python.js');
                                const execution = new python.Execution();
                                const io = execution.__import__('io');
                                const numpy = execution.__import__('numpy');
                                for (const [name, stream] of entries) {
                                    const bytes = new io.BytesIO(stream);
                                    const array = numpy.load(bytes);
                                    content.set(name, array);
                                }
                                this._content.set(type, content);
                            }
                            break;
                        }
                        default: {
                            throw new view.Error(`Unsupported open format type '${type}'.`);
                        }
                    }
                }
                if (stream.position !== position) {
                    stream.seek(0);
                }
            }
        }
        return this._content.get(type);
    }

    async read(type, ...args) {
        if (!this._content.has(type)) {
            switch (type) {
                case 'json': {
                    const json = await import('./json.js');
                    const reader = json.TextReader.open(this._stream);
                    if (reader) {
                        const obj = reader.read();
                        this._content.set('json', obj);
                        return obj;
                    }
                    throw new view.Error('Invalid JSON content.');
                }
                case 'bson': {
                    const json = await import('./json.js');
                    const reader = json.BinaryReader.open(this._stream);
                    if (reader) {
                        return reader.read();
                    }
                    throw new view.Error('Invalid BSON content.');
                }
                case 'xml': {
                    const xml = await import('./xml.js');
                    const reader = xml.TextReader.open(this._stream);
                    if (reader) {
                        return reader.read();
                    }
                    throw new view.Error(`Invalid XML content.`);
                }
                case 'flatbuffers.binary': {
                    const flatbuffers = await import('./flatbuffers.js');
                    const reader = flatbuffers.BinaryReader.open(this._stream);
                    if (reader) {
                        this._content.set('flatbuffers.reader', reader);
                        return reader;
                    }
                    throw new view.Error('Invalid FlatBuffers content.');
                }
                case 'flatbuffers.text': {
                    const flatbuffers = await import('./flatbuffers.js');
                    const obj = await this.peek('json');
                    return flatbuffers.TextReader.open(obj);
                }
                case 'protobuf.binary': {
                    const protobuf = await import('./protobuf.js');
                    return protobuf.BinaryReader.open(this._stream);
                }
                case 'protobuf.text': {
                    const protobuf = await import('./protobuf.js');
                    return protobuf.TextReader.open(this._stream);
                }
                case 'binary.big-endian': {
                    return base.BinaryReader.open(this._stream, false);
                }
                case 'binary': {
                    return base.BinaryReader.open(this._stream);
                }
                case 'text': {
                    const text = await import('./text.js');
                    if (typeof args[0] === 'number') {
                        const length = Math.min(this._stream.length, args[0]);
                        const buffer = this._stream.peek(length);
                        return text.Reader.open(buffer);
                    }
                    return text.Reader.open(this._stream);
                }
                case 'text.decoder': {
                    const text = await import('./text.js');
                    return text.Decoder.open(this._stream);
                }
                default: {
                    break;
                }
            }
        }
        return this.peek(type);
    }

    async tags(type) {
        if (!this._tags.has(type)) {
            let tags = new Map();
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const signatures = [
                    [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A], // HDF5
                    [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19], // PyTorch
                    [0x50, 0x4b], // ZIP
                    [0x1f, 0x8b] // gzip
                ];
                let skip = false;
                if (signatures.some((signature) => signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value))) {
                    skip = true;
                } else if (Array.from(this._tags).some(([key, value]) => key !== 'flatbuffers' && value.size > 0) && type !== 'pb+') {
                    skip = true;
                } else if (Array.from(this._content.values()).some((obj) => obj !== undefined)) {
                    skip = true;
                } else if (stream.length < 0x7ffff000) {
                    const json = await import('./json.js');
                    if (json.TextReader.open(stream)) {
                        skip = true;
                    }
                }
                if (!skip && stream.length < 0x7ffff000) {
                    try {
                        switch (type) {
                            case 'pbtxt': {
                                const protobuf = await import('./protobuf.js');
                                const reader = protobuf.TextReader.open(stream);
                                tags = reader ? reader.signature() : tags;
                                break;
                            }
                            case 'pb': {
                                const protobuf = await import('./protobuf.js');
                                const reader = protobuf.BinaryReader.open(stream);
                                tags = reader.signature();
                                break;
                            }
                            case 'pb+': {
                                const protobuf = await import('./protobuf.js');
                                const reader = protobuf.BinaryReader.open(stream);
                                tags = reader.decode();
                                break;
                            }
                            case 'xml': {
                                const xml = await import('./xml.js');
                                const reader = xml.TextReader.open(stream);
                                if (reader) {
                                    const document = reader.read(1);
                                    const element = document.documentElement;
                                    const namespaceURI = element.namespaceURI;
                                    const localName = element.localName;
                                    const name = namespaceURI ? `${namespaceURI}:${localName}` : localName;
                                    tags.set(name, element);
                                }
                                break;
                            }
                            default: {
                                throw new view.Error(`Unsupported tags format type '${type}'.`);
                            }
                        }
                    } catch {
                        tags.clear();
                    }
                }
                if (stream.position !== position) {
                    stream.seek(position);
                }
            }
            this._tags.set(type, tags);
        }
        return this._tags.get(type);
    }

    async metadata(name) {
        return view.Metadata.open(this, name);
    }
};

view.Container = class {

    constructor(host, identifier, entries) {
        this._host = host;
        this._identifier = identifier;
        this._entries = entries;
    }

    get identifier() {
        return this._identifier;
    }

    async asset(file) {
        return this._host.asset(file);
    }

    async fetch(file, encoding, base) {
        let stream = null;
        if (typeof base === 'string') {
            stream = this._entries.get(`${base}/${file}`) || this._entries.get(`${base}\\${file}`);
        } else {
            stream = this._entries.get(file);
        }
        if (!stream) {
            throw new view.Error('File not found.');
        }
        if (encoding) {
            const decoder = new TextDecoder(encoding);
            const buffer = stream.peek();
            return decoder.decode(buffer);
        }
        return stream;
    }

    async require(id) {
        return this._host.require(id);
    }

    get entries() {
        return this._entries;
    }

    error(error, fatal) {
        this._host.exception(error, fatal);
    }
};

view.ArchiveError = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading archive.';
    }
};

view.ModelFactoryService = class {

    constructor(host) {
        this._host = host;
        this._patterns = new Set(['.zip', '.tar', '.tar.gz', '.tgz', '.gz']);
        this._factories = [];
        /* eslint-disable no-control-regex */
        this.register('./message', ['.message', '.netron', '.maxviz']);
        this.register('./pytorch', ['.pt', '.pth', '.ptl', '.pt1', '.pt2', '.pyt', '.pyth', '.pkl', '.pickle', '.h5', '.t7', '.model', '.dms', '.tar', '.ckpt', '.chkpt', '.tckpt', '.bin', '.pb', '.zip', '.nn', '.torchmodel', '.torchscript', '.pytorch', '.ot', '.params', '.trt', '.ff', '.ptmf', '.jit', '.bin.index.json', 'model.json', '.ir', 'serialized_exported_program.json', 'serialized_state_dict.json', 'archive_format'], ['.model', '.pt2'], [/^\x80.\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19/]);
        this.register('./onnx', ['.onnx', '.onnx.data', '.onnx.meta', '.onn', '.pb', '.onnxtxt', '.pbtxt', '.prototxt', '.txt', '.model', '.pt', '.pth', '.pkl', '.ort', '.ort.onnx', '.ngf', '.json', '.bin', 'onnxmodel'], [], [/^\x08[\x00-\x10]\x12[\x00-\x20]\w\w/, /^\x08[\x00-\x10]\x12\x00\x1A/, /^\x08[\x00-\x10]\x3A/, /^\s*ir_version:\s\d+/, /^....ORTM/]);
        this.register('./litertlm', ['.litertlm'], [], [/^LITERTLM/]);
        this.register('./tflite', ['.tflite', '.lite', '.tfl', '.bin', '.pb', '.tmfile', '.h5', '.model', '.json', '.txt', '.dat', '.nb', '.ckpt', '.onnx'], [], [/^....TFL3/]);
        this.register('./mxnet', ['.json', '.params'], ['.mar']);
        this.register('./coreml', ['.mlmodel', '.bin', 'manifest.json', 'metadata.json', 'featuredescriptions.json', '.pb', '.pbtxt', '.mil'], ['.mlpackage', '.mlmodelc']);
        this.register('./caffe', ['.caffemodel', '.pbtxt', '.prototxt', '.pt', '.txt']);
        this.register('./caffe2', ['.pb', '.pbtxt', '.prototxt']);
        this.register('./torch', ['.t7', '.net']);
        this.register('./tf', ['.pb', '.meta', '.pbtxt', '.prototxt', '.txt', '.pt', '.json', '.index', '.ckpt', '.graphdef', '.pbmm', /.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/, /^events.out.tfevents./, /^.*group\d+-shard\d+of\d+(\.bin)?$/], ['.zip']);
        this.register('./tensorrt', ['.trt', '.trtmodel', '.engine', '.model', '.txt', '.uff', '.pb', '.tmfile', '.onnx', '.pth', '.dnn', '.plan', '.pt', '.dat', '.bin'], [], [/^ptrt/, /^ftrt/]);
        this.register('./keras', ['.h5', '.hd5', '.hdf5', '.keras', '.json', '.cfg', '.model', '.pb', '.pth', '.weights', '.pkl', '.lite', '.tflite', '.ckpt', '.pb', 'model.weights.npz', /^.*group\d+-shard\d+of\d+(\.bin)?$/], ['.zip'], [/^\x89HDF\r\n\x1A\n/]);
        this.register('./safetensors', ['.safetensors', '.safetensors.index.json', 'safetensors-index.json']);
        this.register('./numpy', ['.npz', '.npy', '.pkl', '.pickle', '.model', '.model2', '.mge', '.joblib'], [], [/^\x93NUMPY/, /^PK\x03\x04/]);
        this.register('./lasagne', ['.pkl', '.pickle', '.joblib', '.model', '.pkl.z', '.joblib.z']);
        this.register('./lightgbm', ['.txt', '.pkl', '.model']);
        this.register('./sklearn', ['.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z', '.pickle.dat', '.bin']);
        this.register('./megengine', ['.tm', '.mge', '.pkl']);
        this.register('./pickle', ['.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z', '.pdstates', '.mge', '.bin', '.npy', '.pth']);
        this.register('./cntk', ['.model', '.cntk', '.cmf', '.dnn']);
        this.register('./uff', ['.uff', '.pb', '.pbtxt', '.uff.txt', '.trt', '.engine']);
        this.register('./paddle', ['.pdmodel', '.pdiparams', '.pdparams', '.pdopt', '.paddle', '__model__', '.__model__', '.pbtxt', '.txt', '.tar', '.tar.gz', '.nb', '.json']);
        this.register('./bigdl', ['.model', '.bigdl']);
        this.register('./darknet', ['.cfg', '.model', '.txt', '.weights']);
        this.register('./mediapipe', ['.pbtxt']);
        this.register('./executorch', ['.pte'], [], [/^....ET12/]);
        this.register('./rknn', ['.rknn', '.nb', '.onnx', '.json', '.bin', /^model$/], [], [/^RKNN/, /^VPMN/], /^....RKNN/);
        this.register('./dlc', ['.dlc', /^model$/, '.params']);
        this.register('./armnn', ['.armnn', '.json']);
        this.register('./mnn', ['.mnn']);
        this.register('./ncnn', ['.param', '.bin', '.cfg.ncnn', '.weights.ncnn', '.ncnnmodel']);
        this.register('./tnn', ['.tnnproto', '.tnnmodel']);
        this.register('./tengine', ['.tmfile']);
        this.register('./mslite', ['.ms', '.bin'], [], [/^....MSL0/, /^....MSL1/, /^....MSL2/]);
        this.register('./mindir', ['.mindir']);
        this.register('./barracuda', ['.nn']);
        this.register('./circle', ['.circle'], [], [/^....CIR0/]);
        this.register('./dnn', ['.dnn']);
        this.register('./xmodel', ['.xmodel']);
        this.register('./kmodel', ['.kmodel']);
        this.register('./flux', ['.bson']);
        this.register('./dl4j', ['.json', '.bin']);
        this.register('./openvino', ['.xml', '.bin']);
        this.register('./mlnet', ['.zip', '.mlnet']);
        this.register('./acuity', ['.json']);
        this.register('./imgdnn', ['.dnn', 'params', '.json']);
        this.register('./flax', ['.msgpack']);
        this.register('./om', ['.om', '.onnx', '.pb', '.engine', '.bin'], [], [/^IMOD/, /^PICO/]);
        this.register('./gguf', ['.gguf', /^[^.]+$/], [], [/^GGUF/]);
        this.register('./nnabla', ['.nntxt'], ['.nnp']);
        this.register('./hickle', ['.h5', '.hkl']);
        this.register('./nnef', ['.nnef', '.dat']);
        this.register('./onednn', ['.json']);
        this.register('./espresso', ['.espresso.net', '.espresso.shape', '.espresso.weights'], ['.mlmodelc']);
        this.register('./mlir', ['.mlir', '.mlir.txt', '.mlirbc', '.txt']);
        this.register('./sentencepiece', ['.model']);
        this.register('./hailo', ['.hn', '.har', '.metadata.json']);
        this.register('./tvm', ['.json', '.params']);
        this.register('./dot', ['.dot'], [], [/^\s*(\/\*[\s\S]*?\*\/|\/\/.*|#.*)?\s*digraph\s*([A-Za-z][A-Za-z0-9-_]*|".*?")?\s*{/m]);
        this.register('./jax', ['.jax', '.jax_export', '.jax_exported']);
        this.register('./catboost', ['.cbm', '.pkl'], [], [/^CBM1/]);
        this.register('./weka', ['.model']);
        this.register('./qnn', ['.json', '.bin', '.serialized', '.dlc']);
        this.register('./espdl', ['.espdl'], [], [/^EDL2/]);
        this.register('./kann', ['.kann', '.bin', '.kgraph'], [], [/^....KaNN/]);
        this.register('./xgboost', ['.xgb', '.xgboost', '.json', '.model', '.bin', '.txt', '.ubj'], [], [/^{L\x00\x00/, /^binf/, /^bs64/, /^\s*booster\[0\]:/]);
        this.register('./tosa', ['.tosa', '.json'], [], [/^....TOSA/]);
        this.register('./transformers', ['.json']);
        this.register('', ['.cambricon', '.vnnmodel', '.nnc']);
        /* eslint-enable no-control-regex */
    }

    register(module, extensions, containers, contents) {
        for (const extension of extensions) {
            this._factories.push({ extension, module });
            this._patterns.add(extension);
        }
        for (const content of contents || []) {
            this._factories.push({ content, module });
        }
        for (const container of containers || []) {
            this._patterns.add(container);
        }
    }

    async open(context) {
        try {
            await this._openSignature(context);
            const content = new view.Context(context);
            let model = await this._openContext(content);
            if (!model) {
                const check = (obj) => {
                    if (obj instanceof Error) {
                        throw obj;
                    }
                    return obj instanceof Map && obj.size > 0;
                };
                let entries = context.entries;
                if (!check(entries)) {
                    entries = await content.peek('zip');
                    if (!check(entries)) {
                        entries = await content.peek('tar');
                        if (!check(entries)) {
                            entries = await content.peek('gzip');
                        }
                    }
                }
                if (!check(entries)) {
                    await this._unsupported(content);
                }
                const container = await this._openEntries(entries, context.identifier);
                if (!container) {
                    await this._unsupported(content);
                }
                model = await this._openContext(container);
            }
            if (!model.format || typeof model.format !== 'string' || model.format.length === 0) {
                throw new view.Error('Invalid model format name.');
            }
            if (!/^[a-zA-Z][a-zA-Z0-9-.]*( [a-zA-Z][a-zA-Z0-9-.]*)*( v\d+(\.\d+)*(b\d+)?([.+-][a-zA-Z0-9]+)?)?$/.test(model.format) || model.format.includes('undefined')) {
                throw new view.Error(`Invalid model format name '${model.format}'.`);
            }
            if (model.producer && /[^\x20-\x7E\u00C0-\u00FF\u0370-\u03FF]/.test(model.producer)) {
                throw new view.Error(`Invalid model producer name '${model.producer}'.`);
            }
            return model;
        } catch (error) {
            if (!error.context && context) {
                error.context = context.identifier || '';
                const stream = context.stream;
                if (stream) {
                    try {
                        const hex = (buffer) => Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                        const position = stream.position;
                        stream.seek(0);
                        const head = stream.peek(Math.min(16, stream.length));
                        error.context += `|${hex(head)}`;
                        if (stream.length > 16) {
                            stream.seek(stream.length - 16);
                            const tail = stream.peek(16);
                            error.context += `|${hex(tail)}`;
                        }
                        stream.seek(position);
                    } catch {
                        // continue regardless of error
                    }
                }
            }
            throw error;
        }
    }

    async _unsupported(context) {
        const identifier = context.identifier;
        const stream = context.stream;
        const zip = await import('./zip.js');
        const tar = await import('./tar.js');
        const callbacks = [
            (stream) => zip.Archive.open(stream, 'zip'),
            (stream) => tar.Archive.open(stream),
            (stream) => zip.Archive.open(stream, 'gzip')
        ];
        for (const callback of callbacks) {
            let archive = null;
            try {
                archive = callback(stream);
            } catch {
                // continue regardless of error
            }
            if (archive) {
                throw new view.Error("Archive contains no model files.");
            }
        }
        const regex = async() => {
            if (stream) {
                const entries = [
                    { name: 'Unity metadata', value: /fileFormatVersion:/ },
                ];
                const buffer = stream.peek(Math.min(4096, stream.length));
                const content = String.fromCharCode.apply(null, buffer);
                for (const entry of entries) {
                    if (content.match(entry.value) && (!entry.identifier || context.identifier.match(entry.identifier))) {
                        throw new view.Error(`Invalid file content. File contains ${entry.name}.`);
                    }
                }
            }
        };
        const json = async () => {
            const obj = await context.peek('json');
            if (obj) {
                const formats = [
                    { name: 'Netron metadata', tags: ['[].name', '[].schema'] },
                    { name: 'Netron metadata', tags: ['[].name', '[].attributes'] },
                    { name: 'Netron metadata', tags: ['[].name', '[].category'] },
                    { name: 'Netron test data', tags: ['[].type', '[].target', '[].source', '[].format', '[].link'] },
                    { name: 'Netron configuration', tags: ['recents', 'consent'] },
                    { name: 'Netron metrics data', tags: ['signature', 'metrics'] },
                    { name: 'Darkflow metadata', tags: ['net', 'type', 'model'] },
                    { name: 'keras-yolo2 configuration', tags: ['model', 'train', 'valid'] },
                    { name: 'Vulkan SwiftShader ICD manifest', tags: ['file_format_version', 'ICD'] },
                    { name: 'DeepLearningExamples configuration', tags: ['attention_probs_dropout_prob', 'hidden_act', 'hidden_dropout_prob', 'hidden_size',] },
                    { name: 'GitHub page data', tags: ['payload', 'title'] },
                    { name: 'NuGet assets', tags: ['version', 'targets', 'packageFolders'] },
                    { name: 'NuGet data', tags: ['format', 'restore', 'projects'] },
                    { name: 'NPM package', tags: ['name', 'version', 'dependencies'] },
                    { name: 'NPM package lock data', tags: ['name', 'version', 'lockfileVersion'] },
                    { name: 'NetworkX adjacency_data', tags: ['directed', 'graph', 'nodes'] },
                    { name: 'Waifu2x data', tags: ['name', 'arch_name', 'channels'] },
                    { name: 'Waifu2x data', tags: ['[].nInputPlane', '[].nOutputPlane', '[].weight', '[].bias'] },
                    { name: 'Brain.js data', tags: ['type', 'sizes', 'layers'] },
                    { name: 'Custom Vision metadata', tags: ['CustomVision.Metadata.Version'] },
                    { name: 'W&B metadata', tags: ['program', 'host', 'executable'] },
                    { name: 'TypeScript configuration data', tags: ['compilerOptions'] },
                    { name: 'CatBoost model', tags: ['features_info', 'model_info'] },
                    { name: 'TPU-MLIR tensor location data', tags: ['file-line', 'subnet_id', 'core_id'] }, // https://github.com/sophgo/tpu-mlir/blob/master/lib/Dialect/Tpu/Transforms/Codegen/TensorLocation.cpp
                    { name: 'HTTP Archive data', tags: ['log.version', 'log.creator', 'log.entries'] }, // https://w3c.github.io/web-performance/specs/HAR/Overview.html
                    { name: 'Trace Event data', tags: ['traceEvents'] },
                    { name: 'Trace Event data', tags: ['[].pid', '[].ph'] },
                    { name: 'Diffusers configuration', tags: ['_class_name', '_diffusers_version'] },
                    { name: 'ModelScope configuration', tags: ['framework', 'task'] }, // https://github.com/modelscope/modelscope
                    { name: 'Tokenizer data', tags: ['<eos>', '<bos>'] },
                    { name: 'Jupyter Notebook data', tags: ['cells', 'nbformat'] },
                    { name: 'Kaggle credentials', tags: ['username','key'] },
                    { name: '.NET runtime configuration', tags: ['runtimeOptions.configProperties'] },
                    { name: '.NET dependency manifest', tags: ['runtimeTarget', 'targets', 'libraries'] },
                    { name: 'GuitarML NeuralPi model data', tags: ['model_data', 'state_dict'] },
                    { name: 'GuitarML SmartAmpPro model data', tags: ['conv1d', 'conv1d_stride'] },
                    { name: 'GuitarML SmartAmp model data', tags: ['activation', 'output_channels', 'input_channels', 'residual_channels'] },
                    { name: 'Keras configuration data', tags: ['floatx', 'epsilon', 'backend'] },
                    { name: 'PIMCOMP-NN model data', tags: ['node_list', 'reshape_info'] },
                    { name: 'AIMET encodings', tags: ['activation_encodings'] },
                    { name: 'COCO annotations', tags: ['images', 'annotations', 'categories'] }, // https://cocodataset.org/
                    { name: 'Sentence Transformers modules', tags: ['[].idx', '[].path', '[].type'] }, // https://www.sbert.net/
                    { name: 'Sentence Transformers configuration', tags: ['__version__.sentence_transformers'] }, // https://www.sbert.net/
                    { name: 'Lottie animation', tags: ['v', 'fr', 'ip', 'op', 'w', 'h', 'layers'] }, // https://lottiefiles.github.io/lottie-docs/
                    { name: 'OCI image manifest', tags: ['schemaVersion', 'mediaType'] }, // https://github.com/opencontainers/image-spec
                    { name: 'LabelMe annotation', tags: ['version', 'flags', 'shapes'] }, // https://github.com/labelmeai/labelme
                    { name: 'Ollama model manifest', tags: ['model_format', 'model_family'] }, // https://github.com/ollama/ollama
                ];
                const match = (obj, tag) => {
                    if (tag.startsWith('[].')) {
                        tag = tag.substring(3);
                        return (Array.isArray(obj) && obj.some((item) => Object.prototype.hasOwnProperty.call(item, tag)));
                    }
                    tag = tag.split('.');
                    while (tag.length > 1) {
                        const key = tag.shift();
                        obj = obj[key];
                        if (!obj) {
                            return false;
                        }
                    }
                    return Object.prototype.hasOwnProperty.call(obj, tag[0]);
                };
                for (const format of formats) {
                    if (format.tags.every((tag) => match(obj, tag))) {
                        throw new view.Error(`Invalid file content. File contains ${format.name}.`);
                    }
                }
                const content = `${JSON.stringify(obj).substring(0, 100).replace(/\s/, '').substring(0, 48)}...`;
                throw new view.Error(`Unsupported JSON content '${content.length > 64 ? `${content.substring(0, 100)}...` : content}'.`);
            }
        };
        const pbtxt = async () => {
            const formats = [
                { name: 'ImageNet LabelMap data', tags: ['entry', 'entry.target_class'] },
                { name: 'StringIntLabelMapProto data', tags: ['item', 'item.id', 'item.name'] },
                { name: 'caffe.LabelMap data', tags: ['item', 'item.name', 'item.label'] },
                { name: 'Triton Inference Server configuration', tags: ['input', 'output', 'name', 'platform'] }, // https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
                { name: 'Triton Inference Server configuration', tags: ['input', 'output', 'backend'] },
                { name: 'Triton Inference Server configuration', tags: ['input', 'output', 'max_batch_size'] },
                { name: 'Triton Inference Server configuration', tags: ['input', 'output', 'instance_group'] },
                { name: 'Triton Inference Server configuration', tags: ['default_model_filename', 'max_batch_size'] },
                { name: 'TensorFlow OpList data', tags: ['op', 'op.name', 'op.input_arg'] },
                { name: 'vitis.ai.proto.DpuModelParamList data', tags: ['model', 'model.name', 'model.kernel'] },
                { name: 'object_detection.protos.DetectionModel data', tags: ['model', 'model.ssd'] },
                { name: 'object_detection.protos.DetectionModel data', tags: ['model', 'model.faster_rcnn'] },
                { name: 'tensorflow.CheckpointState data', tags: ['model_checkpoint_path', 'all_model_checkpoint_paths'] },
                { name: 'apollo.perception.camera.traffic_light.detection.DetectionParam data', tags: ['min_crop_size', 'crop_method'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['caffe_ssd'] }, // https://github.com/TexasInstruments/edgeai-mmdetection/blob/master/mmdet/utils/proto/mmdet_meta_arch.proto
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tf_od_api_ssd'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_ssd'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_faster_rcnn'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_yolo'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_retinanet'] },
                { name: 'domi.InsertNewOps data', tags: ['aipp_op'] } // https://github.com/Ascend/parser/blob/development/parser/proto/insert_op.proto
            ];
            const tags = await context.tags('pbtxt');
            if (tags.size > 0) {
                for (const format of formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const entries = [];
                entries.push(...Array.from(tags).filter(([key]) => key.toString().indexOf('.') === -1));
                entries.push(...Array.from(tags).filter(([key]) => key.toString().indexOf('.') !== -1));
                const content = entries.map(([key, value]) => value === true ? key : `${key}:${JSON.stringify(value)}`).join(',');
                throw new view.Error(`Unsupported Protocol Buffers text content '${content.length > 64 ? `${content.substring(0, 100)}...` : content}'.`);
            }
        };
        const pb = async () => {
            const tags = await context.tags('pb+');
            if (Object.keys(tags).length > 0) {
                const formats = [
                    { name: 'sentencepiece.ModelProto data', tags: [[1,[[1,2],[2,5],[3,0]]],[2,[[1,2],[2,2],[3,0],[4,0],[5,2],[6,0],[7,2],[10,5],[16,0],[40,0],[41,0],[42,0],[43,0]]],[3,[]],[4,[]],[5,[]]] }, // https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
                    { name: 'mediapipe.BoxDetectorIndex data', tags: [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]] }, // https://github.com/google-ai-edge/mediapipe/blob/2b5a50fff37f79db8103dbd88f552c1a9be31e51/mediapipe/util/tracking/box_detector.proto
                    { name: 'third_party.tensorflow.python.keras.protobuf.SavedMetadata data', tags: [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]] },
                    { name: 'pblczero.Net data', tags: [[1,5],[2,2],[3,[[1,0],[2,0],[3,0]],[10,[[1,[]],[2,[]],[3,[]],[4,[]],[5,[]],[6,[]]]],[11,[]]]] }, // https://github.com/LeelaChessZero/lczero-common/blob/master/proto/net.proto
                    { name: 'chrome_browser_media.PreloadedData', tags: [[1,2]], identifier: 'preloaded_data.pb' }, // https://github.com/kiwibrowser/src/blob/86afd150b847c9dd6f9ad3faddee1a28b8c9b23b/chrome/browser/media/media_engagement_preload.proto#L9
                    { name: 'mindspore.irpb.Checkpoint', tags: [[1,[[1,2],[2,[[1,0],[2,2],[3,2]]]]]] }, // https://github.com/mindspore-ai/mindspore/blob/master/mindspore/ccsrc/utils/checkpoint.proto
                    { name: 'optimization_guide.proto.PageTopicsOverrideList data', tags: [[1,[[1,2],[2,[]]]]] }, // https://github.com/chromium/chromium/blob/main/components/optimization_guide/proto/page_topics_override_list.proto
                    { name: 'optimization_guide.proto.ModelInfo data', tags: [[1,0],[2,0],[4,0],[6,false],[7,[]],[9,0]] }, // https://github.com/chromium/chromium/blob/22b0d711657b451b61d50dd2e242b3c6e38e6ef5/components/optimization_guide/proto/models.proto#L80
                    { name: 'Horizon binary model', tags: [[1,0],[2,0],[5,[[7,2],[8,2]]],[6,[[1,[[1,2],[2,2]]]]]] }, // https://github.com/HorizonRDK/hobot_dnn
                    { name: 'TensorFlow Profiler data', tags: [[1,[[2,2],[3,[]],[4,[]]]]] }, // https://github.com/tensorflow/tensorflow/blob/master/third_party/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto
                ];
                const match = (tags, schema) => {
                    for (const [key, inner] of schema) {
                        const value = tags[key];
                        if (value === undefined) {
                            continue;
                        }
                        if (inner === false) {
                            return false;
                        }
                        if (Array.isArray(inner)) {
                            if (typeof value !== 'object' || !match(value, inner)) {
                                return false;
                            }
                        } else if (inner !== value) {
                            if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return true;
                };
                for (const format of formats) {
                    if (match(tags, format.tags) && (!format.identifier || identifier === context.identifier)) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const format = (tags) => {
                    const content = Object.entries(tags).map(([key, value]) => {
                        return `${key}:${Object(value) === value ? `{${format(value)}}` : value}`;
                    });
                    return content.join(',');
                };
                const content = format(tags);
                const message = content.length > 64 ? `${content.substring(0, 100)}...` : content;
                throw new view.Error(`Unsupported Protocol Buffers content or ambiguous file extension '${message}'.`);
            }
        };
        const flatbuffers = async () => {
            const stream = context.stream;
            if (stream && stream.length >= 8) {
                let identifier = null;
                const reader = await context.peek('flatbuffers.binary');
                if (reader) {
                    identifier = reader.identifier;
                } else {
                    const data = stream.peek(8);
                    if (data[0] >= 8 && data[0] <= 0x28 && (data[0] & 3) === 0 && data[1] === 0x00 && data[2] === 0x00 && data[3] === 0x00) {
                        identifier = String.fromCharCode.apply(null, data.slice(4, 8));
                    }
                }
                if (identifier) {
                    const formats = [
                        { name: 'ONNX Runtime model data', identifier: 'ORTM' },
                        { name: 'TensorFlow Lite model data', identifier: 'TFL3' },
                        { name: 'ExecuTorch model data', identifier: 'ET12' },
                        { name: 'NNC model data', identifier: 'ENNC' },
                        { name: 'KaNN model data', identifier: 'KaNN' },
                        { name: 'Circle model data', identifier: 'CIR0' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL0' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL1' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL2' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL3' },
                        { name: 'NVDA model data', identifier: 'NVDA' },
                        { name: 'BSTM model data', identifier: 'BSTM' },
                        { name: 'onnu model data', identifier: 'onnu' },
                        { name: 'ONNX Runtime On-Device Training Checkpoint', identifier: 'ODTC' },
                        { name: 'TOSA model data', identifier: 'TOSA' }
                    ];
                    for (const format of formats) {
                        if (identifier === format.identifier) {
                            throw new view.Error(`Invalid file content. File contains ${format.name}.`);
                        }
                    }
                }
            }
        };
        const xml = async () => {
            const document = await context.peek('xml');
            if (document && document.documentElement) {
                const tags = new Set();
                const qualifiedName = (element) => {
                    const namespaceURI = element.namespaceURI;
                    const localName = element.localName;
                    return namespaceURI ? `${namespaceURI}:${localName}` : localName;
                };
                const root = qualifiedName(document.documentElement);
                tags.add(root);
                for (const element of document.documentElement.childNodes) {
                    const name = qualifiedName(element);
                    tags.add(`${root}/${name}`);
                }
                const formats = [
                    { name: 'OpenCV storage data', tags: ['opencv_storage'] },
                    { name: 'XHTML markup', tags: ['http://www.w3.org/1999/xhtml:html'] },
                    { name: '.NET XML documentation', tags: ['doc', 'doc/assembly'] },
                    { name: '.NET XML documentation', tags: ['doc', 'doc/members'] }
                ];
                for (const format of formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.content = context.identifier;
                        throw error;
                    }
                }
                throw new view.Error(`Unsupported XML content '${tags.keys().next().value}'.`);
            }
        };
        const hdf5 = async () => {
            const obj = await context.peek('hdf5');
            if (obj instanceof Error) {
                throw obj;
            }
            if (obj) {
                throw new view.Error(`Invalid file content. File contains HDF5 content.`);
            }
        };
        const unknown = async () => {
            if (stream) {
                throw new view.Error(`Unsupported file content.`);
            }
            throw new view.Error("Unsupported file directory.");
        };
        await regex();
        await json();
        await pbtxt();
        await pb();
        await flatbuffers();
        await xml();
        await hdf5();
        await unknown();
    }

    async _require(id) {
        const module = await this._host.require(id);
        if (!module || !module.ModelFactory) {
            throw new view.Error(`Failed to load module '${id}'.`);
        }
        return new module.ModelFactory();
    }

    async _openContext(context) {
        const modules = this._filter(context).filter((module) => module && module.length > 0);
        const errors = [];
        for (const module of modules) {
            // eslint-disable-next-line no-await-in-loop
            const factory = await this._require(module);
            // eslint-disable-next-line no-await-in-loop
            const type = await factory.match(context);
            if (context.stream && context.stream.position !== 0) {
                throw new view.Error('Invalid stream position.');
            }
            if (type) {
                try {
                    // eslint-disable-next-line no-await-in-loop
                    const model = await factory.open(context);
                    if (!model.identifier) {
                        model.identifier = context.identifier;
                    }
                    model.attachment = new metadata.Attachment();
                    return model;
                } catch (error) {
                    delete context.type;
                    delete context.value;
                    const stream = context.stream;
                    if (stream && stream.position !== 0) {
                        stream.seek(0);
                    }
                    errors.push(error);
                }
            }
            if (context.stream && context.stream.position !== 0) {
                throw new view.Error('Invalid stream position.');
            }
        }
        if (errors.length > 0) {
            if (errors.length === 1) {
                throw errors[0];
            }
            throw new view.Error(errors.map((err) => err.message).join('\n'));
        }
        return null;
    }

    async _openEntries(entries, identifier) {
        try {
            const rootFolder = (files) => {
                const map = files.map((file) => file.split('/').slice(0, -1));
                const at = (index) => (list) => list[index];
                const rotate = (list) => list.length === 0 ? [] : list[0].map((item, index) => list.map(at(index)));
                const equals = (list) => list.every((item) => item === list[0]);
                const folder = rotate(map).filter(equals).map(at(0)).join('/');
                return folder.length === 0 ? folder : `${folder}/`;
            };
            const files = Array.from(entries).filter(([name]) => !(name.endsWith('/') || name.split('/').pop().startsWith('.') || (!name.startsWith('./') && name.startsWith('.'))));
            const folder = rootFolder(files.map(([name]) => name));
            const filter = async (queue, entries) => {
                entries = new Map(Array.from(entries)
                    .filter(([path]) => path.startsWith(folder))
                    .map(([path, stream]) => [path.substring(folder.length), stream]));
                const container = new view.Container(this._host, identifier, entries);
                let matches = [];
                for (const [name, stream] of queue) {
                    const identifier = name.substring(folder.length);
                    const context = new view.Context(container, identifier, stream);
                    const modules = this._filter(context);
                    for (const module of modules) {
                        // eslint-disable-next-line no-await-in-loop
                        const factory = await this._require(module);
                        // eslint-disable-next-line no-await-in-loop
                        const type = await factory.match(context);
                        if (context.stream && context.stream.position !== 0) {
                            throw new view.Error('Invalid stream position.');
                        }
                        delete context.value;
                        if (type) {
                            matches = matches.filter((match) => !factory.filter || factory.filter(context, match));
                            if (matches.every((match) => !match.factory.filter || match.factory.filter(match, context))) {
                                context.factory = factory;
                                matches.push(context);
                            }
                            break;
                        }
                    }
                }
                if (matches.length > 1) {
                    const content = matches.map((context) => context.type).join(',');
                    throw new view.ArchiveError(`Archive contains multiple model files '${content}'.`);
                }
                if (matches.length > 0) {
                    const match = matches.shift();
                    delete match.type;
                    delete match.factory;
                    return match;
                }
                return null;
            };
            const queue = files.filter(([name]) => name.substring(folder.length).indexOf('/') < 0);
            let context = await filter(queue, entries);
            if (!context) {
                const queue = files.filter(([name]) => name.substring(folder.length).indexOf('/') >= 0);
                context = await filter(queue, entries);
            }
            return context;
        } catch (error) {
            throw new view.ArchiveError(error.message);
        }
    }

    accept(identifier, size) {
        const extension = identifier.indexOf('.') === -1 ? '' : identifier.split('.').pop().toLowerCase();
        identifier = identifier.toLowerCase().split('/').pop();
        let accept = false;
        for (const extension of this._patterns) {
            if ((typeof extension === 'string' &&
                    ((extension !== '' && identifier.endsWith(extension)) ||
                     (extension === '' && identifier.indexOf('.') === -1))) ||
                (extension instanceof RegExp && extension.exec(identifier))) {
                accept = true;
                break;
            }
        }
        this._host.event('model_file', {
            file_extension: extension,
            file_size: size || 0,
            file_accept: accept ? 1 : 0
        });
        return accept;
    }

    _filter(context) {
        const identifier = context.identifier.toLowerCase().split('/').pop();
        const stream = context.stream;
        if (stream) {
            const buffer = stream.peek(Math.min(4096, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            const list = this._factories.filter((entry) =>
                (typeof entry.extension === 'string' && identifier.endsWith(entry.extension)) ||
                (entry.extension instanceof RegExp && entry.extension.test(identifier)) ||
                (entry.content instanceof RegExp && entry.content.test(content)));
            return Array.from(new Set(list.map((entry) => entry.module)));
        }
        return [];
    }

    async _openSignature(context) {
        const stream = context.stream;
        if (stream) {
            let empty = true;
            let position = 0;
            while (position < stream.length) {
                const buffer = stream.read(Math.min(4096, stream.length - position));
                position += buffer.length;
                if (!buffer.every((value) => value === 0x00)) {
                    empty = false;
                    break;
                }
            }
            stream.seek(0);
            if (empty) {
                throw new view.Error('File has no content.');
            }
            /* eslint-disable no-control-regex */
            const entries = [
                { name: 'AES Crypt data', value: /^AES[\x01|\x02]\x00/ },
                { name: 'AppleDouble data', value: /^\x00\x05\x16\x07/ },
                { name: 'base64 data', value: /^gAAAAAB/ },
                { name: 'Bash script', value: /^(#!\/usr\/bin\/env|#!\/bin\/bash)\s/ },
                { name: 'BCNN model', value: /^BCNN/ },
                { name: 'BModel data', value: /^\xEE\xAA\x55\xFF/ }, // https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Builder/BM168x/bmodel.fbs
                { name: 'Cambricon model', value: /^\x7fMEF/ },
                { name: 'Cambricon model', value: /^cambricon_offline/ },
                { name: 'CviModel data', value: /^CviModel/ }, // https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Builder/CV18xx/proto/cvimodel.fbs
                { name: 'DRTcrypt data', value: /^DRTcrypt/ },
                { name: 'ELF executable', value: /^\x7FELF/ },
                { name: 'encrypted data', value: /^ENCRYPTED_FILE|EV_ENCRYPTED/ },
                { name: 'encrypted data', value: /^Salted__/ },
                { name: 'encrypted data', value: /^KINGSOFTOFFICE/ },
                { name: 'GGML data', value: /^lmgg|fmgg|tjgg|algg|fugg/ },
                { name: 'Git LFS header', value: /^\s*oid sha256:/ },
                { name: 'Git LFS header', value: /^version https:\/\/git-lfs.github.com/ },
                { name: 'HTML markup', value: /^\s*<!(doctype|DOCTYPE)\s*(html|HTML)>/ },
                { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*HTML\s+(PUBLIC|SYSTEM)?/ },
                { name: 'HTML markup', value: /^\s*<(html|HTML)(\s+[^>]+)?>/ },
                { name: 'Keras Tokenizer data', value: /^"{\\"class_name\\":\s*\\"Tokenizer\\"/ },
                { name: 'llama2.c checkpoint', value: /^..\x00\x00..\x00\x00..\x00\x00..\x00\x00..\x00\x00..\x00\x00..\x00\x00/, identifier: /^stories\d+[KM]\.bin/ },
                { name: 'Mathematica Notebook data', value: /^\(\*\sContent-type:\sapplication\/vnd\.wolfram\.mathematica\s\*\)/ },
                { name: 'Momentum Human Rig model', value: /^Momentum Model Definition/ }, // https://github.com/facebookresearch/MHR
                { name: 'obfuscated data', value: /^obfs/ },
                { name: 'Optimium model', value: /^EZMODEL/ }, // https://github.com/EZ-Optimium/Optimium,
                { name: 'PNG image', value: /^\x89PNG/ },
                { name: 'Python source code', value: /^((#.*(\n|\r\n))|('''.*'''(\n|\r\n))|("""[\s\S]*""")|(\n|\r\n))*(from[ ]+([a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*)[ ]+import[ ]+[a-zA-Z]\w*)/ },
                { name: 'Python source code', value: /^((#.*(\n|\r\n))|('''.*'''(\n|\r\n))|("""[\s\S]*""")|(\n|\r\n))*(import[ ]+[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*([ ]+as[ ]+[a-zA-Z]\w*)?[ ]*(,|;|\n|\r\n))/ },
                { name: 'Python virtual environment configuration', value: /^home[ ]*=[ ]*/, identifier: /^pyvenv\.cfg/ },
                { name: 'Rich Text Format data', value: /^{\\rtf/ },
                { name: 'SenseTime model', value: /^STEF/ },
                { name: 'SQLite data', value: /^SQLite format/ },
                { name: 'Terraform configuration', value: /^(\s*#[^\n]*\n)*(resource\s+"[a-z]|provider\s+"[a-z]|module\s+"[a-z]|variable\s+"[a-z]|data\s+"[a-z]|terraform\s*\{|locals\s*\{|output\s+"[a-z])/ },
                { name: 'TensorFlow Hub module', value: /^\x08\x03$/, identifier: /^tfhub_module\.pb/ },
                { name: 'Tokenizer data', value: /^IQ== 0\n/ },
                { name: 'TSD header', value: /^%TSD-Header-###%/ },
                { name: 'undocumented HALCON model', value: /^HDLMODEL/ },
                { name: 'undocumented license data', value: /^This model and the software may not be used or distributed in any manner except as authorized under a valid written agreemen/ },
                { name: 'undocumented NNC data', value: /^((\xC0|\xBC)\x0F\x00\x00ENNC|NNC3)/ },
                { name: 'undocumented RKNX data', value: /^RKNX\x00\x00\x00\x00/ },
                { name: 'V8 context snapshot', value: /^.\x00\x00\x00.\x00\x00\x00/, identifier: /^v8_context_snapshot\.bin/ },
                { name: 'V8 natives blob', value: /^./, identifier: /^natives_blob\.bin/ },
                { name: 'V8 snapshot', value: /^.\x00\x00\x00.\x00\x00\x00/, identifier: /^snapshot_blob\.bin/ },
                { name: 'ViSQOL model', value: /^svm_type\s/ },
                { name: 'VNN model', value: /^\x2F\x4E\x00\x00.\x00\x00\x00/, identifier: /.vnnmodel$/ },
                { name: 'Windows executable', value: /^MZ[\s\S]*PE\x00\x00/ },
            ];
            /* eslint-enable no-control-regex */
            const buffer = stream.peek(Math.min(4096, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            for (const entry of entries) {
                if (content.match(entry.value) && (!entry.identifier || context.identifier.match(entry.identifier))) {
                    throw new view.Error(`Invalid file content. File contains ${entry.name}.`);
                }
            }
        }
    }

    async import() {
        if (this._host.type === 'Browser' || this._host.type === 'Python') {
            const modules = ['./message', './onnx', './pytorch', './tflite', './mlnet', './onnx-proto', './onnx-schema', './tflite-schema'];
            const assets = ['onnx-metadata.json', 'pytorch-metadata.json', 'tflite-metadata.json'];
            await Promise.all([
                ...modules.map((module) => this._host.require(module).catch(() => {})),
                ...assets.map((asset) => this._host.asset(asset).catch(() => {})),
            ]);
        }
    }
};

view.Metadata = class {

    static async open(context, name) {
        view.Metadata._metadata = view.Metadata._metadata || new Map();
        const metadata = view.Metadata._metadata;
        if (!metadata.has(name)) {
            let data = null;
            try {
                data = await context.asset(name);
            } catch {
                // continue regardless of error
            }
            const types = JSON.parse(data);
            metadata.set(name, new view.Metadata(types));
        }
        return metadata.get(name);
    }

    constructor(types) {
        this._types = new Map();
        this._attributes = new Map();
        this._inputs = new Map();
        if (Array.isArray(types)) {
            for (const type of types) {
                if (this._types.has(type.name)) {
                    // throw new view.Error(`Duplicate type metadata '${type.name}'.`);
                }
                this._types.set(type.name, type);
                if (type.identifier !== undefined) {
                    this._types.set(type.identifier, type);
                }
            }
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name: name.toString() });
        }
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = `${type}:${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.attributes)) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(`${type}:${attribute.name}`, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }

    input(type, name) {
        const key = `${type}:${name}`;
        if (!this._inputs.has(key)) {
            this._inputs.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.inputs)) {
                for (const input of metadata.inputs) {
                    this._inputs.set(`${type}:${input.name}`, input);
                }
            }
        }
        return this._inputs.get(key);
    }
};

view.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading model.';
    }
};

if (typeof window !== 'undefined' && window.exports) {
    window.exports.view = view;
}

export const View = view.View;
export const ModelFactoryService = view.ModelFactoryService;
export const ModelSidebar = view.ModelSidebar;
export const NodeSidebar = view.NodeSidebar;
export const TensorSidebar = view.TensorSidebar;
export const Documentation = view.Documentation;
export const Formatter = view.Formatter;
export const Tensor = view.Tensor;
export const Quantization = view.Quantization;
