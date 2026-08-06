import * as base from './base.js';
import * as browser from './browser.js';

const vscode = {};

vscode.Host = class extends browser.Host {

    constructor() {
        super();
        window.document.documentElement.classList.add('vscode-webview');
        this._api = window.acquireVsCodeApi();
        this._requests = new Map();
        this._requestId = 0;
        this._environment.name = 'HNNX';
        this._environment.type = 'VS Code';
        this._environment.packaged = true;
        this._environment.menu = true;
        this._environment.serial = true;
        window.addEventListener('message', (event) => this._message(event.data));
    }

    async view(view) {
        this._view = view;
    }

    async start() {
        this._view.show('welcome spinner');
        this._api.postMessage({ type: 'ready' });
    }

    async execute(name, value) {
        if (name === 'open') {
            this._api.postMessage({ type: 'pickEncodings' });
            return undefined;
        }
        if (name === 'save-onnx-as') {
            this._api.postMessage({ type: 'saveOnnxAs', edits: value.edits });
            return undefined;
        }
        if (name === 'infer-onnx-shapes') {
            this._api.postMessage({ type: 'inferOnnxShapes', edits: value.edits });
            return undefined;
        }
        if (name === 'set-theme') {
            this._api.postMessage({ type: 'setTheme', value });
            return undefined;
        }
        return super.execute(name, value);
    }

    async fetchWorkspace(file, encoding) {
        const id = ++this._requestId;
        const promise = new Promise((resolve, reject) => {
            this._requests.set(id, { resolve, reject, encoding });
        });
        this._api.postMessage({ type: 'fetch', id, file });
        return promise;
    }

    async fetch(file, encoding, base) {
        const target = base ? `${base}/${file}` : file;
        return this.fetchWorkspace(target, encoding);
    }

    openURL(url) {
        this._api.postMessage({ type: 'openExternal', url });
    }

    event() {
    }

    exception(error, fatal) {
        if (error) {
            this._api.postMessage({
                type: 'log',
                fatal: Boolean(fatal),
                message: error.message || String(error),
                stack: error.stack || ''
            });
        }
    }

    async _message(message) {
        switch (message && message.type) {
            case 'open': {
                const files = [];
                if (message.model) {
                    files.push(new File([this._bytes(message.model.data)], message.model.name));
                }
                if (message.encodings) {
                    files.push(new File([this._bytes(message.encodings.data)], message.encodings.name));
                }
                await this._openFiles(files);
                break;
            }
            case 'attach': {
                if (message.encodings) {
                    const file = new File([this._bytes(message.encodings.data)], message.encodings.name);
                    await this._openFiles([file]);
                }
                break;
            }
            case 'fetchResult': {
                const request = this._requests.get(message.id);
                if (request) {
                    this._requests.delete(message.id);
                    if (message.error) {
                        request.reject(new Error(message.error));
                    } else if (request.encoding === 'utf-8') {
                        request.resolve(new TextDecoder('utf-8').decode(message.data));
                    } else {
                        request.resolve(new base.BinaryStream(this._bytes(message.data)));
                    }
                }
                break;
            }
            case 'graphEditSaveResult': {
                this._view.graphEditSaveResult(message);
                break;
            }
            case 'graphEditShapeInferenceResult': {
                await this._view.graphEditShapeInferenceResult(message);
                break;
            }
            case 'theme': {
                this._view.setTheme(message.preference, message.effective);
                break;
            }
            default:
                break;
        }
    }

    _bytes(value) {
        if (value instanceof Uint8Array) {
            return value;
        }
        if (value instanceof ArrayBuffer) {
            return new Uint8Array(value);
        }
        if (ArrayBuffer.isView(value)) {
            return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
        }
        if (value && value.type === 'Buffer' && Array.isArray(value.data)) {
            return Uint8Array.from(value.data);
        }
        if (Array.isArray(value)) {
            return Uint8Array.from(value);
        }
        if (value && typeof value === 'object') {
            return Uint8Array.from(Object.keys(value).sort((a, b) => Number(a) - Number(b)).map((key) => value[key]));
        }
        throw new Error('VS Code did not provide binary file data.');
    }
};

export const Host = vscode.Host;
