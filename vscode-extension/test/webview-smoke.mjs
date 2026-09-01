const targets = await fetch('http://127.0.0.1:9333/json/list').then((response) => response.json());
const target = targets.find((item) => item.type === 'iframe' && item.url.startsWith('vscode-webview:'));
if (!target) {
    throw new Error('VS Code did not expose a Webview target.');
}

const socket = new WebSocket(target.webSocketDebuggerUrl);
let sequence = 0;
const pending = new Map();
const contexts = new Map();

socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.executionContextCreated') {
        const context = message.params.context;
        if (context.auxData && context.auxData.isDefault) {
            contexts.set(context.auxData.frameId, context.id);
        }
    }
    if (message.id && pending.has(message.id)) {
        pending.get(message.id)(message);
        pending.delete(message.id);
    }
};

await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
});

const call = (method, params = {}) => new Promise((resolve) => {
    const id = ++sequence;
    pending.set(id, resolve);
    socket.send(JSON.stringify({ id, method, params }));
});

await call('Runtime.enable');
const deadline = Date.now() + 30000;
let result = null;
while (Date.now() < deadline && !result) {
    for (const contextId of contexts.values()) {
        const response = await call('Runtime.evaluate', {
            contextId,
            expression: `JSON.stringify({
                body: document.body && document.body.className,
                nodes: document.querySelectorAll('.node').length,
                badges: Array.from(document.querySelectorAll('.node-item-quantization'))
                    .filter((element) => element.textContent.includes('A8→A16')).length,
                setup: document.querySelector('#setup-overlay')?.classList.contains('visible') || false,
                text: document.querySelector('#message-text') && document.querySelector('#message-text').innerText
            })`,
            returnByValue: true
        });
        const value = response.result && response.result.result && response.result.result.value;
        if (value) {
            const candidate = JSON.parse(value);
            if (candidate.body && candidate.body.includes('default') && candidate.nodes > 0 && candidate.badges > 0) {
                result = candidate;
                break;
            }
        }
    }
    if (!result) {
        await new Promise((resolve) => setTimeout(resolve, 250));
    }
}
socket.close();

if (!result) {
    throw new Error('HNNX Webview did not render the ONNX graph with AIMET badges.');
}
process.stdout.write(`${JSON.stringify(result)}\n`);
