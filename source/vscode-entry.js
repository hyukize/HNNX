import { Host } from './vscode.js';
import { View } from './view.js';

window.addEventListener('load', async () => {
    try {
        const host = new Host();
        const view = new View(host);
        window.__view__ = view;
        await view.start();
    } catch (error) {
        const message = document.getElementById('message-text');
        if (message) {
            message.innerText = error.message || String(error);
        }
        document.body.setAttribute('class', 'welcome message');
    }
});
