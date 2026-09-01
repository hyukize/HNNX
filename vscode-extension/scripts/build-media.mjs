import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';

const directory = path.dirname(fileURLToPath(import.meta.url));
const extension = path.resolve(directory, '..');
const root = path.resolve(extension, '..');
const media = path.join(extension, 'media');

const run = (command, args, cwd) => new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)));
});

await run(process.execPath, ['package.js', 'build', 'web'], root);
await fs.rm(media, { recursive: true, force: true });
await fs.cp(path.join(root, 'dist', 'web'), media, { recursive: true });

// The VS Code custom editor accepts ONNX files only. Preserve Netron's full
// format support in desktop and browser builds while keeping the Marketplace
// package limited to ONNX and its shared runtime dependencies.
const files = new Set([
    'aimet.js',
    'base.js',
    'browser.js',
    'dagre.js',
    'drop.js',
    'favicon.ico',
    'flatbuffers.js',
    'flexbuffers.js',
    'grapher.css',
    'grapher.js',
    'hdf5.js',
    'icon.png',
    'index.html',
    'json.js',
    'onnx-metadata.json',
    'onnx-proto.js',
    'onnx-schema.js',
    'onnx.js',
    'protobuf.js',
    'python.js',
    'tar.js',
    'text.js',
    'view.js',
    'vscode-entry.js',
    'vscode.js',
    'xml.js',
    'zip.js'
]);
const entries = await fs.readdir(media);
await Promise.all(entries
    .filter((entry) => !files.has(entry))
    .map((entry) => fs.rm(path.join(media, entry), { recursive: true, force: true })));
await fs.copyFile(
    path.join(root, 'source', 'onnx-graphsurgeon.py'),
    path.join(media, 'onnx-graphsurgeon.py')
);
