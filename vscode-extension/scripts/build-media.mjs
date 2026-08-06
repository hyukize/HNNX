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
await fs.copyFile(
    path.join(root, 'source', 'onnx-graphsurgeon.py'),
    path.join(media, 'onnx-graphsurgeon.py')
);
