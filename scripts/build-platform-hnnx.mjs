import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import path from 'path';
import process from 'process';
import { spawn } from 'child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const platform = process.argv[2];
const metadata = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf-8'));
const { version } = metadata;
const electronBuilder = path.join(root, 'node_modules', 'electron-builder', 'out', 'cli', 'cli.js');

if (!['windows', 'linux'].includes(platform)) {
    throw new Error(`Unsupported HNNX desktop build platform '${platform || ''}'.`);
}

const run = (args) => new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [electronBuilder, ...args], { cwd: root, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`electron-builder exited with ${code}`)));
});

const common = [
    '--publish', 'never',
    '--config.productName=HNNX',
    '--config.appId=io.github.hyukize.hnnx',
    '--config.copyright=Copyright © 2026 Jonghyuk Park',
    '--config.extraMetadata.name=hnnx',
    '--config.extraMetadata.productName=HNNX',
    '--config.extraMetadata.desktopName=HNNX',
    `--config.extraMetadata.version=${version}`,
    '--config.extraMetadata.appId=io.github.hyukize.hnnx',
    '--config.extraMetadata.description=ONNX graph workbench for visualization, AIMET analysis, and graph editing',
    '--config.extraMetadata.author=Jonghyuk Park',
    '--config.extraMetadata.homepage=https://github.com/hyukize/HNNX',
    '--config.extraMetadata.aimet=true',
    '--config.extraMetadata.disableUpdates=true',
    '--config.asarUnpack=source/onnx-graphsurgeon.py'
];

await run(['install-app-deps']);
if (platform === 'windows') {
    await run([
        '--win', 'nsis', '--x64',
        ...common,
        '--config.win.icon=publish/icon.ico',
        '--config.win.azureSignOptions=',
        `--config.win.artifactName=HNNX-${version}-x64-setup.\${ext}`
    ]);
} else {
    await run([
        '--linux', 'AppImage', 'deb', '--x64',
        ...common,
        '--config.linux.icon=publish/icon.png',
        '--config.linux.executableName=hnnx',
        '--config.linux.category=Development',
        '--config.linux.maintainer=Jonghyuk Park <hyukize@users.noreply.github.com>',
        '--config.linux.syncDesktopName=true',
        `--config.linux.artifactName=HNNX-${version}-x64.\${ext}`
    ]);
}
