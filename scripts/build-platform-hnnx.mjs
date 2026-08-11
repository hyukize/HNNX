import { fileURLToPath } from 'url';
import path from 'path';
import process from 'process';
import { spawn } from 'child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const platform = process.argv[2];
const version = '0.1.18';
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
    '--config.appId=ai.mrxrunway.hnnx',
    '--config.copyright=Copyright © 2026 Jonghyuk Park',
    '--config.extraMetadata.name=hnnx',
    '--config.extraMetadata.productName=HNNX',
    '--config.extraMetadata.desktopName=HNNX',
    `--config.extraMetadata.version=${version}`,
    '--config.extraMetadata.appId=ai.mrxrunway.hnnx',
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
