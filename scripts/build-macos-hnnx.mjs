import { fileURLToPath } from 'url';
import path from 'path';
import { spawn } from 'child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const electronBuilder = path.join(root, 'node_modules', 'electron-builder', 'out', 'cli', 'cli.js');
const version = '0.1.17';
const app = path.join(root, 'dist', 'mac-arm64', 'HNNX.app');

const run = (command, args) => new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`${path.basename(command)} exited with ${code}`)));
});

const configuration = [
    '--arm64',
    '--publish',
    'never',
    '--config.mac.identity=null',
    '--config.mac.notarize=false',
    '--config.productName=HNNX',
    '--config.appId=ai.mrxrunway.hnnx',
    '--config.copyright=Copyright © 2026 Jonghyuk Park',
    '--config.extraMetadata.name=hnnx',
    '--config.extraMetadata.productName=HNNX',
    `--config.extraMetadata.version=${version}`,
    '--config.extraMetadata.appId=ai.mrxrunway.hnnx',
    '--config.extraMetadata.description=ONNX graph workbench for visualization, AIMET analysis, and graph editing',
    '--config.extraMetadata.author=Jonghyuk Park',
    '--config.extraMetadata.homepage=https://github.com/hyukize/HNNX',
    '--config.extraMetadata.aimet=true',
    '--config.extraMetadata.disableUpdates=true',
    '--config.asarUnpack=source/onnx-graphsurgeon.py',
    `--config.mac.artifactName=HNNX-${version}-arm64.\${ext}`,
    `--config.dmg.artifactName=HNNX-${version}-arm64.\${ext}`,
    `--config.dmg.title=HNNX ${version}`
];

await run(process.execPath, [electronBuilder, 'install-app-deps']);
await run(process.execPath, [electronBuilder, '--mac', 'dir', ...configuration]);

// An unsigned Electron executable contains only a linker-generated signature.
// Signing the complete bundle ad hoc seals its resources and prevents macOS
// from reporting the downloaded app as structurally damaged. This is not a
// substitute for an Apple Developer ID signature or notarization.
await run('codesign', ['--force', '--deep', '--sign', '-', '--timestamp=none', app]);
await run('codesign', ['--verify', '--deep', '--strict', '--verbose=2', app]);

await run(process.execPath, [
    electronBuilder,
    '--mac',
    'dmg',
    '--prepackaged',
    app,
    ...configuration
]);
