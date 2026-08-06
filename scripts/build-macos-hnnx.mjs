import { fileURLToPath } from 'url';
import path from 'path';
import { spawn } from 'child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';

const run = (args) => new Promise((resolve, reject) => {
    const child = spawn(npx, args, { cwd: root, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`npx exited with ${code}`)));
});

await run(['electron-builder', 'install-app-deps']);
await run([
    'electron-builder',
    '--mac',
    'dmg',
    '--arm64',
    '--publish',
    'never',
    '--config.mac.identity=null',
    '--config.mac.notarize=false',
    '--config.productName=HNNX',
    '--config.appId=ai.mrxrunway.hnnx',
    '--config.extraMetadata.name=hnnx',
    '--config.extraMetadata.productName=HNNX',
    '--config.extraMetadata.version=0.1.7',
    '--config.extraMetadata.appId=ai.mrxrunway.hnnx',
    '--config.extraMetadata.aimet=true',
    '--config.extraMetadata.disableUpdates=true',
    '--config.asarUnpack=source/onnx-graphsurgeon.py',
    '--config.mac.artifactName=HNNX-0.1.7-arm64.${ext}',
    '--config.dmg.artifactName=HNNX-0.1.7-arm64.${ext}',
    '--config.dmg.title=HNNX 0.1.7'
]);
