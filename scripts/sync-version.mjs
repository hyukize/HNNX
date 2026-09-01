import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import path from 'path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf-8'));
const writeJson = async (file, value) => fs.writeFile(file, `${JSON.stringify(value, null, 4)}\n`);

const packageFile = path.join(root, 'package.json');
const metadata = await readJson(packageFile);
const { version } = metadata;
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
    throw new Error(`Invalid HNNX version '${version}'.`);
}

const lockFile = path.join(root, 'package-lock.json');
const lock = await readJson(lockFile);
lock.name = metadata.name;
lock.version = version;
lock.packages[''].name = metadata.name;
lock.packages[''].version = version;
await writeJson(lockFile, lock);

const pythonFile = path.join(root, 'pyproject.toml');
const python = await fs.readFile(pythonFile, 'utf-8');
const synchronizedPython = python.replace(/^(version\s*=\s*")[^"]+("\s*)$/m, `$1${version}$2`);
if (synchronizedPython === python && !python.includes(`version = "${version}"`)) {
    throw new Error('Unable to locate the Python package version in pyproject.toml.');
}
await fs.writeFile(pythonFile, synchronizedPython);

const extensionFile = path.join(root, 'vscode-extension', 'package.json');
const extension = await readJson(extensionFile);
extension.version = version;
extension.scripts.package = `npm run build:media && npx --yes @vscode/vsce package --no-dependencies --out hnnx-${version}.vsix`;
await writeJson(extensionFile, extension);

process.stdout.write(`Synchronized HNNX ${version}.\n`);
