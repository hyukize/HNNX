
import * as child_process from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const execute = (command, args) => new Promise((resolve, reject) => {
    const child = child_process.spawn(command, args, {
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (data) => {
        stdout += data.toString();
    });
    child.stderr.on('data', (data) => {
        stderr += data.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
        if (code === 0) {
            resolve(stdout);
        } else {
            reject(new Error(stderr.trim() || `NVIDIA ONNX GraphSurgeon exited with code ${code}.`));
        }
    });
});

const normalizePython = (candidate) => candidate && candidate.startsWith('~/') ?
    path.join(os.homedir(), candidate.slice(2)) : candidate;

const pythonCandidates = (configured = '') => {
    const candidates = [];
    if (configured) {
        candidates.push(configured);
    }
    if (process.env.HNNX_PYTHON) {
        candidates.push(process.env.HNNX_PYTHON);
    }
    candidates.push(path.join(os.homedir(), '.hnnx', 'venv', 'bin', 'python3'));
    if (process.platform === 'darwin') {
        candidates.push('/opt/homebrew/bin/python3', '/usr/local/bin/python3');
    }
    candidates.push('python3', 'python');
    return Array.from(new Set(candidates.map((candidate) => normalizePython(candidate))));
};

export const validateOnnxGraphSurgeonPython = async (candidate) => {
    if (!candidate || typeof candidate !== 'string') {
        return false;
    }
    try {
        await execute(normalizePython(candidate), ['-c', 'import onnx, onnx_graphsurgeon']);
        return true;
    } catch {
        return false;
    }
};

export const findOnnxGraphSurgeonPython = async (configured = '') => {
    for (const candidate of pythonCandidates(configured)) {
        try {
            /* eslint-disable-next-line no-await-in-loop */
            const valid = await validateOnnxGraphSurgeonPython(candidate);
            if (!valid) {
                continue;
            }
            return candidate;
        } catch {
            // Try the next Python environment.
        }
    }
    const error = new Error(
        'A Python environment with NVIDIA ONNX GraphSurgeon was not found. ' +
        'Open GraphSurgeon Settings and select its Python interpreter.'
    );
    error.code = 'GRAPH_SURGEON_PYTHON_NOT_FOUND';
    throw error;
};

export const applyOnnxGraphSurgeonEdits = async (script, input, output, edits, configuredPython = '') => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hnnx-'));
    const editsPath = path.join(directory, 'edits.json');
    try {
        await fs.promises.writeFile(editsPath, JSON.stringify(edits), 'utf-8');
        const python = await findOnnxGraphSurgeonPython(configuredPython);
        await execute(python, [
            script,
            '--mode', 'edit',
            '--input', input,
            '--output', output,
            '--edits', editsPath
        ]);
    } finally {
        await fs.promises.rm(directory, { recursive: true, force: true });
    }
};

export const inferOnnxGraphSurgeonShapes = async (script, input, edits, configuredPython = '') => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hnnx-'));
    const editsPath = path.join(directory, 'edits.json');
    try {
        await fs.promises.writeFile(editsPath, JSON.stringify(edits), 'utf-8');
        const python = await findOnnxGraphSurgeonPython(configuredPython);
        const output = await execute(python, [
            script,
            '--mode', 'infer',
            '--input', input,
            '--edits', editsPath
        ]);
        return JSON.parse(output);
    } finally {
        await fs.promises.rm(directory, { recursive: true, force: true });
    }
};
