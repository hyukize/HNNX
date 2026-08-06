import * as playwright from '@playwright/test';
import { Buffer } from 'node:buffer';

playwright.test.setTimeout(120000);

const varint = (value) => {
    const bytes = [];
    do {
        let byte = value % 128;
        value = Math.floor(value / 128);
        if (value > 0) {
            byte |= 0x80;
        }
        bytes.push(byte);
    } while (value > 0);
    return Buffer.from(bytes);
};

const join = (parts) => Buffer.concat(parts);
const scalar = (field, value) => join([varint(field << 3), varint(value)]);
const bytes = (field, value) => {
    value = Buffer.isBuffer(value) ? value : Buffer.from(value);
    return join([varint((field << 3) | 2), varint(value.length), value]);
};
const string = (field, value) => bytes(field, Buffer.from(value, 'utf-8'));
const node = (name, type, inputs, outputs) => join([
    ...inputs.map((value) => string(1, value)),
    ...outputs.map((value) => string(2, value)),
    string(3, name),
    string(4, type)
]);
const valueInfo = (name) => string(1, name);

const editableModel = () => {
    const nodes = [
        node('source', 'Relu', ['x'], ['a']),
        node('cast', 'Cast', ['a'], ['b']),
        node('add', 'Add', ['b', 'x'], ['y'])
    ];
    const graph = join([
        ...nodes.map((value) => bytes(1, value)),
        string(2, 'stage3-adversarial'),
        bytes(11, valueInfo('x')),
        bytes(11, valueInfo('alt')),
        bytes(12, valueInfo('y'))
    ]);
    return join([
        scalar(1, 8),
        bytes(7, graph),
        bytes(8, scalar(2, 17))
    ]);
};

const seeded = (seed) => {
    let value = seed >>> 0;
    return () => {
        value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
        return value / 0x100000000;
    };
};

const historyProgram = (seed, maximum) => {
    const random = seeded(seed);
    const operations = [];
    let depth = maximum;
    for (let index = 0; index < 12; index++) {
        let operation = null;
        if (depth === 0) {
            operation = 'redo';
        } else if (depth === maximum) {
            operation = 'undo';
        } else {
            operation = random() < 0.5 ? 'undo' : 'redo';
        }
        depth += operation === 'undo' ? -1 : 1;
        operations.push(operation);
    }
    return operations;
};

const cases = Array.from({ length: 100 }, (_, index) => {
    const number = index + 1;
    const family = index % 5;
    const maximum = family === 0 ? 2 : 3;
    const random = seeded(number * 3571);
    const neutrals = Array.from({ length: 5 }, () => Math.floor(random() * 5));
    return { number, family, maximum, operations: historyProgram(number * 7919, maximum), neutrals };
});

const fingerprints = new Set(cases.map((testCase) => JSON.stringify([
    testCase.family, testCase.operations, testCase.neutrals
])));
if (fingerprints.size !== 100) {
    throw new Error('Stage 3 requires 100 structurally unique adversarial UI workflows.');
}

const openEditor = async (page, number) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    await playwright.expect(page.locator('#open-file-button')).toBeVisible({ timeout: 60000 });
    await page.locator('#open-file-dialog').setInputFiles({
        name: `stage3-adversarial-${number}.onnx`,
        mimeType: 'application/octet-stream',
        buffer: editableModel()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
};

const addFamilyEdit = async (page, testCase) => {
    const add = page.locator('#graph-edit-add-button');
    const overlay = page.locator('#graph-edit-add-overlay');
    const search = page.locator('#graph-edit-add-search');
    if (testCase.family === 1) {
        await page.locator('#input-name-alt:visible').click();
        await page.keyboard.press('d');
        await playwright.expect(page.locator('#input-name-alt:visible')).toHaveCount(0);
        return 'delete-alt';
    }
    if (testCase.family === 2) {
        await page.locator('.graph-item-output:visible').click();
        await page.keyboard.press('d');
        await playwright.expect(page.locator('.graph-item-output:visible')).toHaveCount(0);
        return 'delete-output';
    }
    if (testCase.family === 3) {
        await add.click();
        await search.fill('Relu');
        await page.locator('.graph-edit-add-item:visible', { hasText: /^Relu/ }).click();
        await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill(`stage3_relu_${testCase.number}`);
        await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill(`stage3_value_${testCase.number}`);
        await page.locator('#graph-edit-add-form input[aria-label="X"]').fill('x');
        await page.locator('#graph-edit-add-form button', { hasText: 'ADD RELU' }).click();
        await playwright.expect(overlay).not.toBeVisible();
        return 'add-relu';
    }
    if (testCase.family === 4) {
        await add.click();
        await search.fill('Graph Input');
        await page.locator('.graph-edit-add-item:visible', { hasText: /^Graph Input/ }).click();
        await page.locator('#graph-edit-add-form input[aria-label="Input name"]').fill(`stage3_input_${testCase.number}`);
        await page.locator('#graph-edit-add-form input[aria-label="Shape"]').fill('2, 3');
        await page.locator('#graph-edit-add-form button', { hasText: 'ADD GRAPH INPUT' }).click();
        await playwright.expect(overlay).not.toBeVisible();
        return 'add-input';
    }
    return null;
};

const deleteCastAndReconnect = async (page) => {
    await page.getByText('Cast', { exact: true }).first().click();
    await page.keyboard.press('d');
    await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(0);
    await page.locator('.graph-edit-output-port[aria-label="Use output a"]').click();
    await page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]').click();
    await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(1);
};

const expected = async (page, testCase, depth) => {
    const familyActive = testCase.family !== 0 && depth >= 1;
    const castDeleted = depth >= testCase.maximum - 1;
    const reconnected = depth >= testCase.maximum;
    await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(castDeleted ? 0 : 1);
    await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(reconnected || !castDeleted ? 1 : 0);
    await playwright.expect(page.locator('#edge-b')).toHaveCount(castDeleted ? 0 : 1);
    if (testCase.family === 1) {
        await playwright.expect(page.locator('#input-name-alt:visible')).toHaveCount(familyActive ? 0 : 1);
    } else if (testCase.family === 2) {
        await playwright.expect(page.locator('.graph-item-output:visible')).toHaveCount(familyActive ? 0 : 1);
    } else if (testCase.family === 3) {
        await playwright.expect(page.getByText(`stage3_relu_${testCase.number}`, { exact: true })).toHaveCount(familyActive ? 1 : 0);
    } else if (testCase.family === 4) {
        await playwright.expect(page.locator(`#input-name-stage3_input_${testCase.number}:visible`)).toHaveCount(familyActive ? 1 : 0);
    }
};

const burstHistory = async (page, operations) => {
    const session = await page.context().newCDPSession(page);
    /* eslint-disable no-await-in-loop -- CDP key events must preserve user input order. */
    for (const operation of operations) {
        const modifiers = operation === 'redo' ? 12 : 4;
        await session.send('Input.dispatchKeyEvent', {
            type: 'rawKeyDown', key: 'z', code: 'KeyZ', modifiers,
            windowsVirtualKeyCode: 90, nativeVirtualKeyCode: 90
        });
        await session.send('Input.dispatchKeyEvent', {
            type: 'keyUp', key: 'z', code: 'KeyZ', modifiers: 0,
            windowsVirtualKeyCode: 90, nativeVirtualKeyCode: 90
        });
    }
    /* eslint-enable no-await-in-loop */
    await session.detach();
};

const historySettled = async (page, depth, maximum) => {
    if (depth > 0) {
        await playwright.expect(page.locator('#graph-edit-undo-button')).toBeEnabled();
    }
    if (depth < maximum) {
        await playwright.expect(page.locator('#graph-edit-redo-button')).toBeEnabled();
    }
};

const neutral = async (page, testCase, round) => {
    const action = testCase.neutrals[round % testCase.neutrals.length];
    if (action === 0) {
        await page.locator('#graph-edit-layout-button').click();
        await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    } else if (action === 1) {
        await page.locator('#graph-edit-button').click();
        await page.locator('#graph-edit-button').click();
        await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    } else if (action === 2) {
        await page.locator('.graph-edit-output-port[aria-label="Use output a"]').click();
        await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
        await page.keyboard.press('q');
        await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);
    } else if (action === 3) {
        await page.locator('#graph-edit-infer-button').click();
        const status = page.locator('#graph-edit-status');
        if (await status.getAttribute('class').then((value) => value.includes('invalid'))) {
            await playwright.expect(status).toContainText(/SHAPE INFERENCE (BLOCKED|FAILED)/);
        } else {
            await playwright.expect(page.locator('#graph-edit-node-menu')).toBeVisible();
        }
        if (await page.locator('#graph-edit-node-menu').isVisible()) {
            await page.locator('#target').focus();
            await page.keyboard.press('q');
            await playwright.expect(page.locator('#graph-edit-node-menu')).not.toBeVisible();
        }
    } else {
        await page.locator('#graph-edit-add-button').click();
        const search = page.locator('#graph-edit-add-search');
        await search.fill('dqX');
        await search.press('Backspace');
        await search.press('q');
        await playwright.expect(search).toHaveValue('dqq');
        await page.locator('#target').focus();
        await page.keyboard.press('q');
        await playwright.expect(page.locator('#graph-edit-add-overlay')).not.toBeVisible();
    }
};

for (const testCase of cases) {
    playwright.test(`stage3 adversarial workflow ${String(testCase.number).padStart(3, '0')} family-${testCase.family}`, async ({ page }) => {
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await openEditor(page, testCase.number);
        await addFamilyEdit(page, testCase);
        await deleteCastAndReconnect(page);
        let depth = testCase.maximum;
        await expected(page, testCase, depth);

        let cursor = 0;
        let round = 0;
        /* eslint-disable no-await-in-loop -- each batch validates the state produced by the previous batch. */
        while (cursor < testCase.operations.length) {
            const size = 3 + ((testCase.number + round) % 4);
            const batch = testCase.operations.slice(cursor, cursor + size);
            for (const operation of batch) {
                depth += operation === 'undo' ? -1 : 1;
            }
            await burstHistory(page, batch);
            await historySettled(page, depth, testCase.maximum);
            await expected(page, testCase, depth);
            await neutral(page, testCase, round);
            await expected(page, testCase, depth);
            cursor += batch.length;
            round++;
        }
        /* eslint-enable no-await-in-loop */

        await burstHistory(page, Array(testCase.maximum + 2).fill('undo'));
        await historySettled(page, 0, testCase.maximum);
        await expected(page, testCase, 0);
        await burstHistory(page, Array(testCase.maximum + 2).fill('redo'));
        await historySettled(page, testCase.maximum, testCase.maximum);
        await expected(page, testCase, testCase.maximum);
        await page.locator('#graph-edit-reset-button').click();
        await expected(page, testCase, 0);
        await playwright.expect(page.locator('#graph-edit-undo-button')).toBeDisabled();
        await playwright.expect(page.locator('#graph-edit-redo-button')).toBeDisabled();
        playwright.expect(errors).toEqual([]);
    });
}
