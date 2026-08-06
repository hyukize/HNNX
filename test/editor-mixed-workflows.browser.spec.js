import * as playwright from '@playwright/test';
import { Buffer } from 'node:buffer';

playwright.test.setTimeout(30000);
playwright.test.describe.configure({ mode: 'serial' });

const protobufVarint = (value) => {
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

const protobufJoin = (parts) => Buffer.concat(parts);
const protobufValue = (field, value) => protobufJoin([
    protobufVarint(field << 3), protobufVarint(value)
]);
const protobufBytes = (field, value) => {
    value = Buffer.isBuffer(value) ? value : Buffer.from(value);
    return protobufJoin([
        protobufVarint((field << 3) | 2), protobufVarint(value.length), value
    ]);
};
const protobufString = (field, value) => protobufBytes(field, Buffer.from(value, 'utf-8'));
const protobufNode = (name, type, inputs, outputs) => protobufJoin([
    ...inputs.map((value) => protobufString(1, value)),
    ...outputs.map((value) => protobufString(2, value)),
    protobufString(3, name),
    protobufString(4, type)
]);
const protobufValueInfo = (name) => protobufString(1, name);

const editableModel = () => {
    const nodes = [
        protobufNode('source', 'Relu', ['x'], ['a']),
        protobufNode('cast', 'Cast', ['a'], ['b']),
        protobufNode('add', 'Add', ['b', 'x'], ['y'])
    ];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'mixed-editor-workflow'),
        protobufBytes(11, protobufValueInfo('x')),
        protobufBytes(12, protobufValueInfo('y'))
    ]);
    return protobufJoin([
        protobufValue(1, 8),
        protobufBytes(7, graph),
        protobufBytes(8, protobufValue(2, 17))
    ]);
};

const physicalKey = async (page, key, code, virtualKeyCode, modifiers = 0) => {
    const session = await page.context().newCDPSession(page);
    await session.send('Input.dispatchKeyEvent', {
        type: 'rawKeyDown', key, code, modifiers,
        windowsVirtualKeyCode: virtualKeyCode,
        nativeVirtualKeyCode: virtualKeyCode
    });
    await session.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key, code, modifiers: 0,
        windowsVirtualKeyCode: virtualKeyCode,
        nativeVirtualKeyCode: virtualKeyCode
    });
    await session.detach();
};

const permutations = (values) => {
    if (values.length === 0) {
        return [[]];
    }
    return values.flatMap((value, index) => permutations([
        ...values.slice(0, index), ...values.slice(index + 1)
    ]).map((rest) => [value, ...rest]));
};

const neutralOrders = permutations(['infer', 'layout', 'view', 'cancel', 'drag']).slice(0, 100);
if (neutralOrders.length !== 100 || new Set(neutralOrders.map((order) => order.join('|'))).size !== 100) {
    throw new Error('Expected 100 structurally unique mixed editor workflows.');
}

const openEditor = async (page, index) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: `mixed-workflow-${index}.onnx`,
        mimeType: 'application/octet-stream',
        buffer: editableModel()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
};

const deleteCast = async (page, mode) => {
    const cast = page.getByText('Cast', { exact: true }).first();
    await cast.click();
    if (mode === 0) {
        await physicalKey(page, 'ㅇ', 'KeyD', 68);
    } else if (mode === 1) {
        await page.keyboard.press('d');
    } else {
        await page.locator('#graph-edit-node-menu .danger').click();
    }
    await playwright.expect(cast).toHaveCount(0);
};

const connectSourceToAdd = async (page, drag) => {
    const source = page.locator('.graph-edit-output-port[aria-label="Use output a"]');
    const target = page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]');
    if (drag) {
        const sourceBounds = await source.boundingBox();
        const targetBounds = await target.boundingBox();
        await page.mouse.move(
            sourceBounds.x + sourceBounds.width / 2,
            sourceBounds.y + sourceBounds.height / 2
        );
        await page.mouse.down();
        await page.mouse.move(
            targetBounds.x + targetBounds.width / 2,
            targetBounds.y + targetBounds.height / 2,
            { steps: 3 }
        );
        await page.mouse.up();
    } else {
        await source.click();
        await target.click();
    }
    await playwright.expect(page.locator('.edge-path[id^="edge-a-"]')).toHaveCount(1);
};

const history = async (page, redo, mode) => {
    if (mode === 0) {
        await page.locator(redo ? '#graph-edit-redo-button' : '#graph-edit-undo-button').click();
    } else if (mode === 1) {
        await page.keyboard.press(redo ? 'Meta+Shift+z' : 'Meta+z');
    } else {
        await physicalKey(page, 'ㅋ', 'KeyZ', 90, redo ? 12 : 4);
    }
};

const neutralAction = async (page, action) => {
    if (action === 'infer') {
        await page.locator('#graph-edit-infer-button').click();
        await playwright.expect(page.locator('#graph-edit-node-menu')).toBeVisible();
        await playwright.expect(page.locator('#graph-edit-node-menu')).toContainText('Shape inference failed');
        await physicalKey(page, 'ㅂ', 'KeyQ', 81);
        await playwright.expect(page.locator('#graph-edit-node-menu')).not.toBeVisible();
    } else if (action === 'layout') {
        await page.locator('#graph-edit-layout-button').click();
        await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    } else if (action === 'view') {
        await page.locator('#graph-edit-button').click();
        await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit/);
        await page.locator('#graph-edit-button').click();
        await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    } else if (action === 'cancel') {
        await page.locator('.graph-edit-output-port[aria-label="Use output a"]').click();
        await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
        await page.keyboard.press('q');
        await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);
    } else if (action === 'drag') {
        const node = page.locator('.graph-node').filter({ hasText: 'Relu' }).first();
        const bounds = await node.boundingBox();
        await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
        await page.mouse.down();
        await page.mouse.move(
            bounds.x + bounds.width / 2 + 12,
            bounds.y + bounds.height / 2 + 8,
            { steps: 2 }
        );
        await page.mouse.up();
        await playwright.expect(page.locator('#graph-edit-status')).toContainText('Moved');
    }
};

for (let index = 0; index < neutralOrders.length; index++) {
    const order = neutralOrders[index];
    const name = order.map((action) => action[0]).join('');
    playwright.test(`mixed editor workflow ${String(index + 1).padStart(3, '0')} ${name}`, async ({ page }) => {
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await openEditor(page, index + 1);
        await deleteCast(page, index % 3);
        await connectSourceToAdd(page, index % 2 === 1);

        for (const action of order) {
            // The order itself is the test dimension, so these actions must
            // execute serially rather than as concurrent promises.
            // eslint-disable-next-line no-await-in-loop
            await neutralAction(page, action);
        }

        const undoMode = index % 3;
        const redoMode = (index + 1) % 3;
        await history(page, false, undoMode);
        await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(0);
        await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(0);
        await history(page, true, redoMode);
        await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(1);
        await history(page, false, (index + 2) % 3);
        await history(page, false, undoMode);
        await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(1);
        await playwright.expect(page.locator('#edge-b')).toHaveCount(1);

        await history(page, true, redoMode);
        await history(page, true, (index + 2) % 3);
        await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(0);
        await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(1);
        await page.locator('#graph-edit-reset-button').click();
        await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(1);
        await playwright.expect(page.locator('#edge-b')).toHaveCount(1);
        await playwright.expect(page.locator('#graph-edit-undo-button')).toBeDisabled();
        playwright.expect(errors).toEqual([]);
    });
}
