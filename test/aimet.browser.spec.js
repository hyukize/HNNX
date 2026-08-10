import * as path from 'path';
import * as playwright from '@playwright/test';
import * as url from 'url';
import { Buffer } from 'node:buffer';
import process from 'node:process';

playwright.test.setTimeout(30000);

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
const protobufValue = (field, value) => protobufJoin([protobufVarint(field << 3), protobufVarint(value)]);
const protobufBytes = (field, value) => {
    value = Buffer.isBuffer(value) ? value : Buffer.from(value);
    return protobufJoin([protobufVarint((field << 3) | 2), protobufVarint(value.length), value]);
};
const protobufString = (field, value) => protobufBytes(field, Buffer.from(value, 'utf-8'));
const protobufNode = (name, type, inputs, outputs) => protobufJoin([
    ...inputs.map((value) => protobufString(1, value)),
    ...outputs.map((value) => protobufString(2, value)),
    protobufString(3, name),
    protobufString(4, type)
]);
const protobufValueInfo = (name) => protobufString(1, name);
const protobufTensorValueInfo = (name, dataType, dimensions) => {
    const shape = protobufJoin(dimensions.map((dimension) =>
        protobufBytes(1, protobufValue(1, dimension))));
    const tensor = protobufJoin([protobufValue(1, dataType), protobufBytes(2, shape)]);
    return protobufJoin([protobufString(1, name), protobufBytes(2, protobufBytes(1, tensor))]);
};
const svgNumbers = (value) => (value.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) || []).map(Number);
const physicalKey = async (page, key, code, virtualKeyCode, modifiers = 0) => {
    const session = await page.context().newCDPSession(page);
    await session.send('Input.dispatchKeyEvent', {
        type: 'rawKeyDown',
        key,
        code,
        modifiers,
        windowsVirtualKeyCode: virtualKeyCode,
        nativeVirtualKeyCode: virtualKeyCode
    });
    await session.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key,
        code,
        modifiers: 0,
        windowsVirtualKeyCode: virtualKeyCode,
        nativeVirtualKeyCode: virtualKeyCode
    });
    await session.detach();
};

const primaryModifier = () => {
    const isMac = process.platform === 'darwin';
    return {
        key: isMac ? 'Meta' : 'Control',
        mask: isMac ? 4 : 2
    };
};

const editableOnnx = () => {
    const nodes = [
        protobufNode('abs', 'Abs', ['x'], ['a']),
        protobufNode('neg', 'Neg', ['alt'], ['b']),
        protobufNode('add', 'Add', ['a', 'b'], ['y']),
        protobufNode('clip', 'Clip', ['x', '', 'optional_value'], ['clipped'])
    ];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'editable'),
        protobufBytes(11, protobufValueInfo('x')),
        protobufBytes(11, protobufValueInfo('alt')),
        protobufBytes(11, protobufValueInfo('optional_value')),
        protobufBytes(12, protobufValueInfo('y'))
    ]);
    const opset = protobufValue(2, 13);
    return protobufJoin([protobufValue(1, 8), protobufBytes(7, graph), protobufBytes(8, opset)]);
};

const routingOnnx = () => {
    const nodes = [
        protobufNode('split', 'Split', ['x'], ['left', 'right']),
        protobufNode('concat', 'Concat', ['left', 'right'], ['y'])
    ];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'routing'),
        protobufBytes(11, protobufValueInfo('x')),
        protobufBytes(12, protobufValueInfo('y'))
    ]);
    const opset = protobufValue(2, 13);
    return protobufJoin([protobufValue(1, 8), protobufBytes(7, graph), protobufBytes(8, opset)]);
};

const fanoutRoutingOnnx = () => {
    const nodes = [
        protobufNode('cast', 'Cast', ['x'], ['a']),
        protobufNode('abs', 'Abs', ['a'], ['left']),
        protobufNode('neg', 'Neg', ['a'], ['right']),
        protobufNode('add', 'Add', ['left', 'right'], ['y'])
    ];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'fanout-routing'),
        protobufBytes(11, protobufValueInfo('x')),
        protobufBytes(12, protobufValueInfo('y'))
    ]);
    return protobufJoin([protobufValue(1, 8), protobufBytes(7, graph), protobufBytes(8, protobufValue(2, 17))]);
};

const bundledRoutingOnnx = () => {
    const tensors = Array.from({ length: 8 }, (_, index) => `part_${index}`);
    const repeated = tensors.flatMap((name) => Array(4).fill(name));
    const nodes = [
        protobufNode('split_8', 'Split', ['x'], tensors),
        protobufNode('concat_8', 'Concat', repeated, ['y'])
    ];
    const shape = [1, 64, 1, 2048];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'bundled-routing'),
        protobufBytes(11, protobufTensorValueInfo('x', 1, [1, 512, 1, 2048])),
        protobufBytes(12, protobufTensorValueInfo('y', 1, [1, 512, 1, 2048])),
        ...tensors.map((name) => protobufBytes(13, protobufTensorValueInfo(name, 1, shape)))
    ]);
    return protobufJoin([protobufValue(1, 8), protobufBytes(7, graph), protobufBytes(8, protobufValue(2, 17))]);
};

const bundledRoutingEncodings = () => Buffer.from(JSON.stringify({
    version: '2.0.0',
    activation_encodings: Array.from({ length: 8 }, (_, index) => ({
        name: `part_${index}`,
        output_dtype: 'uint8',
        y_scale: 0.125,
        y_zero_point: 12
    })),
    param_encodings: []
}));

// Regression for the incremental delete path: deleting a pass-through node
// must not leave detached SVG edges behind when its consumer is reconnected.
const castDeletionOnnx = () => {
    const nodes = [
        protobufNode('source', 'Relu', ['x'], ['a']),
        protobufNode('cast', 'Cast', ['a'], ['b']),
        protobufNode('add', 'Add', ['b', 'x'], ['y'])
    ];
    const graph = protobufJoin([
        ...nodes.map((node) => protobufBytes(1, node)),
        protobufString(2, 'cast-delete'),
        protobufBytes(11, protobufValueInfo('x')),
        protobufBytes(12, protobufValueInfo('y'))
    ]);
    return protobufJoin([protobufValue(1, 8), protobufBytes(7, graph), protobufBytes(8, protobufValue(2, 13))]);
};

playwright.test('AIMET encodings attachment', async ({ page }) => {
    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    const model = path.resolve(dir, 'aimet.onnx');
    const data = path.resolve(dir, 'aimet.onnx.data');
    const encodings = path.resolve(dir, 'aimet.encodings');

    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles([model, data, encodings]);
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.waitForSelector('.node-item-quantization', { timeout: 10000 });
    await playwright.expect(page.locator('html')).toHaveClass(/has-encodings/);
    await playwright.expect(page.locator('#encodings-toggle-button')).toBeVisible();

    const attachmentSession = await page.context().newCDPSession(page);
    const attachmentStability = await attachmentSession.send('Runtime.evaluate', {
        expression: `new Promise((resolve) => {
            const snapshot = () => JSON.stringify(Array.from(document.querySelectorAll('.graph-node'))
                .map((node) => node.getAttribute('transform')));
            const before = snapshot();
            requestAnimationFrame(() => requestAnimationFrame(() => resolve(before === snapshot())));
        })`,
        awaitPromise: true,
        returnByValue: true
    });
    await attachmentSession.detach();
    playwright.expect(attachmentStability.result.value).toBe(true);

    const canvas = page.locator('#canvas');
    const zoomBefore = await canvas.boundingBox();
    await page.mouse.move(300, 200);
    const zoomModifier = primaryModifier();
    await page.keyboard.down(zoomModifier.key);
    await page.mouse.wheel(0, -1);
    await page.keyboard.up(zoomModifier.key);
    await page.waitForTimeout(50);
    const zoomAfter = await canvas.boundingBox();
    const zoomRatio = zoomAfter.width / zoomBefore.width;
    playwright.expect(zoomRatio).toBeGreaterThan(1);

    const inputBadge = page.locator('#input-name-x .node-item-quantization');
    await playwright.expect(inputBadge).toContainText('Q:A8');
    await inputBadge.click();
    const inputSidebar = page.locator('#sidebar-content');
    await playwright.expect(inputSidebar).toContainText('Graph Input QParam');
    await playwright.expect(inputSidebar).toContainText('A8');

    const badge = page.locator('.node-item-quantization', { hasText: 'Q:A8→A16' });
    await playwright.expect(badge).toBeVisible();
    await badge.click();
    const nodeSidebar = page.locator('#sidebar-content');
    await playwright.expect(nodeSidebar).toContainText('Input Tensor Precision');
    await playwright.expect(nodeSidebar).toContainText('Output QParams');
    await playwright.expect(nodeSidebar).toContainText('explicit tensor encoding');
    await playwright.expect(nodeSidebar).not.toContainText('0.125');
    await playwright.expect(nodeSidebar).toContainText('0.0078125');
    const sidebar = page.locator('#sidebar');
    const resize = page.locator('#sidebar-resize');
    await page.waitForTimeout(150);
    const before = await sidebar.boundingBox();
    const handle = await resize.boundingBox();
    await page.mouse.move(handle.x + handle.width / 2, handle.y + 100);
    await page.mouse.down();
    await page.mouse.move(handle.x - 120, handle.y + 100);
    await page.mouse.up();
    const after = await sidebar.boundingBox();
    playwright.expect(after.width).toBeGreaterThan(before.width + 100);

    await page.locator('#sidebar-model-button').click();
    const modelSidebar = page.locator('#sidebar-content');
    await playwright.expect(modelSidebar).toContainText('AIMET encodings 2.0.0');
    await playwright.expect(modelSidebar).toContainText('2 / 3');
    await playwright.expect(modelSidebar).toContainText("Encoding tensor 'missing.weight' was not found in the model.");
    const persisted = await sidebar.boundingBox();
    playwright.expect(persisted.width).toBe(after.width);

    await page.locator('#sidebar-info-button').click();
    const statistics = page.locator('#sidebar-content');
    await playwright.expect(statistics).toContainText('Node Precision');
    await playwright.expect(statistics.locator('input[value="with QParam"]')).toHaveCount(1);
    await playwright.expect(statistics.locator('input[value="matched QParams"]')).toHaveCount(1);
    await playwright.expect(statistics.locator('input[value="mismatched QParams"]')).toHaveCount(1);
    await playwright.expect(statistics.locator('input[value="MIXED"]')).toHaveCount(1);
    await playwright.expect(statistics.locator('input[value="A8"]')).toHaveCount(1);
    await playwright.expect(statistics.locator('input[value="A16"]')).toHaveCount(1);

    const findModifier = primaryModifier();
    await page.keyboard.press(`${findModifier.key}+f`);
    const findSearch = page.locator('.sidebar-find-search');
    await playwright.expect(findSearch).toBeVisible();
    const findBounds = await findSearch.boundingBox();
    const findSidebarBounds = await sidebar.boundingBox();
    playwright.expect(findBounds.x).toBeGreaterThan(findSidebarBounds.x);
    playwright.expect(findBounds.x + findBounds.width).toBeLessThan(findSidebarBounds.x + findSidebarBounds.width);

    await page.locator('#graph-edit-button').click();
    const warning = page.locator('#graph-edit-warning-overlay');
    await playwright.expect(warning).toBeVisible();
    await playwright.expect(warning).toContainText('AIMET encodings are not updated');
    await page.locator('#graph-edit-warning-cancel').click();
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit/);
    await playwright.expect(page.locator('.node-item-quantization')).toHaveCount(2);

    await page.locator('#graph-edit-button').click();
    await playwright.expect(warning).toBeVisible();
    await page.locator('#graph-edit-warning-continue').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-encodings-disabled/);
    await playwright.expect(page.locator('#graph-edit-encodings-warning')).toBeVisible();
    await playwright.expect(page.locator('.node-item-quantization')).toHaveCount(0);
});

playwright.test('HNNX showcase fixture covers quantization, bundled routing, TopK, and edit mode', async ({ page }) => {
    const self = url.fileURLToPath(import.meta.url);
    const root = path.resolve(path.dirname(self), '..');
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles([
        path.join(root, 'examples', 'hnnx-showcase.onnx'),
        path.join(root, 'examples', 'hnnx-showcase.encodings')
    ]);
    await page.waitForSelector('body.default', { timeout: 10000 });
    await playwright.expect(page.getByText('TopK', { exact: true }).first()).toBeVisible();
    await playwright.expect(page.locator('.node-item-quantization').first()).toBeVisible();
    await playwright.expect(page.locator('.edge-path-bundle')).toHaveCount(1);
    await playwright.expect(page.locator('.edge-label-bundle')).toContainText('×4');
    await playwright.expect(page.locator('.edge-label-bundle')).toContainText('A8');
    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('#graph-edit-warning-overlay')).toBeVisible();
    await page.locator('#graph-edit-warning-continue').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
});

playwright.test('ONNX GraphSurgeon Editor previews input and graph output edits with undo and redo', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'editable.onnx',
        mimeType: 'application/octet-stream',
        buffer: editableOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });

    await playwright.expect(page.locator('#graph-edit-button')).toHaveAttribute('title', /Beta/);
    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Select a connection line');
    await playwright.expect(page.locator('#graph-edit-infer-button')).toBeVisible();
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeVisible();
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toHaveText('REFRESH VIEW');
    await page.setViewportSize({ width: 720, height: 720 });
    const saveButtonBounds = await page.locator('#graph-edit-save-button').boundingBox();
    playwright.expect(saveButtonBounds).not.toBeNull();
    playwright.expect(saveButtonBounds.x + saveButtonBounds.width).toBeLessThanOrEqual(720);
    await page.setViewportSize({ width: 1280, height: 720 });
    const redraw = async () => {
        const button = page.locator('#graph-edit-redraw-button');
        if (await button.isEnabled()) {
            await button.click();
        }
        await playwright.expect(page.locator('#graph-edit-status')).toContainText('Graph view updated');
        await playwright.expect(button).toBeDisabled();
    };
    const measuredInputPort = page.locator('.graph-edit-input-port[aria-label="Connect to neg.X"]');
    const inputPortPosition = svgNumbers(await measuredInputPort.getAttribute('transform'));
    const inputNodePosition = svgNumbers(await measuredInputPort.locator('..').getAttribute('transform'));
    const inputEdgePath = svgNumbers(await page.locator('#edge-alt').getAttribute('d'));
    const inputPortDistance = Math.hypot(
        inputNodePosition[0] + inputPortPosition[0] - inputEdgePath.at(-2),
        inputNodePosition[1] + inputPortPosition[1] - inputEdgePath.at(-1)
    );
    playwright.expect(inputPortDistance).toBeLessThan(1);
    const measuredOutputPort = page.locator('.graph-edit-output-port[aria-label="Use output a"]');
    const outputPortPosition = svgNumbers(await measuredOutputPort.getAttribute('transform'));
    const outputNodePosition = svgNumbers(await measuredOutputPort.locator('..').getAttribute('transform'));
    const outputEdgePath = svgNumbers(await page.locator('#edge-a').getAttribute('d'));
    const outputPortDistance = Math.hypot(
        outputNodePosition[0] + outputPortPosition[0] - outputEdgePath[0],
        outputNodePosition[1] + outputPortPosition[1] - outputEdgePath[1]
    );
    playwright.expect(outputPortDistance).toBeLessThan(1);
    await measuredOutputPort.click({ modifiers: ['Shift'] });
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('OUTPUT SELECTED');
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-connection-replace/);
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);
    const draggableNeg = page.locator('.graph-node').filter({ hasText: 'Neg' }).first();
    const altInputNode = page.locator('#input-name-alt');
    const negTransformBeforeDrag = await draggableNeg.getAttribute('transform');
    const negEdgeBeforeDrag = await page.locator('#edge-alt').getAttribute('d');
    const negBounds = await draggableNeg.boundingBox();
    const altInputBounds = await altInputNode.boundingBox();
    await page.mouse.move(negBounds.x + negBounds.width / 2, negBounds.y + negBounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(
        negBounds.x + negBounds.width / 2 + 48,
        altInputBounds.y - negBounds.height - 80 + negBounds.height / 2,
        { steps: 3 }
    );
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-node-dragging/);
    await page.mouse.up();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText(/Only \d+ connected line/);
    const negTransformAfterDrag = await draggableNeg.getAttribute('transform');
    const negEdgeAfterDrag = await page.locator('#edge-alt').getAttribute('d');
    playwright.expect(negTransformAfterDrag).not.toBe(negTransformBeforeDrag);
    playwright.expect(negEdgeAfterDrag).not.toBe(negEdgeBeforeDrag);
    playwright.expect(negEdgeAfterDrag).toContain('C');
    playwright.expect(negEdgeAfterDrag).not.toContain('Q');
    const movedNegBounds = await draggableNeg.boundingBox();
    const movedAltInputBounds = await altInputNode.boundingBox();
    playwright.expect(movedNegBounds.y + movedNegBounds.height).toBeLessThan(movedAltInputBounds.y);
    const reversedInputPort = page.locator('.graph-edit-input-port[aria-label="Connect to neg.X"]');
    const reversedOutputPort = page.locator('.graph-edit-output-port[aria-label="Use output alt"]');
    const reversedInputPosition = svgNumbers(await reversedInputPort.getAttribute('transform'));
    const reversedOutputPosition = svgNumbers(await reversedOutputPort.getAttribute('transform'));
    const reversedInputNodePosition = svgNumbers(await reversedInputPort.locator('..').getAttribute('transform'));
    const reversedOutputNodePosition = svgNumbers(await reversedOutputPort.locator('..').getAttribute('transform'));
    const reversedEdgePath = svgNumbers(await page.locator('#edge-alt').getAttribute('d'));
    playwright.expect(Math.hypot(
        reversedOutputNodePosition[0] + reversedOutputPosition[0] - reversedEdgePath[0],
        reversedOutputNodePosition[1] + reversedOutputPosition[1] - reversedEdgePath[1]
    )).toBeLessThan(1);
    playwright.expect(Math.hypot(
        reversedInputNodePosition[0] + reversedInputPosition[0] - reversedEdgePath.at(-2),
        reversedInputNodePosition[1] + reversedInputPosition[1] - reversedEdgePath.at(-1)
    )).toBeLessThan(1);
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    await page.locator('#hit-edge-optional_value').click({ force: true });
    await playwright.expect(page.locator('#graph-edit-connection-label')).toContainText('optional_value → clip.max');
    await playwright.expect(page.locator('#graph-edit-connection-disconnect')).toBeEnabled();
    const optionalEdge = page.locator('#edge-optional_value');
    await playwright.expect(optionalEdge).toBeVisible();
    await page.locator('#graph-edit-connection-disconnect').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Disconnected');
    await playwright.expect(optionalEdge).toBeHidden();
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(optionalEdge).toBeVisible();
    await page.locator('#graph-edit-redo-button').click();
    await playwright.expect(optionalEdge).toBeHidden();
    const clipNode = page.getByText('Clip', { exact: true }).first();
    await clipNode.click();
    const nodeMenu = page.locator('#graph-edit-node-menu');
    await playwright.expect(nodeMenu).toBeVisible();
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port')).toHaveCount(3);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-state')).toHaveText([
        'REQUIRED', 'OPTIONAL · NO INPUT', 'OPTIONAL · NO INPUT'
    ]);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-required')).toHaveCount(1);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-optional')).toHaveCount(2);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-required input[type="checkbox"]')).toHaveCount(0);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-optional input[type="checkbox"]')).toHaveCount(2);
    const requiredSearch = nodeMenu.locator('input[aria-label="Tensor for input input"]');
    await requiredSearch.focus();
    const requiredCandidates = nodeMenu.locator('.graph-edit-node-port-required .graph-edit-node-candidate');
    playwright.expect(await requiredCandidates.count()).toBeGreaterThan(1);
    await requiredSearch.fill('a');
    await requiredSearch.press('Enter');
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed clip.input: x → a');
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await clipNode.click();
    const optionalSearch = nodeMenu.locator('input[aria-label="Tensor for optional input max"]');
    await optionalSearch.focus();
    await playwright.expect(nodeMenu.locator('.graph-edit-node-candidate-none')).toHaveText('No input');
    await playwright.expect(nodeMenu.locator('.graph-edit-node-candidate-none')).toHaveClass(/current/);
    await optionalSearch.fill('a');
    await optionalSearch.press('Enter');
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('(disconnected) → a');
    await playwright.expect(optionalEdge).not.toHaveClass(/graph-edit-edge-disconnected/);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await clipNode.click();
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-state')).toHaveText([
        'REQUIRED', 'OPTIONAL · NO INPUT', 'OPTIONAL · ON'
    ]);
    await nodeMenu.locator('input[aria-label="Node name"]').fill('clip_renamed');
    await nodeMenu.locator('input[aria-label="Node name"]').press('Enter');
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Renamed node clip → clip_renamed');
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await clipNode.click();
    const outputName = nodeMenu.locator('input[aria-label="Output tensor clipped"]');
    await outputName.fill('clipped_renamed');
    await outputName.press('Enter');
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Renamed output tensor clipped → clipped_renamed');
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await clipNode.click();
    await playwright.expect(nodeMenu.locator('.danger')).toBeEnabled();
    await nodeMenu.locator('.danger').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Deleted clip');
    await playwright.expect(page.getByText('Clip', { exact: true })).toHaveCount(0);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await page.locator('#hit-edge-alt').first().click({ force: true });
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('CONNECTION SELECTED');
    await playwright.expect(page.locator('#target')).toBeFocused();
    await playwright.expect(page.locator('#graph-edit-connection-actions')).toBeVisible();
    await playwright.expect(page.locator('#graph-edit-connection-label')).toContainText('alt → neg.X');
    await playwright.expect(page.locator('#graph-edit-connection-disconnect')).toBeEnabled();
    const originalAltPath = await page.locator('#edge-alt').getAttribute('d');
    await page.locator('#graph-edit-connection-replace').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-connection-replace/);
    await page.locator('.graph-edit-output-port[aria-label="Use output a"]:visible').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed neg.X: alt → a');
    const replacedAltPath = await page.locator('#edge-alt').getAttribute('d');
    playwright.expect(replacedAltPath).not.toBe(originalAltPath);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#edge-alt')).toHaveAttribute('d', originalAltPath);
    await page.locator('#hit-edge-alt').click({ force: true });
    await playwright.expect(page.locator('#graph-edit-connection-label')).toContainText('alt → neg.X');
    await physicalKey(page, 'ㅇ', 'KeyD', 68);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Required input is now unresolved');
    await playwright.expect(page.locator('#graph-edit-connection-actions')).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toHaveClass(/invalid/);
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeDisabled();
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeEnabled();
    await redraw();
    const missingRequiredPort = page.locator('.graph-edit-input-port-required-missing[aria-label="Connect to neg.X"]');
    await playwright.expect(missingRequiredPort).toBeVisible();
    await playwright.expect(draggableNeg).toHaveAttribute('transform', negTransformAfterDrag);
    const negNode = page.getByText('Neg', { exact: true }).first();
    await negNode.click();
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-required')).toHaveClass(/graph-edit-node-port-disconnected/);
    await playwright.expect(nodeMenu.locator('.graph-edit-node-port-state')).toHaveText(['REQUIRED · MISSING']);
    await playwright.expect(nodeMenu.locator('button', { hasText: 'CLEAR' })).toBeDisabled();
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);
    const sourcePort = page.locator('.graph-edit-output-port[aria-label="Use output a"]').first();
    await playwright.expect(sourcePort).toBeVisible();
    await sourcePort.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Choose a blue input port');
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    await playwright.expect(sourcePort).toHaveClass(/graph-edit-output-port-selected/);
    const targetBounds = await page.locator('#target').boundingBox();
    await page.mouse.move(targetBounds.x + 5, targetBounds.y + 5);
    await page.mouse.down({ button: 'right' });
    await page.mouse.up({ button: 'right' });
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('INCOMPLETE GRAPH');
    const inputPort = page.locator('.graph-edit-input-port[aria-label="Connect to neg.X"]');
    const sourceBounds = await sourcePort.boundingBox();
    await sourcePort.hover();
    await page.mouse.down();
    await playwright.expect(sourcePort).toHaveClass(/graph-edit-output-port-drag-armed/);
    await page.mouse.move(
        sourceBounds.x + sourceBounds.width / 2 + 32,
        sourceBounds.y + sourceBounds.height / 2 + 24,
        { steps: 3 }
    );
    await playwright.expect(page.locator('.graph-edit-connection-preview')).toBeVisible();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    const inputBounds = await inputPort.boundingBox();
    await page.mouse.move(inputBounds.x + inputBounds.width / 2, inputBounds.y + inputBounds.height / 2);
    await playwright.expect(inputPort).toHaveClass(/graph-edit-input-port-drag-target/);
    await playwright.expect(inputPort).not.toHaveClass(/graph-edit-input-port-invalid/);
    await playwright.expect(page.locator('.graph-edit-input-port[aria-label="Connect to abs.X"]')).toHaveClass(/graph-edit-input-port-invalid/);
    await page.mouse.up();
    await playwright.expect(page.locator('.graph-edit-connection-preview')).toHaveCount(0);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText(/Changed|Graph view updated/);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);
    await playwright.expect(page.locator('.edge-path[id^="edge-a"]')).toHaveCount(2);
    // The drag gesture installs a guard for its own synthetic trailing click.
    // It must never consume an immediate click on an unrelated toolbar button.
    await page.locator('#graph-edit-layout-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await redraw();
    }
    const residualEdges = page.locator('.edge-path[id^="edge-a"]');
    await playwright.expect(residualEdges).toHaveCount(2);
    const firstPath = await residualEdges.nth(0).getAttribute('d');
    const secondPath = await residualEdges.nth(1).getAttribute('d');
    playwright.expect(firstPath).not.toBe(secondPath);
    await sourcePort.click({ modifiers: ['Shift'] });
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('OUTPUT SELECTED');
    await playwright.expect(page.locator('#graph-edit-node-menu')).not.toBeVisible();
    await page.keyboard.press('q');
    await sourcePort.click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    await playwright.expect(page.locator('#graph-edit-node-menu')).not.toBeVisible();
    await page.keyboard.press('q');
    await playwright.expect(page.locator('#graph-edit-undo-button')).toBeEnabled();
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();

    const editModifier = primaryModifier();
    await page.keyboard.press(`${editModifier.key}+z`);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Undid');
    await page.keyboard.press(`${editModifier.key}+Shift+z`);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Redid');
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await redraw();
    }

    await page.locator('.graph-edit-output-port[aria-label="Use output b"]:visible').click();
    const graphOutputPort = page.locator('.graph-edit-input-port[aria-label="Connect to graph output y"]:visible');
    await playwright.expect(graphOutputPort).toBeVisible();
    await graphOutputPort.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed graph output');
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await redraw();
    }
    await playwright.expect(page.locator('.graph-item-output:visible')).toHaveCount(1);
    await playwright.expect(page.locator('.graph-item-output:visible')).toHaveText('y');
    await page.locator('.graph-item-output:visible').click();
    await page.locator('#graph-edit-node-menu button', { hasText: 'DISCONNECT' }).click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Output must be reconnected before save');
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeDisabled();
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await redraw();
    }
    const missingGraphOutput = page.locator('.graph-edit-graph-output-port-missing[aria-label="Connect to graph output y"]:visible');
    await playwright.expect(missingGraphOutput).toBeVisible();
    await page.locator('.graph-edit-output-port[aria-label="Use output b"]:visible').click();
    await missingGraphOutput.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed graph output');
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await redraw();
    }
    await page.locator('.graph-item-output:visible').click();
    const graphOutputName = page.locator('#graph-edit-node-menu input[aria-label="Graph output name"]');
    await playwright.expect(graphOutputName).toHaveValue('y');
    await graphOutputName.fill('final_output');
    await graphOutputName.press('Enter');
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Renamed graph output y → final_output');
    await redraw();
    await playwright.expect(page.locator('.graph-item-output:visible').filter({ hasText: /^final_output$/ })).toHaveCount(1);
    await page.locator('#graph-edit-undo-button').click();
    await redraw();
    await playwright.expect(page.locator('.graph-item-output:visible').filter({ hasText: /^y$/ })).toHaveCount(1);

    await page.locator('#graph-edit-layout-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Graph rebuilt and re-laid out');
    await playwright.expect(page.locator('.graph-item-output:visible').filter({ hasText: /^y$/ })).toHaveCount(1);

    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    await playwright.expect(page.locator('#graph-edit-reset-button')).toBeEnabled();
    await page.locator('#graph-edit-reset-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Restored the graph');
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    await playwright.expect(page.locator('#graph-edit-reset-button')).toBeDisabled();
    await playwright.expect(page.locator('#graph-edit-undo-button')).toBeDisabled();
    await playwright.expect(page.locator('#graph-edit-status')).not.toHaveClass(/invalid/);
    await playwright.expect(page.locator('.graph-item-output:visible').filter({ hasText: /^y$/ })).toHaveCount(1);
    await playwright.expect(page.locator('#hit-edge-alt')).toHaveCount(1);
});

playwright.test('HNNX workspace shortcuts enter, inspect, layout, view, and save without stealing text input', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'shortcut-workspace.onnx',
        mimeType: 'application/octet-stream',
        buffer: editableOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await playwright.expect(page.locator(
        '#graph-edit-save-button + #encodings-toggle-button + #graph-edit-layout-button')).toHaveCount(1);
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeVisible();
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    const shortcutModifier = primaryModifier();
    await physicalKey(page, 'ㄴ', 'KeyS', 83, shortcutModifier.mask);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Save failed');

    await physicalKey(page, 'ㄷ', 'KeyE', 69);
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    await playwright.expect(page.locator('#graph-edit-button')).toHaveText('VIEW');

    await page.locator('#graph-edit-add-button').click();
    const search = page.locator('#graph-edit-add-search');
    await search.fill('Relu');
    await page.keyboard.type('view');
    await playwright.expect(search).toHaveValue('Reluview');
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    await page.locator('#graph-edit-add-close').click();

    await physicalKey(page, 'ㄱ', 'KeyR', 82);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    await playwright.expect(page.locator('#graph-edit-layout-button')).toBeEnabled();
    await playwright.expect(page.locator('#graph-edit-layout-button')).toHaveText('RE-LAYOUT');
    await physicalKey(page, 'ㅍ', 'KeyV', 86);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit/);

    await physicalKey(page, 'ㅑ', 'KeyI', 73);
    const inferenceMenu = page.locator('#graph-edit-node-menu.graph-edit-inference-menu');
    await playwright.expect(inferenceMenu).toBeVisible();
    await inferenceMenu.getByText('CLOSE', { exact: true }).click();

    await physicalKey(page, 'ㄷ', 'KeyE', 69);
    await page.getByText('Clip', { exact: true }).first().click();
    await page.keyboard.press('d');
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    await physicalKey(page, 'ㅍ', 'KeyV', 86);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit/);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('unsaved change');
    await playwright.expect(page.locator('#graph-edit-save-button')).toBeEnabled();
    await page.locator('#graph-edit-save-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Save failed');
});

playwright.test('connection drag guard does not consume the next toolbar click', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'drag-guard.onnx', mimeType: 'application/octet-stream', buffer: editableOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();
    await page.locator('#hit-edge-alt').first().click({ force: true });
    await page.locator('#graph-edit-connection-disconnect').click();
    if (await page.locator('#graph-edit-redraw-button').isEnabled()) {
        await page.locator('#graph-edit-redraw-button').click();
        await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    }

    const session = await page.context().newCDPSession(page);
    const dragResult = await session.send('Runtime.evaluate', {
        expression: `(() => {
            const source = document.querySelector('.graph-edit-output-port[aria-label="Use output a"]');
            const target = document.querySelector('.graph-edit-input-port[aria-label="Connect to neg.X"]');
            const from = source.getBoundingClientRect();
            const to = target.getBoundingClientRect();
            const event = (type, x, y, buttons) => new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 73, pointerType: 'mouse',
                isPrimary: true, button: 0, buttons, clientX: x, clientY: y
            });
            source.dispatchEvent(event('pointerdown', from.x + from.width / 2, from.y + from.height / 2, 1));
            document.dispatchEvent(event('pointermove', to.x + to.width / 2, to.y + to.height / 2, 1));
            document.dispatchEvent(event('pointerup', to.x + to.width / 2, to.y + to.height / 2, 0));
            document.getElementById('graph-edit-layout-button').click();
            return true;
        })()`,
        returnByValue: true
    });
    await session.detach();
    playwright.expect(dragResult.result.value).toBe(true);
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
});

playwright.test('ONNX GraphSurgeon Editor deletes a Cast and reconnects without rebuilding the graph', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'cast-delete.onnx', mimeType: 'application/octet-stream', buffer: castDeletionOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();

    const cast = page.getByText('Cast', { exact: true }).first();
    await cast.click();
    await page.locator('#graph-edit-node-menu .danger').click();
    await playwright.expect(cast).toHaveCount(0);
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();

    const source = page.locator('.graph-edit-output-port[aria-label="Use output a"]');
    const addInput = page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]');
    await source.click();
    await addInput.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed add.A: (disconnected) → a');
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    await playwright.expect(page.locator('.edge-path[id^="edge-a-"]')).toHaveCount(1);

    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.getByText('Cast', { exact: true })).toHaveCount(1);
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    await page.locator('#graph-edit-redo-button').click();
    await page.locator('#graph-edit-redo-button').click();
    await playwright.expect(page.locator('.edge-path[id^="edge-a-"]')).toHaveCount(1);
    playwright.expect(errors).toEqual([]);
});

playwright.test('ONNX GraphSurgeon Editor restores delete and reconnect history after a full render', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'cast-render-history.onnx', mimeType: 'application/octet-stream', buffer: castDeletionOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();

    const cast = () => page.getByText('Cast', { exact: true }).first();
    await cast().click();
    await page.locator('#graph-edit-node-menu .danger').click();
    await page.locator('.graph-edit-output-port[aria-label="Use output a"]').click();
    await page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]').click();
    await playwright.expect(page.locator('.edge-path[id^="edge-a-"]')).toHaveCount(1);

    // Shape inference and re-layout both replace the complete SVG tree. The
    // history must stop referring to the detached pre-render node and edges.
    await page.waitForTimeout(1000);
    await page.locator('#graph-edit-layout-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    const firstLayout = await page.context().newCDPSession(page);
    const firstLayoutResult = await firstLayout.send('Runtime.evaluate', {
        expression: `JSON.stringify({
            nodes: Array.from(document.querySelectorAll('.graph-node')).map((node) => node.getAttribute('transform')),
            edges: Array.from(document.querySelectorAll('.edge-paths path')).map((edge) => edge.getAttribute('d'))
        })`,
        returnByValue: true
    });
    await firstLayout.detach();
    await page.waitForTimeout(1000);
    await page.locator('#graph-edit-layout-button').click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    const secondLayout = await page.context().newCDPSession(page);
    const secondLayoutResult = await secondLayout.send('Runtime.evaluate', {
        expression: `JSON.stringify({
            nodes: Array.from(document.querySelectorAll('.graph-node')).map((node) => node.getAttribute('transform')),
            edges: Array.from(document.querySelectorAll('.edge-paths path')).map((edge) => edge.getAttribute('d'))
        })`,
        returnByValue: true
    });
    await secondLayout.detach();
    playwright.expect(secondLayoutResult.result.value).toEqual(firstLayoutResult.result.value);
    await page.locator('#graph-edit-button').click();
    await page.locator('#graph-edit-button').click();

    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(cast()).toHaveCount(0);
    await playwright.expect(page.locator('#edge-a, .edge-path[id^="edge-a-"]')).toHaveCount(0);
    await playwright.expect(page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]').first())
        .toHaveClass(/graph-edit-input-port-required-missing/);

    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(cast()).toHaveCount(1);
    await playwright.expect(page.locator('#edge-b')).toHaveCount(1);
    playwright.expect(errors).toEqual([]);
});

playwright.test('ONNX GraphSurgeon Editor preserves D and Q shortcuts across focus states', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'shortcut.onnx', mimeType: 'application/octet-stream', buffer: castDeletionOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();

    const cast = () => page.getByText('Cast', { exact: true }).first();
    await cast().click();
    await physicalKey(page, 'ㅇ', 'KeyD', 68);
    await playwright.expect(cast()).toHaveCount(0);
    await playwright.expect(page.locator('#graph-edit-redraw-button')).toBeDisabled();
    const historyModifier = primaryModifier();
    await physicalKey(page, 'ㅋ', 'KeyZ', 90, historyModifier.mask);
    await playwright.expect(cast()).toHaveCount(1);
    await physicalKey(page, 'ㅋ', 'KeyZ', 90, historyModifier.mask | 8);
    await playwright.expect(cast()).toHaveCount(0);
    await physicalKey(page, 'ㅋ', 'KeyZ', 90, historyModifier.mask);
    await playwright.expect(cast()).toHaveCount(1);

    await cast().click();
    const menu = page.locator('#graph-edit-node-menu');
    const name = menu.locator('input[aria-label="Node name"]');
    await name.focus();
    await page.keyboard.press('d');
    await page.keyboard.press('q');
    await playwright.expect(name).toHaveValue('castdq');
    await playwright.expect(cast()).toHaveCount(1);
    await playwright.expect(menu).toBeVisible();

    await page.locator('#target').focus();
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);
    await playwright.expect(menu).not.toBeVisible();
    await page.locator('.graph-edit-output-port[aria-label="Use output a"]').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-source/);

    await cast().click();
    await page.keyboard.press('d');
    await playwright.expect(cast()).toHaveCount(0);
    playwright.expect(errors).toEqual([]);
});

playwright.test('ONNX GraphSurgeon Editor applies D to graph inputs and outputs', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'endpoints.onnx', mimeType: 'application/octet-stream', buffer: editableOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();

    const graphInput = page.locator('#input-name-alt:visible');
    await graphInput.click();
    await playwright.expect(page.locator('#graph-edit-node-menu')).toContainText('Graph input');
    await physicalKey(page, 'ㅇ', 'KeyD', 68);
    await playwright.expect(page.locator('#input-name-alt:visible')).toHaveCount(0);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('#input-name-alt:visible')).toHaveCount(1);

    await page.locator('.graph-item-output:visible').click();
    await playwright.expect(page.locator('#graph-edit-node-menu')).toContainText('Graph output');
    await page.keyboard.press('d');
    await playwright.expect(page.locator('.graph-item-output:visible')).toHaveCount(0);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('.graph-item-output:visible')).toHaveCount(1);
    playwright.expect(errors).toEqual([]);
});

playwright.test('ONNX GraphSurgeon Editor rejects cycles and clears redo after a new branch', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'validation.onnx', mimeType: 'application/octet-stream', buffer: castDeletionOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();

    const outputY = page.locator('.graph-edit-output-port[aria-label="Use output y"]');
    await outputY.click();
    const selfInput = page.locator('.graph-edit-input-port[aria-label="Connect to add.A"]');
    await playwright.expect(selfInput).toHaveClass(/graph-edit-input-port-invalid/);
    await playwright.expect(selfInput.locator('title')).toContainText('own output');
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);

    await outputY.click();
    const cycleInput = page.locator('.graph-edit-input-port[aria-label="Connect to cast.input"]');
    await playwright.expect(cycleInput).toHaveClass(/graph-edit-input-port-invalid/);
    await playwright.expect(cycleInput.locator('title')).toContainText('cycle');
    await physicalKey(page, 'ㅂ', 'KeyQ', 81);

    await page.getByText('Cast', { exact: true }).first().click();
    await page.keyboard.press('d');
    await page.locator('#graph-edit-undo-button').click();
    await page.getByText('Relu', { exact: true }).first().click();
    await page.keyboard.press('d');
    await playwright.expect(page.locator('#graph-edit-redo-button')).toBeDisabled();
});

playwright.test('ONNX GraphSurgeon Editor keeps close multi-output routes compact', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'routing.onnx',
        mimeType: 'application/octet-stream',
        buffer: routingOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    const relayout = page.locator('#graph-edit-layout-button');
    await playwright.expect(relayout).toBeVisible();
    const initialRoutes = await Promise.all(['left', 'right'].map((name) =>
        page.locator(`.edge-paths #edge-${name}`).first().getAttribute('d')));
    await relayout.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    const viewRoutes = await Promise.all(['left', 'right'].map((name) =>
        page.locator(`.edge-paths #edge-${name}`).first().getAttribute('d')));
    playwright.expect(viewRoutes).toEqual(initialRoutes);

    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit/);
    const editRoutes = await Promise.all(['left', 'right'].map((name) =>
        page.locator(`.edge-paths #edge-${name}`).first().getAttribute('d')));
    playwright.expect(editRoutes).toEqual(viewRoutes);

    const concat = page.locator('.graph-node:visible').filter({ hasText: 'Concat' }).first();
    const bounds = await concat.boundingBox();
    await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width / 2 + 20, bounds.y + bounds.height / 2 + 12);
    await page.mouse.up();

    const movedTransform = await concat.getAttribute('transform');
    await relayout.click();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('re-laid out');
    const restoredTransform = await concat.getAttribute('transform');
    playwright.expect(restoredTransform).not.toEqual(movedTransform);

    const routes = [page.locator('.edge-paths #edge-left').first(), page.locator('.edge-paths #edge-right').first()];
    const paths = await Promise.all(routes.map((route) => route.getAttribute('d')));
    for (const data of paths) {
        playwright.expect(data).toMatch(/^M/);
        playwright.expect(data).toContain('C');
        playwright.expect(data).not.toContain('A');
        playwright.expect(data).not.toContain('Q');
    }
    const session = await page.context().newCDPSession(page);
    const alignment = await session.send('Runtime.evaluate', {
        expression: `(() => {
            const path = document.querySelector('#edge-left');
            const port = Array.from(document.querySelectorAll(
                '.graph-edit-output-port[aria-label="Use output left"]')).find((element) =>
                element.getClientRects().length > 0);
            const point = path.getPointAtLength(0).matrixTransform(path.getScreenCTM());
            const matrix = port.getScreenCTM();
            return Math.hypot(point.x - matrix.e, point.y - matrix.f);
        })()`,
        returnByValue: true
    });
    playwright.expect(alignment.result.value).toBeLessThan(1);

    // A connection drag is the only gesture that enables edge auto-pan.
    // Give the canvas a deterministic off-screen region through CDP because
    // Netron intentionally disables window.eval() in its browser surface.
    const target = page.locator('#target');
    await session.send('Runtime.evaluate', {
        expression: 'document.getElementById("canvas").style.minHeight = "3000px"'
    });
    const scrollTop = async () => {
        const result = await session.send('Runtime.evaluate', {
            expression: 'document.getElementById("target").scrollTop',
            returnByValue: true
        });
        return result.result.value;
    };
    const source = page.locator('.graph-edit-output-port[aria-label="Use output left"]');
    await source.scrollIntoViewIfNeeded();
    const sourceBounds = await source.boundingBox();
    const targetBounds = await target.boundingBox();
    const scrollBefore = await scrollTop();
    await page.mouse.move(sourceBounds.x + sourceBounds.width / 2, sourceBounds.y + sourceBounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(targetBounds.x + targetBounds.width / 2, targetBounds.y + targetBounds.height - 2);
    await page.waitForTimeout(200);
    const scrollAfter = await scrollTop();
    await page.mouse.up();
    await session.detach();
    playwright.expect(scrollAfter).toBeGreaterThan(scrollBefore);

    await page.locator('#graph-edit-infer-button').click();
    const inferenceMenu = page.locator('#graph-edit-node-menu.graph-edit-inference-menu');
    await playwright.expect(inferenceMenu).toBeVisible();
    await playwright.expect(inferenceMenu.locator('.graph-edit-node-menu-title')).toHaveText('Shape inference failed');
    await playwright.expect(inferenceMenu.locator('.graph-edit-inference-problem'))
        .toContainText('require the macOS app or VS Code extension');
});

playwright.test('fan-out edges and edit ports share one tensor output origin', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'fanout-routing.onnx',
        mimeType: 'application/octet-stream',
        buffer: fanoutRoutingOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await playwright.expect(page.locator('.edge-paths path[id^="edge-a"]')).toHaveCount(2);
    const initial = await page.context().newCDPSession(page);
    const initialResult = await initial.send('Runtime.evaluate', {
        expression: `Array.from(document.querySelectorAll('.edge-paths path[id^="edge-a"]')).map((path) => {
            const point = path.getPointAtLength(0).matrixTransform(path.getScreenCTM());
            return { x: point.x, y: point.y };
        })`,
        returnByValue: true
    });
    await initial.detach();
    playwright.expect(initialResult.result.value).toHaveLength(2);
    playwright.expect(Math.hypot(
        initialResult.result.value[0].x - initialResult.result.value[1].x,
        initialResult.result.value[0].y - initialResult.result.value[1].y
    )).toBeLessThan(1);
    await page.locator('#graph-edit-button').click();
    const ports = page.locator('.graph-edit-output-port[aria-label="Use output a"]:visible');
    await playwright.expect(ports).toHaveCount(1);
    const session = await page.context().newCDPSession(page);
    const result = await session.send('Runtime.evaluate', {
        expression: `(() => {
            const starts = Array.from(document.querySelectorAll('.edge-paths path[id^="edge-a"]')).map((path) => {
                const point = path.getPointAtLength(0).matrixTransform(path.getScreenCTM());
                return { x: point.x, y: point.y };
            });
            const ports = Array.from(document.querySelectorAll(
                '.graph-edit-output-port[aria-label="Use output a"]')).filter((port) =>
                port.getClientRects().length > 0).map((port) => {
                const matrix = port.getScreenCTM();
                return { x: matrix.e, y: matrix.f };
            });
            return starts.map((start) => Math.min(...ports.map((port) =>
                Math.hypot(start.x - port.x, start.y - port.y))));
        })()`,
        returnByValue: true
    });
    await session.detach();
    playwright.expect(result.result.value).toHaveLength(2);
    for (const distance of result.result.value) {
        playwright.expect(distance).toBeLessThan(1);
    }

    await page.locator('#graph-edit-add-button').click();
    await page.locator('#graph-edit-add-search').fill('Relu');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Relu/ }).click();
    await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill('relu_fanout');
    await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill('relu_fanout_output');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD RELU' }).click();
    await ports.first().click();
    await page.locator('.graph-edit-input-port[aria-label="Connect to relu_fanout.X"]').click();
    await playwright.expect(ports).toHaveCount(1);
    const updated = await page.context().newCDPSession(page);
    const updatedResult = await updated.send('Runtime.evaluate', {
        expression: `(() => {
            const starts = Array.from(document.querySelectorAll('.edge-paths path[id^="edge-a"]')).map((path) => {
                const point = path.getPointAtLength(0).matrixTransform(path.getScreenCTM());
                return { x: point.x, y: point.y };
            });
            const ports = Array.from(document.querySelectorAll(
                '.graph-edit-output-port[aria-label="Use output a"]')).filter((port) =>
                port.getClientRects().length > 0).map((port) => {
                const matrix = port.getScreenCTM();
                return { x: matrix.e, y: matrix.f };
            });
            return starts.map((start) => Math.min(...ports.map((port) =>
                Math.hypot(start.x - port.x, start.y - port.y))));
        })()`,
        returnByValue: true
    });
    await updated.detach();
    playwright.expect(updatedResult.result.value).toHaveLength(3);
    for (const distance of updatedResult.result.value) {
        playwright.expect(distance).toBeLessThan(1);
    }
});

playwright.test('Split to Concat parallel tensors stay in one fixed edge bundle', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles([
        { name: 'bundle.onnx', mimeType: 'application/octet-stream', buffer: bundledRoutingOnnx() },
        { name: 'bundle.encodings', mimeType: 'application/json', buffer: bundledRoutingEncodings() }
    ]);
    await page.waitForSelector('body.default', { timeout: 10000 });

    const bundlePath = page.locator('.edge-path-bundle');
    const bundleLabel = page.locator('.edge-label-bundle');
    await playwright.expect(bundlePath).toHaveCount(1);
    await playwright.expect(bundlePath).toHaveAttribute('id', 'edge-part_3-3');
    await playwright.expect(bundleLabel).toContainText('\u00D732 \u00B7 1\u00D764\u00D71\u00D72048 \u00B7 A8');
    await playwright.expect(bundleLabel).not.toContainText(/\u25B8|\u25BE/);
    await playwright.expect(page.locator('.edge-label-bundle-toggle')).toHaveCount(0);
    await playwright.expect(page.locator('.edge-path-bundle-hidden')).toHaveCount(0);
    await playwright.expect(page.locator('.edge-path[id^="edge-part_"]')).toHaveCount(1);
    await page.locator('#hit-edge-part_3-3').click({ force: true });
    await playwright.expect(page.locator('#sidebar')).toBeVisible();
    await page.locator('#sidebar-closebutton').click();

    await page.locator('#zoom-in-button').click({ clickCount: 20 });
    await page.waitForTimeout(200);
    const target = page.locator('#target');
    const targetBox = await target.boundingBox();
    await page.mouse.move(targetBox.x + targetBox.width / 2, targetBox.y + targetBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(targetBox.x + targetBox.width / 2 - 160, targetBox.y + targetBox.height / 2 - 100);
    await page.mouse.up();
    const split = page.locator('.graph-node').filter({ hasText: 'Split' }).first();
    const beforeNodeTransform = await split.getAttribute('transform');
    const beforeOriginTransform = await page.locator('#origin').getAttribute('transform');
    await page.locator('#graph-edit-button').click();
    await playwright.expect(bundlePath).toHaveCount(1);
    await playwright.expect(split).toHaveAttribute('transform', beforeNodeTransform);
    await playwright.expect(page.locator('#origin')).toHaveAttribute('transform', beforeOriginTransform);
    await page.locator('#graph-edit-warning-continue').click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit(?:\s|$)/);
    const bundledOutput = split.locator('.graph-edit-output-port');
    await playwright.expect(bundledOutput).toHaveCount(1);
    await playwright.expect(bundledOutput).toHaveAttribute('aria-label', 'Choose bundled outputs ×32');
    await playwright.expect(bundledOutput.locator('.graph-edit-port-bundle-count')).toHaveText('×32');
    await playwright.expect(bundledOutput.locator('title')).toContainText('choose one of 8 ONNX output tensors');
    await bundledOutput.locator('.graph-edit-output-port-marker').click();
    const outputSearch = page.locator('.graph-edit-port-choice-search');
    await playwright.expect(outputSearch).toBeVisible();
    await outputSearch.fill('part_7');
    const outputChoice = page.locator('.graph-edit-port-choice:visible');
    await playwright.expect(outputChoice).toHaveCount(1);
    await outputChoice.click();
    await playwright.expect(page.locator('html')).toHaveClass(/onnx-graph-edit-source/);
    const concat = page.locator('.graph-node').filter({ hasText: 'Concat' }).first();
    const bundledInput = concat.locator('.graph-edit-input-port');
    await playwright.expect(bundledInput).toHaveCount(1);
    await playwright.expect(bundledInput).toHaveAttribute('aria-label', 'Choose bundled inputs ×32 for concat_8');
    await bundledInput.locator('.graph-edit-input-port-marker').click();
    await playwright.expect(page.locator('.graph-edit-port-choice-menu')).toBeVisible();
    await page.locator('.graph-edit-port-choice-search').fill('inputs[31]');
    await playwright.expect(page.locator('.graph-edit-port-choice:visible')).toHaveCount(1);
    await page.keyboard.press('q');

    // A bundled connection must remain reconnectable through its exact
    // Split output and Concat input slot after the representative edge is
    // disconnected.
    await page.locator('#hit-edge-part_3-3').click({ force: true });
    await page.keyboard.press('d');
    await playwright.expect(bundlePath).toBeHidden();
    const sourceBounds = await bundledOutput.locator('.graph-edit-output-port-marker').boundingBox();
    const targetBounds = await bundledInput.locator('.graph-edit-input-port-marker').boundingBox();
    await page.mouse.move(
        sourceBounds.x + sourceBounds.width / 2,
        sourceBounds.y + sourceBounds.height / 2
    );
    await page.mouse.down();
    await page.mouse.move(
        targetBounds.x + targetBounds.width / 2,
        targetBounds.y + targetBounds.height / 2,
        { steps: 4 }
    );
    await playwright.expect(page.locator('.graph-edit-connection-preview')).toBeVisible();
    await page.mouse.up();
    await playwright.expect(bundlePath).toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Changed');
});

playwright.test('ONNX GraphSurgeon Editor moves endpoints and adds graph items', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }
    const chooser = page.waitForEvent('filechooser');
    await page.locator('#open-file-button').click();
    await (await chooser).setFiles({
        name: 'editable.onnx',
        mimeType: 'application/octet-stream',
        buffer: editableOnnx()
    });
    await page.waitForSelector('body.default', { timeout: 10000 });
    await page.locator('#graph-edit-button').click();
    await playwright.expect(page.locator('#input-name-x')).toHaveCount(1);

    const dragNode = async (node, dx = 36, dy = 24, during = null) => {
        const before = await node.getAttribute('transform');
        const bounds = await node.boundingBox();
        await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
        await page.mouse.down();
        await page.mouse.move(bounds.x + bounds.width / 2 + dx, bounds.y + bounds.height / 2 + dy, { steps: 3 });
        if (during) {
            await during();
        }
        await page.mouse.up();
        await playwright.expect(node).not.toHaveAttribute('transform', before);
        await playwright.expect(page.locator('#graph-edit-status')).toContainText('Only');
    };
    await dragNode(page.locator('#input-name-x'));
    const graphOutput = page.locator('.node.graph-output').first();
    const canvas = page.locator('#canvas');
    const canvasWidth = Number(await canvas.getAttribute('width'));
    await dragNode(graphOutput, 520, 260, async () => {
        await playwright.expect.poll(async () => Number(await canvas.getAttribute('width')))
            .toBeGreaterThan(canvasWidth);
    });
    await playwright.expect(page.locator('html')).not.toHaveClass(/onnx-graph-edit-node-dragging/);
    const settledWidth = await canvas.getAttribute('width');
    const settledCanvasBounds = await canvas.boundingBox();
    await page.mouse.move(20, 20);
    await page.mouse.move(700, 500, { steps: 5 });
    await page.waitForTimeout(50);
    await playwright.expect(canvas).toHaveAttribute('width', settledWidth);
    const finalCanvasBounds = await canvas.boundingBox();
    playwright.expect(finalCanvasBounds.x).toBe(settledCanvasBounds.x);
    playwright.expect(finalCanvasBounds.y).toBe(settledCanvasBounds.y);
    playwright.expect(Number(await canvas.getAttribute('width'))).toBeGreaterThan(canvasWidth);
    const canvasBounds = await canvas.boundingBox();
    const outputBounds = await graphOutput.boundingBox();
    playwright.expect(outputBounds.x + outputBounds.width)
        .toBeLessThanOrEqual(canvasBounds.x + canvasBounds.width + 1);
    playwright.expect(outputBounds.y + outputBounds.height)
        .toBeLessThanOrEqual(canvasBounds.y + canvasBounds.height + 1);

    const addButton = page.locator('#graph-edit-add-button');
    const overlay = page.locator('#graph-edit-add-overlay');
    const search = page.locator('#graph-edit-add-search');
    await addButton.click();
    await playwright.expect(overlay).toBeVisible();
    await search.fill('Relu');
    await page.locator('.graph-edit-add-item', { hasText: /^Relu/ }).click();
    const addedNodeName = page.locator('#graph-edit-add-form input[aria-label="Node name"]');
    const addedNodeInput = page.locator('#graph-edit-add-form input[aria-label="X"]');
    await playwright.expect(addedNodeInput).toHaveValue('');
    await playwright.expect(addedNodeInput).toHaveAttribute('placeholder', 'No input selected');
    await addedNodeName.fill('relu_addedX');
    await addedNodeName.press('Backspace');
    await playwright.expect(addedNodeName).toHaveValue('relu_added');
    await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill('relu_result');
    await addedNodeInput.fill('x');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD RELU' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added Relu node relu_added');
    const addedRelu = page.locator('.graph-node').filter({ hasText: 'Relu' });
    await playwright.expect(addedRelu).toHaveCount(1);
    await addedRelu.click();
    await page.keyboard.press('d');
    await playwright.expect(addedRelu).toHaveCount(0);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(addedRelu).toHaveCount(1);

    await addButton.click();
    await search.fill('Split');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Split/ }).click();
    const splitName = page.locator('#graph-edit-add-form input[aria-label="Node name"]');
    const splitInput = page.locator('#graph-edit-add-form input[aria-label="input"]');
    const splitCount = page.locator('#graph-edit-add-form input[aria-label="Output count"]');
    const splitOutputs = page.locator('#graph-edit-add-form input[aria-label="Output tensors (comma-separated)"]');
    const splitSizes = page.locator('#graph-edit-add-form input[aria-label="Split sizes (optional)"]');
    await playwright.expect(splitCount).toHaveValue('2');
    await playwright.expect(splitOutputs).toHaveValue('split_output_0, split_output_1');
    await splitCount.fill('3');
    await playwright.expect(splitOutputs).toHaveValue('split_output_0, split_output_1, split_output_2');
    await splitName.fill('split_added');
    await splitInput.fill('x');
    await splitSizes.fill('1, 1, 2');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD SPLIT' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added Split node split_added');
    await playwright.expect(page.locator('.graph-node').filter({ hasText: 'Split' })).toHaveCount(1);

    await addButton.click();
    await search.fill('TopK');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^TopK/ }).click();
    await playwright.expect(page.locator('#graph-edit-add-form input[aria-label="K"]')).toHaveValue('1');
    await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill('topk_added');
    await page.locator('#graph-edit-add-form input[aria-label="Values output tensor"]').fill('top_values');
    await page.locator('#graph-edit-add-form input[aria-label="Indices output tensor"]').fill('top_indices');
    await page.locator('#graph-edit-add-form input[aria-label="X"]').fill('x');
    await page.locator('#graph-edit-add-form input[aria-label="K"]').fill('2');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD TOPK' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added TopK node topk_added');
    await playwright.expect(page.locator('.graph-node').filter({ hasText: 'TopK' })).toHaveCount(1);

    await addButton.click();
    await search.fill('');
    await Promise.all(['LayerNormalization', 'Clip', 'GatherElements', 'Tile', 'ArgMax', 'ArgMin'].map((name) =>
        playwright.expect(page.locator('.graph-edit-add-item:visible', { hasText: new RegExp(`^${name}`) })).toHaveCount(1)));
    await page.locator('#graph-edit-add-close').click();

    await addButton.click();
    await search.fill('Squeeze');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Squeeze/ }).click();
    await playwright.expect(page.locator('#graph-edit-add-form input[aria-label="Axes (optional)"]')).toHaveValue('');
    await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill('squeeze_added');
    await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill('squeeze_result');
    await page.locator('#graph-edit-add-form input[aria-label="data"]').fill('x');
    await page.locator('#graph-edit-add-form input[aria-label="Axes (optional)"]').fill('0');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD SQUEEZE' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added Squeeze node squeeze_added');

    await addButton.click();
    await search.fill('ReduceSum');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^ReduceSum/ }).click();
    await playwright.expect(page.locator('#graph-edit-add-form input[aria-label="Axes (optional)"]')).toHaveValue('-1');
    await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill('reduce_sum_added');
    await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill('reduce_sum_result');
    await page.locator('#graph-edit-add-form input[aria-label="data"]').fill('x');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD REDUCESUM' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added ReduceSum node reduce_sum_added');

    await addButton.click();
    await search.fill('Slice');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Slice/ }).click();
    await page.locator('#graph-edit-add-form input[aria-label="Node name"]').fill('slice_added');
    await page.locator('#graph-edit-add-form input[aria-label="Output tensor"]').fill('slice_result');
    await page.locator('#graph-edit-add-form input[aria-label="data"]').fill('x');
    await page.locator('#graph-edit-add-form input[aria-label="Starts (comma-separated)"]').fill('0');
    await page.locator('#graph-edit-add-form input[aria-label="Ends (comma-separated)"]').fill('1');
    await page.locator('#graph-edit-add-form input[aria-label="Axes (optional)"]').fill('0');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD SLICE' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added Slice node slice_added');

    await addButton.click();
    await search.fill('Graph Input');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Graph Input/ }).click();
    await page.locator('#graph-edit-add-form input[aria-label="Input name"]').fill('extra_input');
    await page.locator('#graph-edit-add-form input[aria-label="Shape"]').fill('1, 4');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD GRAPH INPUT' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added graph input extra_input');
    const addedInput = page.locator('#input-name-extra_input:visible');
    await playwright.expect(addedInput).toHaveCount(1);
    await addedInput.click();
    await page.keyboard.press('d');
    await playwright.expect(addedInput).toHaveCount(0);
    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(addedInput).toHaveCount(1);

    await addButton.click();
    await playwright.expect(overlay).toBeVisible();
    await search.fill('Graph Output');
    await page.locator('.graph-edit-add-item:visible', { hasText: /^Graph Output/ }).click();
    await page.locator('#graph-edit-add-form input[aria-label="Source tensor"]').fill('relu_result');
    await page.locator('#graph-edit-add-form input[aria-label="Output name"]').fill('relu_output');
    await page.locator('#graph-edit-add-form button', { hasText: 'ADD GRAPH OUTPUT' }).click();
    await playwright.expect(overlay).not.toBeVisible();
    await playwright.expect(page.locator('#graph-edit-status')).toContainText('Added graph output relu_output');
    await playwright.expect(page.locator('.node.graph-output').filter({ hasText: 'relu_output' })).toHaveCount(1);

    await page.locator('#graph-edit-undo-button').click();
    await playwright.expect(page.locator('.node.graph-output').filter({ hasText: 'relu_output' })).toHaveCount(0);
    await page.locator('#graph-edit-redo-button').click();
    await playwright.expect(page.locator('.node.graph-output').filter({ hasText: 'relu_output' })).toHaveCount(1);
});

playwright.test('runtime theme override switches all bundled dark media rules', async ({ page }) => {
    await page.goto('http://127.0.0.1:8765/dist/web/');
    await page.waitForSelector('body.welcome', { timeout: 10000 });
    const themes = await page.evaluate(async () => {
        const module = await import('./view.js');
        const themeView = Object.create(module.View.prototype);
        themeView._host = { document: window.document };
        themeView._themePreference = 'auto';
        themeView._themeMediaRules = null;
        themeView.setTheme('dark', 'dark');
        const dark = {
            preference: window.document.documentElement.dataset.theme,
            effective: window.document.documentElement.dataset.effectiveTheme,
            background: window.getComputedStyle(window.document.body).backgroundColor
        };
        themeView.setTheme('light', 'light');
        const light = {
            preference: window.document.documentElement.dataset.theme,
            effective: window.document.documentElement.dataset.effectiveTheme,
            background: window.getComputedStyle(window.document.body).backgroundColor
        };
        return { dark, light };
    });
    playwright.expect(themes.dark).toEqual({
        preference: 'dark',
        effective: 'dark',
        background: 'rgb(32, 41, 54)'
    });
    playwright.expect(themes.light).toEqual({
        preference: 'light',
        effective: 'light',
        background: 'rgb(236, 236, 236)'
    });
});
