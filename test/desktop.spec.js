
import * as fs from 'fs';
import * as path from 'path';
import * as playwright from '@playwright/test';
import * as url from 'url';

playwright.test.setTimeout(120000);

playwright.test('desktop', async () => {

    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    // Keep the desktop smoke test self-contained. The upstream Netron test
    // referenced a separately downloaded third_party model, which is not part
    // of an HNNX checkout or a Gitea Actions workspace.
    const file = path.resolve(dir, 'aimet.onnx');
    const encodings = path.resolve(dir, 'aimet.encodings');
    playwright.expect(fs.existsSync(file)).toBeTruthy();
    playwright.expect(fs.existsSync(encodings)).toBeTruthy();

    // Launch app
    const electron = await playwright._electron;
    const args = ['.', '--no-sandbox'];
    const app = await electron.launch({ args });
    const page = await app.firstWindow();

    playwright.expect(page).toBeDefined();
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 25000 });
    await page.waitForTimeout(1000);

    const consent = await page.locator('#message-button');
    if (await consent.isVisible({ timeout: 25000 })) {
        await consent.click();
    }

    // Open the model
    await app.evaluate(async (electron, location) => {
        const windows = electron.BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            const window = windows[0];
            window.webContents.send('open', { path: location });
        }
    }, file);

    // Wait for the graph to render
    await page.waitForSelector('#canvas', { state: 'attached', timeout: 10000 });
    await page.waitForSelector('body.default', { timeout: 10000 });

    await playwright.expect(page.getByText('Abs', { exact: true })).toBeVisible();

    const fixedToolbarButtons = [
        '#graph-edit-save-button',
        '#graph-edit-infer-button',
        '#graph-edit-layout-button',
        '#graph-edit-button'
    ];
    const toolbarPositionsBeforeEncodings = await page.locator(fixedToolbarButtons.join(', ')).evaluateAll((elements, selectors) => {
        const positions = new Map(elements.map((element) => [element.id, element.getBoundingClientRect().x]));
        return selectors.map((selector) => positions.get(selector.slice(1)));
    }, fixedToolbarButtons);
    await playwright.expect(page.locator(
        '#graph-edit-save-button + #graph-edit-infer-button + #graph-edit-layout-button + #graph-edit-button + #encodings-toggle-button')).toHaveCount(1);

    // Desktop attachments bypass browser.Host._openContext(). Verify that the
    // native path still records the source and reveals the ENC control after
    // the quantization data has been applied.
    await app.evaluate(async (electron, location) => {
        const windows = electron.BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            windows[0].webContents.send('open', { path: location });
        }
    }, encodings);
    await page.waitForSelector('.node-item-quantization', { timeout: 10000 });
    await playwright.expect(page.locator('html')).toHaveClass(/has-encodings/);
    const encodingsToggle = page.locator('#encodings-toggle-button');
    await playwright.expect(encodingsToggle).toBeVisible();
    await playwright.expect(encodingsToggle).toHaveAttribute('title', /Hide AIMET encodings/);
    const toolbarPositionsAfterEncodings = await page.locator(fixedToolbarButtons.join(', ')).evaluateAll((elements, selectors) => {
        const positions = new Map(elements.map((element) => [element.id, element.getBoundingClientRect().x]));
        return selectors.map((selector) => positions.get(selector.slice(1)));
    }, fixedToolbarButtons);
    playwright.expect(toolbarPositionsAfterEncodings).toEqual(toolbarPositionsBeforeEncodings);

    // Open find sidebar and verify that the rendered HNNX graph is searchable.
    await app.evaluate(async (electron) => {
        const windows = electron.BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            const window = windows[0];
            window.webContents.send('find', {});
        }
    });
    await page.waitForTimeout(500);
    const search = await page.waitForSelector('#search', { state: 'visible', timeout: 5000 });
    playwright.expect(search).toBeDefined();

    await search.fill('Abs');
    await page.waitForSelector('.sidebar-find-content li', { state: 'attached' });
    const item = await page.waitForSelector('.sidebar-find-content li:has-text("Abs")');
    await item.dblclick();
    await playwright.expect(page.locator('#sidebar-content')).toContainText('Abs');

    await app.close();
});
