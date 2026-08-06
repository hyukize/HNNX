import playwright from '@playwright/test';

export default playwright.defineConfig({
    outputDir: '../dist/editor-mixed-test-results',
    reporter: './playwright.reporter.js',
    testMatch: '**/editor-mixed-workflows.browser.spec.js',
    webServer: {
        command: 'python3 -m http.server 8765 --bind 127.0.0.1',
        cwd: process.cwd(),
        url: 'http://127.0.0.1:8765/dist/web/',
        reuseExistingServer: true
    }
});
