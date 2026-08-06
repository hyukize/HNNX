import playwright from '@playwright/test';

export default playwright.defineConfig({
    outputDir: '../dist/aimet-test-results',
    reporter: './playwright.reporter.js',
    testMatch: '**/aimet.browser.spec.js'
});
