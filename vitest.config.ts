import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        // Enable globals (describe, it, expect) without imports
        globals: true,

        // Use Node environment for TypeScript testing
        environment: 'node',

        // Include test files
        include: ['src/**/*.test.ts', 'src/**/*.spec.ts'],

        // Coverage configuration
        coverage: {
            provider: 'v8',
            reporter: ['text', 'html', 'lcov'],
            include: ['src/**/*.ts'],
            exclude: ['src/**/*.test.ts', 'src/**/*.spec.ts', 'src/public/**'],
            reportsDirectory: './coverage',
        },

        // Reporter for terminal output
        reporters: ['verbose'],

        // Timeout for async tests
        testTimeout: 10000,
    },
});
