import { defineConfig } from 'vitest/config'

// Unit tests only; browser tests in tests/ run with Playwright.
export default defineConfig({
  test: { include: ['src/**/*.test.ts'] },
})
