import { defineConfig } from '@playwright/test'

// End-to-end test: starts the real server (API + built UI) and plays a hand in
// Chromium. Run `npm run build` first (or `make test-e2e`), inside the `mahjong`
// conda env.
export default defineConfig({
  testDir: 'tests',
  timeout: 240_000,
  use: { baseURL: 'http://127.0.0.1:8767', viewport: { width: 1400, height: 920 } },
  webServer: {
    command: 'mahjong serve --port 8767',
    url: 'http://127.0.0.1:8767/api/meta',
    reuseExistingServer: false,
    env: { MAHJONG_GAME_LOG_DIR: 'test-results/games' },
  },
})
