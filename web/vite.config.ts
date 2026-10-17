import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the UI runs on Vite (port 5173) and forwards /api to the
// FastAPI server (`mahjong serve` or `uvicorn mahjong.server.app:app`, port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
