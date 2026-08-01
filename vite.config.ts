import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // 生产环境基础路径：
  // - 默认 '/AI-Oceanography-Course/'（GitHub Pages 子路径）
  // - 自定义域名部署（如 EdgeOne Pages 托管 www.aioceanography.top）时，
  //   在构建环境变量中设置 BASE_PATH=/ 即可从根路径访问
  base: mode === 'production' ? (process.env.BASE_PATH || '/AI-Oceanography-Course/') : '/',
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
  plugins: [inspectAttr(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
