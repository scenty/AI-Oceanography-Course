import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // 如果使用 GitHub Pages 子路径，请将 '/AI-Oceanography-Course/' 改为 '/您的仓库名/'
  // 如果使用自定义域名，请改为 '/'
  base: mode === 'production' ? '/AI-Oceanography-Course/' : './',
  plugins: [inspectAttr(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
