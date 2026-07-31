# AgentRecord — AI-Oceanography-Course

## 2026-07-31：排查点赞计数器不工作

### 现象
不同电脑显示固定点赞数，点击后不会真正增加（刷新后回到原值，且 24h 内无法再点）。

### 结论（已验证）
线上 JS 包中 Upstash 配置被编译为空字符串，远程写入从未生效，仅回退本机 `localStorage`。

### 相关历史
- 提交 `3c09045`：改用 Upstash Redis 直连（凭证经 `VITE_` 暴露在前端，且 Secrets 未配置）
- `api/like.mjs` 仍在仓库中，但前端一度不再调用

---

## 2026-07-31：方案 B — 恢复 Vercel Edge 代理

用户选择方案 B：前端只调自有 API，凭证留在服务端。

### 已做改动
- 重写 `src/lib/likesRemote.ts`：GET/POST 走 `VITE_LIKES_API_URL`，GET 失败时回退 GitHub raw
- `src/page.tsx`：远程失败时回滚乐观更新并 `clearLiked()`，避免卡死 24h
- 新增 `vercel.json`；`api/like.mjs` 改为 `export const config = { runtime: 'edge' }`
- CI 去掉 Upstash 环境变量；对 `public/likes.json` 做 `paths-ignore`
- 更新 `.env.example`、`README.md`

### 用户侧待完成（否则线上仍无法跨设备计数）
1. 在 Vercel 部署本仓库的 `api/like.mjs`，配置 `LIKES_GITHUB_PAT` / `LIKES_GH_OWNER` / `LIKES_GH_REPO`
2. 将接口完整 URL 写入 GitHub Secret `VITE_LIKES_API_URL`
3. 重新触发 GitHub Pages Deploy

### 2026-07-31：说明如何导入 Vercel
用户询问「Vercel 如何导入本仓库」。已将逐步操作写入 `README.md`「点赞计数器说明 → 部署 Vercel API」。
核心路径：vercel.com 用 GitHub 登录 → Add New Project → Import `scenty/AI-Oceanography-Course` → 配环境变量 → Deploy → 用 `/api/like` 作为 `VITE_LIKES_API_URL`。

### 2026-07-31：排查 https://ai-oceanography-course.vercel.app/api/like
用户反馈接口似乎未成功。本机探测结果：
- `*.vercel.app` DNS 被污染（解析到 Facebook/Twitter 等 IP）
- 强制走 Vercel anycast（76.76.21.21 等）时 TLS 仍被重置
- 结论：当前网络环境（及许多国内网络）**无法稳定访问 Vercel**，无法从这里验证函数业务逻辑是否正确
- 另：方案 B 的本地代码改动尚未 commit/push，线上 Vercel 仍部署旧版 `api/like.mjs`

若浏览器也打不开该 URL，问题在访问性而非（或不仅是）环境变量；课程站点受众在国内时，Vercel 作点赞后端可能不可行，需改 Cloudflare Workers 等。

后台补测（curl / DNS dig）已结束：直连超时、系统 DNS 解析到 Facebook IP（`31.13.86.21` 等），与上述结论一致。

### 2026-07-31：Vercel 运行时日志根因
用户提供的 Function Log：
1. `default export returned a Response` — 运行时按 Node `(req,res)=>void` 解析，返回的 Response 被忽略
2. 随后 `Vercel Runtime Timeout Error: Task timed out after 300 seconds`

已将 `api/like.mjs` 改为具名导出 `GET` / `POST` / `OPTIONS`（Web fetch 风格）。
**需把改动 push 到 GitHub 并让 Vercel 重新部署**后才会生效。
