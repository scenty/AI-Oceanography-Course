/**
 * 点赞持久化到 GitHub 仓库中的 public/likes.json：
 * - 读取：公开 raw 地址，无需 token（公开仓库）。
 * - 写入：需部署 api/like.mjs（如 Vercel），并配置 VITE_LIKES_API_URL 指向该 POST 接口；
 *   服务端使用 LIKES_GITHUB_PAT（contents:write）等变量，见 api/like.mjs 顶部注释。
 */

const LS_KEY = 'aio_likes';

export function getLikesJsonRawUrl(): string | null {
  const owner = import.meta.env.VITE_GH_OWNER?.trim();
  const repo = import.meta.env.VITE_GH_REPO?.trim();
  const branch = (import.meta.env.VITE_GH_BRANCH?.trim() || 'main');
  if (!owner || !repo) return null;
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/public/likes.json`;
}

/** POST 的完整 URL，例如 https://xxx.vercel.app/api/like */
export function getLikesApiPostUrl(): string | null {
  const u = import.meta.env.VITE_LIKES_API_URL?.trim();
  return u || null;
}

export function readLikesLocalStorage(): number {
  const raw = window.localStorage.getItem(LS_KEY);
  const n = raw ? Number(raw) : 0;
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
}

export function writeLikesLocalStorage(n: number) {
  window.localStorage.setItem(LS_KEY, String(Math.max(0, Math.floor(n))));
}
