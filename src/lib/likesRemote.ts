/**
 * 点赞持久化到 GitHub 仓库中的 public/likes.json：
 * - 读取：优先走 Cloudflare Worker GET（VITE_LIKES_API_URL）；否则用公开 raw 地址。
 * - 写入：仅通过 VITE_LIKES_API_URL 指向的 POST（cloudflare/like-worker.js），
 *   PAT 等密钥只存在于 Cloudflare 服务端，不会打进前端包。
 *
 * GitHub Actions 需配置 Secret：VITE_LIKES_API_URL
 * Cloudflare Worker 需配置：LIKES_GITHUB_PAT、LIKES_GH_OWNER、LIKES_GH_REPO
 * （可选：LIKES_GH_BRANCH、LIKES_JSON_PATH）
 */

const LS_KEY = 'aio_likes';
const LS_LIKED_AT = 'aio_liked_at';

/** POST/GET 的完整 URL，例如 https://aio-likes.yourname.workers.dev */
export function getLikesApiUrl(): string | null {
  const u = import.meta.env.VITE_LIKES_API_URL?.trim();
  return u || null;
}

export function getLikesJsonRawUrl(): string | null {
  const owner = import.meta.env.VITE_GH_OWNER?.trim();
  const repo = import.meta.env.VITE_GH_REPO?.trim();
  const branch = import.meta.env.VITE_GH_BRANCH?.trim() || 'main';
  if (!owner || !repo) return null;
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/public/likes.json`;
}

function parseCount(value: unknown): number | null {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.floor(n));
}

async function fetchCountFromUrl(url: string, init?: RequestInit): Promise<number | null> {
  return fetch(url, init)
    .then(async (res) => {
      if (!res.ok) return null;
      const data = (await res.json()) as { count?: unknown };
      return parseCount(data.count);
    })
    .catch(() => null);
}

/** 读取远程点赞总数；失败时回退到 localStorage */
export async function fetchRemoteLikes(): Promise<number> {
  const apiUrl = getLikesApiUrl();
  if (apiUrl) {
    const fromApi = await fetchCountFromUrl(apiUrl, { method: 'GET' });
    if (fromApi !== null) {
      writeLikesLocalStorage(fromApi);
      return fromApi;
    }
  }

  const rawUrl = getLikesJsonRawUrl();
  if (rawUrl) {
    const fromRaw = await fetchCountFromUrl(`${rawUrl}?t=${Date.now()}`);
    if (fromRaw !== null) {
      writeLikesLocalStorage(fromRaw);
      return fromRaw;
    }
  }

  return readLikesLocalStorage();
}

/**
 * 远程点赞 +1。
 * - 已配置 API：POST 成功返回服务端总数，失败返回 null
 * - 未配置 API：仅更新 localStorage 并返回本地值（无法跨设备）
 */
export async function incrementRemoteLikes(): Promise<number | null> {
  const apiUrl = getLikesApiUrl();
  if (!apiUrl) {
    const next = readLikesLocalStorage() + 1;
    writeLikesLocalStorage(next);
    return next;
  }

  return fetch(apiUrl, { method: 'POST' })
    .then(async (res) => {
      if (!res.ok) return null;
      const data = (await res.json()) as { count?: unknown };
      const count = parseCount(data.count);
      if (count === null) return null;
      writeLikesLocalStorage(count);
      return count;
    })
    .catch(() => null);
}

export function readLikesLocalStorage(): number {
  const raw = window.localStorage.getItem(LS_KEY);
  const n = raw ? Number(raw) : 0;
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
}

export function writeLikesLocalStorage(n: number) {
  window.localStorage.setItem(LS_KEY, String(Math.max(0, Math.floor(n))));
}

/** 检查用户最近 24 小时内是否已经点过赞（前端防重复） */
export function hasLikedRecently(): boolean {
  const raw = window.localStorage.getItem(LS_LIKED_AT);
  if (!raw) return false;
  const likedAt = Number(raw);
  if (!Number.isFinite(likedAt)) return false;
  return Date.now() - likedAt < 24 * 60 * 60 * 1000;
}

export function markLiked() {
  window.localStorage.setItem(LS_LIKED_AT, String(Date.now()));
}

export function clearLiked() {
  window.localStorage.removeItem(LS_LIKED_AT);
}
