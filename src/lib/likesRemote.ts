/**
 * 点赞持久化：通过 Upstash Redis REST API 实现跨用户/跨设备的真正计数。
 *
 * 配置方式（本地开发）：
 *   在项目根目录创建 .env.local，写入：
 *   VITE_UPSTASH_REDIS_REST_URL=https://<id>.upstash.io
 *   VITE_UPSTASH_REDIS_REST_TOKEN=<token>
 *
 * 配置方式（GitHub Actions）：
 *   在仓库 Settings > Secrets and variables > Actions > Repository secrets 中添加：
 *   VITE_UPSTASH_REDIS_REST_URL
 *   VITE_UPSTASH_REDIS_REST_TOKEN
 *
 * 如果没有配置 Upstash，则回退到 localStorage（仅本机有效）。
 */

const LS_KEY = 'aio_likes';
const LS_LIKED_AT = 'aio_liked_at';

interface UpstashConfig {
  url: string;
  token: string;
}

export function getUpstashConfig(): UpstashConfig | null {
  const url = import.meta.env.VITE_UPSTASH_REDIS_REST_URL?.trim();
  const token = import.meta.env.VITE_UPSTASH_REDIS_REST_TOKEN?.trim();
  if (!url || !token) return null;
  return { url, token };
}

/** 从 Upstash Redis 读取当前点赞总数；失败时回退到 localStorage */
export async function fetchRemoteLikes(): Promise<number> {
  const cfg = getUpstashConfig();
  if (!cfg) return readLikesLocalStorage();

  try {
    const res = await fetch(`${cfg.url}/get/likes`, {
      headers: { Authorization: `Bearer ${cfg.token}` },
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = (await res.json()) as { result: string | number | null };
    const count = data.result == null ? 0 : Number(data.result);
    const safeCount = Number.isFinite(count) ? Math.max(0, Math.floor(count)) : 0;
    writeLikesLocalStorage(safeCount);
    return safeCount;
  } catch {
    return readLikesLocalStorage();
  }
}

/** 调用 Upstash Redis INCR 使点赞数 +1；返回新的总数，失败返回 null */
export async function incrementRemoteLikes(): Promise<number | null> {
  const cfg = getUpstashConfig();
  if (!cfg) return null;

  try {
    const res = await fetch(`${cfg.url}/incr/likes`, {
      headers: { Authorization: `Bearer ${cfg.token}` },
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = (await res.json()) as { result: string | number | null };
    const count = data.result == null ? 0 : Number(data.result);
    const safeCount = Number.isFinite(count) ? Math.max(0, Math.floor(count)) : 0;
    writeLikesLocalStorage(safeCount);
    return safeCount;
  } catch {
    return null;
  }
}

export function readLikesLocalStorage(): number {
  const raw = window.localStorage.getItem(LS_KEY);
  const n = raw ? Number(raw) : 0;
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
}

export function writeLikesLocalStorage(n: number) {
  window.localStorage.setItem(LS_KEY, String(Math.max(0, Math.floor(n))));
}

/** 检查用户最近 24 小时内是否已经点过赞（用于前端防重复） */
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
