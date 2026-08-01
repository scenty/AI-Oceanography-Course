/**
 * EdgeOne Pages Edge Function：通过 GitHub Contents API 读写仓库内 public/likes.json。
 * 路由：/api/like（由本文件路径 edge-functions/api/like.js 自动生成）
 * 与 cloudflare/like-worker.js 逻辑等价，前端无需改动，仅替换 VITE_LIKES_API_URL。
 *
 * 环境变量（EdgeOne Pages 项目设置 → 环境变量）：
 *   LIKES_GITHUB_PAT（classic PAT，需 contents:write 权限）
 *   LIKES_GH_OWNER、LIKES_GH_REPO
 *   可选：LIKES_GH_BRANCH（默认 main）、LIKES_JSON_PATH（默认 public/likes.json）
 *
 * 前端调用地址：https://<项目名>.edgeone.app/api/like
 */

const cors = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json; charset=utf-8' },
  });
}

function decodeGithubFileContent(content) {
  const b64 = String(content).replace(/\s/g, '');
  return atob(b64);
}

function parseCount(text) {
  const o = JSON.parse(text);
  const n = Number(o.count);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
}

function encodeGithubFileContent(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function githubHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'aio-likes-function',
  };
}

async function getFile(env) {
  const branch = env.LIKES_GH_BRANCH || 'main';
  const path = env.LIKES_JSON_PATH || 'public/likes.json';
  const url = `https://api.github.com/repos/${env.LIKES_GH_OWNER}/${env.LIKES_GH_REPO}/contents/${path}?ref=${encodeURIComponent(branch)}`;
  const r = await fetch(url, { headers: githubHeaders(env.LIKES_GITHUB_PAT) });
  const body = r.status === 404 ? null : await r.json().catch(() => null);
  return { r, body, branch, path };
}

async function handleGet(env) {
  if (!env.LIKES_GITHUB_PAT || !env.LIKES_GH_OWNER || !env.LIKES_GH_REPO) {
    return json({ count: 0, ok: false, reason: 'not_configured' });
  }

  const { r, body } = await getFile(env);
  if (r.status === 404) return json({ count: 0, ok: true });
  if (!r.ok) return json({ ok: false, status: r.status }, 502);
  const count = parseCount(decodeGithubFileContent(body.content));
  return json({ count, ok: true });
}

async function handlePost(env) {
  if (!env.LIKES_GITHUB_PAT || !env.LIKES_GH_OWNER || !env.LIKES_GH_REPO) {
    return json({ ok: false, reason: 'not_configured' }, 503);
  }

  const maxAttempts = 6;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const { r, body, branch, path } = await getFile(env);
    if (!r.ok && r.status !== 404) {
      return json({ ok: false, status: r.status }, 502);
    }

    let nextCount;
    let sha;

    if (r.status === 404) {
      nextCount = 1;
      sha = undefined;
    } else {
      nextCount = parseCount(decodeGithubFileContent(body.content)) + 1;
      sha = body.sha;
    }

    const putBody = {
      message: `chore: site likes +1 (now ${nextCount})`,
      content: encodeGithubFileContent(`${JSON.stringify({ count: nextCount })}\n`),
      branch,
    };
    if (sha) putBody.sha = sha;

    const putRes = await fetch(
      `https://api.github.com/repos/${env.LIKES_GH_OWNER}/${env.LIKES_GH_REPO}/contents/${path}`,
      {
        method: 'PUT',
        headers: { ...githubHeaders(env.LIKES_GITHUB_PAT), 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify(putBody),
      },
    );

    if (putRes.ok) {
      return json({ count: nextCount, ok: true });
    }

    if (putRes.status === 409) {
      continue;
    }

    const errText = await putRes.text();
    return json({ ok: false, status: putRes.status, detail: errText.slice(0, 200) }, 502);
  }

  return json({ ok: false, reason: 'conflict_retry_exhausted' }, 409);
}

export default async function onRequest(context) {
  const { request, env } = context;
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: cors });
  }
  if (request.method === 'GET') return handleGet(env);
  if (request.method === 'POST') return handlePost(env);
  return json({ ok: false, reason: 'method_not_allowed' }, 405);
}
