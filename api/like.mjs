/**
 * Vercel Edge：通过 GitHub Contents API 读写仓库内 public/likes.json。
 * 环境变量：LIKES_GITHUB_PAT（classic PAT，需 contents:write）、LIKES_GH_OWNER、LIKES_GH_REPO；
 * 可选：LIKES_GH_BRANCH（默认 main）、LIKES_JSON_PATH（默认 public/likes.json）。
 *
 * 部署：将本仓库导入 Vercel（Root Directory 为仓库根），仅使用 /api/like；
 * 前端通过 VITE_LIKES_API_URL 调用，例如 https://xxx.vercel.app/api/like
 *
 * 注意：必须使用 Web 风格的具名导出（GET/POST/OPTIONS）。
 * default export 返回 Response 会被 Node 风格运行时忽略，导致请求挂起直到超时。
 */
export const config = { runtime: 'edge' };

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

function getEnv() {
  return {
    token: process.env.LIKES_GITHUB_PAT,
    owner: process.env.LIKES_GH_OWNER,
    repo: process.env.LIKES_GH_REPO,
    branch: process.env.LIKES_GH_BRANCH || 'main',
    path: process.env.LIKES_JSON_PATH || 'public/likes.json',
  };
}

function githubHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  };
}

async function getFile(env) {
  const contentsUrl = `https://api.github.com/repos/${env.owner}/${env.repo}/contents/${env.path}?ref=${encodeURIComponent(env.branch)}`;
  const r = await fetch(contentsUrl, { headers: githubHeaders(env.token) });
  const body = r.status === 404 ? null : await r.json().catch(() => null);
  return { r, body };
}

export function OPTIONS() {
  return new Response(null, { status: 204, headers: cors });
}

export async function GET() {
  const env = getEnv();
  if (!env.token || !env.owner || !env.repo) {
    return json({ count: 0, ok: false, reason: 'not_configured' });
  }

  const { r, body } = await getFile(env);
  if (r.status === 404) return json({ count: 0, ok: true });
  if (!r.ok) return json({ ok: false, status: r.status }, 502);
  const text = decodeGithubFileContent(body.content);
  const count = parseCount(text);
  return json({ count, ok: true });
}

export async function POST() {
  const env = getEnv();
  if (!env.token || !env.owner || !env.repo) {
    return json({ ok: false, reason: 'not_configured' }, 503);
  }

  const maxAttempts = 6;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const { r, body } = await getFile(env);
    if (!r.ok && r.status !== 404) {
      return json({ ok: false, status: r.status }, 502);
    }

    let nextCount;
    let sha;

    if (r.status === 404) {
      nextCount = 1;
      sha = undefined;
    } else {
      const text = decodeGithubFileContent(body.content);
      nextCount = parseCount(text) + 1;
      sha = body.sha;
    }

    const newContent = `${JSON.stringify({ count: nextCount })}\n`;
    const putBody = {
      message: `chore: site likes +1 (now ${nextCount})`,
      content: encodeGithubFileContent(newContent),
      branch: env.branch,
    };
    if (sha) putBody.sha = sha;

    const putRes = await fetch(
      `https://api.github.com/repos/${env.owner}/${env.repo}/contents/${env.path}`,
      {
        method: 'PUT',
        headers: { ...githubHeaders(env.token), 'Content-Type': 'application/json; charset=utf-8' },
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
