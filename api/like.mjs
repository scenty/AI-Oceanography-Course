/**
 * Vercel Edge：通过 GitHub Contents API 读写仓库内 public/likes.json。
 * 环境变量：LIKES_GITHUB_PAT（classic PAT，需 contents:write）、LIKES_GH_OWNER、LIKES_GH_REPO；
 * 可选：LIKES_GH_BRANCH（默认 main）、LIKES_JSON_PATH（默认 public/likes.json）。
 */
export const runtime = 'edge';

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

export default async function handler(req) {
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: cors });
  }

  const token = process.env.LIKES_GITHUB_PAT;
  const owner = process.env.LIKES_GH_OWNER;
  const repo = process.env.LIKES_GH_REPO;
  const branch = process.env.LIKES_GH_BRANCH || 'main';
  const path = process.env.LIKES_JSON_PATH || 'public/likes.json';

  if (!token || !owner || !repo) {
    if (req.method === 'GET') return json({ count: 0, ok: false, reason: 'not_configured' });
    return json({ ok: false, reason: 'not_configured' }, 503);
  }

  const ghHeaders = {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  };

  const contentsUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${encodeURIComponent(branch)}`;

  async function getFile() {
    const r = await fetch(contentsUrl, { headers: ghHeaders });
    const body = r.status === 404 ? null : await r.json().catch(() => null);
    return { r, body };
  }

  if (req.method === 'GET') {
    const { r, body } = await getFile();
    if (r.status === 404) return json({ count: 0, ok: true });
    if (!r.ok) return json({ ok: false, status: r.status }, 502);
    const text = decodeGithubFileContent(body.content);
    const count = parseCount(text);
    return json({ count, ok: true });
  }

  if (req.method !== 'POST') {
    return json({ ok: false, reason: 'method_not_allowed' }, 405);
  }

  const maxAttempts = 6;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const { r, body } = await getFile();
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
      branch,
    };
    if (sha) putBody.sha = sha;

    const putRes = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/contents/${path}`,
      {
        method: 'PUT',
        headers: { ...ghHeaders, 'Content-Type': 'application/json; charset=utf-8' },
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
