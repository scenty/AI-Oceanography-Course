# 人工智能海洋学课程网站

> 《人工智能海洋学》课程官方主页，由中山大学海洋科学学院 **卢文芳** 老师授课。
> 
> 在线访问：[https://scenty.github.io/AI-Oceanography-Course/](https://scenty.github.io/AI-Oceanography-Course/)

---

## 网站概述

本网站是一个基于 **React + TypeScript + Vite** 构建的单页应用（SPA），采用深色海洋主题视觉风格，用于展示课程信息、教学资源与编程实践内容。网站部署在 **GitHub Pages** 上，支持点赞互动与 JupyterLite 在线编程环境。

---

## 网站内容

网站采用长滚动单页设计，通过顶部导航栏的锚点快速定位到各模块：

### 1. Hero 首屏
- 全屏 Canvas 粒子动画背景
- 课程主标题与副标题
- 点赞动效与实时计数

### 2. 课程介绍（About）
- 课程定位与目标
- 教学特色与考核方式

### 3. 课程内容（Course Content）
- **10 章理论课程**，以 Tab 方式切换浏览
- 涵盖：人工智能概述、Python 基础、机器学习、神经网络、CNN、RNN、Transformer、强化学习等
- 部分章节（第 4–10 章）折叠隐藏，仅展示标题

### 4. 编程实践（Labs）
- **5 章实验课程**
- 提供 `.py` 源码与 `.ipynb` Jupyter 笔记本下载
- 部分实验内嵌 **JupyterLite** 在线环境，可直接在浏览器中运行 Python 代码

### 5. 对外教学（External Teaching）
- AIO5 论坛、MARINE Summer School 等活动资料与照片

### 6. 授课教师（Instructor）
- 教师名片、研究方向与联系方式

### 7. 页脚（Footer）
- 快速导航链接
- 外部资源链接
- 点赞数展示

---

## 项目结构

```
AI-Oceanography-Course/
├── .github/workflows/deploy.yml   # GitHub Actions CI/CD：自动构建并部署到 GitHub Pages
├── api/
│   └── like.mjs                   # ⚠️ 旧 Vercel Edge 点赞函数，已被 cloudflare/like-worker.js 取代（保留作参考）
├── cloudflare/
│   └── like-worker.js             # Cloudflare Worker：点赞计数器后端
├── public/                        # 静态资源（构建时复制到 dist）
│   ├── images/                    # 课程配图（神经网络、CNN、Transformer 等）
│   ├── files/                     # 教学大纲、PDF 讲义
│   ├── external-teaching/         # 对外教学活动照片与大纲
│   ├── coding/                    # 学生编程练习（.py 文件，含 TODO 空待填写）
│   ├── notebook/                  # Jupyter 笔记本（学生版，部分空白）
│   └── likes.json                 # 点赞数数据源
├── src/
│   ├── main.tsx                   # React 应用入口
│   ├── page.tsx                   # 根组件：组合所有 Section，管理全局点赞状态
│   ├── index.css                  # 全局样式：Tailwind 指令、CSS 变量、深色主题
│   ├── components/
│   │   ├── Navbar.tsx             # 顶部固定导航栏
│   │   ├── ParticleBackground.tsx # Canvas 粒子背景动画
│   │   ├── ScrollReveal.tsx       # 滚动进入动画包装器
│   │   └── ui/                    # shadcn/ui 基础组件（50+）
│   ├── sections/                  # 页面区块组件
│   │   ├── Hero.tsx
│   │   ├── About.tsx
│   │   ├── CourseContent.tsx
│   │   ├── Labs.tsx
│   │   ├── ExternalTeaching.tsx
│   │   ├── Instructor.tsx
│   │   ├── Footer.tsx
│   │   ├── Syllabus.tsx           # （已开发，暂未引用）
│   │   └── Resources.tsx          # （已开发，暂未引用）
│   ├── data/
│   │   └── courseContent.ts       # 课程核心数据：10 章理论内容 + 实验代码模板
│   ├── hooks/
│   │   └── use-mobile.ts          # 移动端检测 Hook
│   └── lib/
│       ├── utils.ts               # cn() 工具 + getImagePath() 路径适配
│       └── likesRemote.ts         # 点赞 localStorage + 远程同步逻辑
├── AI-Killing/                    # 独立 Python 研究脚本（与主站无集成）
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | React 19 + TypeScript ~5.9 |
| 构建工具 | Vite 7 |
| UI 组件 | shadcn/ui（new-york 风格）+ Radix UI |
| 样式 | Tailwind CSS 3.4 + PostCSS |
| 动画 | Framer Motion + Canvas 原生粒子系统 |
| 图标 | lucide-react |
| 图表 | recharts |
| 笔记本引擎 | JupyterLite（Pyodide 内核） |
| 部署 | GitHub Pages |
| 点赞后端 | Cloudflare Worker |

---

## 本地开发

```bash
# 安装依赖
npm ci

# 启动开发服务器（http://127.0.0.1:5173）
npm run dev

# 类型检查 + 生产构建（输出到 dist/）
npm run build

# 预览生产构建
npm run preview

# 代码检查
npm run lint
```

---

## 点赞计数器说明

前端只调用自有的 **Cloudflare Worker** 接口（`cloudflare/like-worker.js`），GitHub PAT 等密钥留在服务端，不会打进前端包。

> 历史说明：此前使用 Vercel Edge（`api/like.mjs`），因国内网络无法稳定访问 `*.vercel.app`（`ERR_CONNECTION_TIMED_OUT`）而迁移到 Cloudflare Workers。`api/like.mjs` 保留作参考，不再使用。

### 1. 创建 GitHub PAT（如已有可跳过）

GitHub → **Settings → Developer settings → Personal access tokens**：
- Classic token：勾选 `repo`（含 contents 读写）
- Fine-grained token：只授权 `AI-Oceanography-Course` 仓库，开 **Contents: Read and write**

### 2. 部署 Cloudflare Worker

#### 创建 Worker

1. 打开 [https://dash.cloudflare.com](https://dash.cloudflare.com) 并登录（没有账号就免费注册，不需要绑定域名、不需要信用卡）
2. 首次使用会要求设置 **workers.dev 子域**（例如 `yourname.workers.dev`），按提示确认即可
3. 左侧菜单 → **Workers & Pages** → 点 **Create**（或 **Create Application**）→ 选 **Create Worker**
4. 名称填 `aio-likes`（可自定义），直接点 **Deploy** —— 先用默认 Hello World 代码部署，不用管代码内容
5. 部署成功后点 **Edit Code**（或 Worker 页面的 **Edit code** 按钮）进入在线编辑器
6. 用仓库中 `cloudflare/like-worker.js` 的**全部内容**替换编辑器里的代码，点右上角 **Deploy**

#### 配置环境变量

Worker 页面 → **Settings** → **Variables and Secrets** → **Add**：

| 变量 | 类型 | 值 |
|------|------|-----|
| `LIKES_GITHUB_PAT` | **Secret** | 第 1 步申请的 GitHub PAT |
| `LIKES_GH_OWNER` | Text | `scenty` |
| `LIKES_GH_REPO` | Text | `AI-Oceanography-Course` |

可选：`LIKES_GH_BRANCH`（默认 `main`）、`LIKES_JSON_PATH`（默认 `public/likes.json`）

> 注意：`LIKES_GITHUB_PAT` 的类型务必选 **Secret**（加密存储）；添加后 Cloudflare 会自动重新部署使其生效。

#### 验证

浏览器直接打开 Worker 地址：

`https://aio-likes.<你的子域>.workers.dev`

应返回类似 `{"count":0,"ok":true}` 的 JSON。该完整 URL 即下一步要用的 `VITE_LIKES_API_URL`（**根路径即可，不要加 /api/like**）。

排错：
- `{"ok":false,"reason":"not_configured"}`：环境变量没配或名字打错
- `502`：PAT 权限不足、过期，或 `LIKES_GH_OWNER` / `LIKES_GH_REPO` 写错
- 想看运行日志：Worker 页面 → **Observability** / **Logs** 可开实时日志

### 3. 配置前端构建

在 GitHub 仓库 **Settings → Secrets and variables → Actions** 中，把已有的 `VITE_LIKES_API_URL` **修改为**：

- `VITE_LIKES_API_URL` = `https://aio-likes.<你的子域>.workers.dev`

然后 push 到 `main` 或在 Actions 页面手动 **Re-run** 最近一次 Deploy 工作流。本地开发可复制 `.env.example` 为 `.env.local`。

> 点赞写入会更新 `public/likes.json`；工作流已对它 `paths-ignore`，避免每次点赞触发整站重部署。

### 4. 收尾（可选）

- Vercel 侧：确认 Cloudflare 链路可用后，可在 Vercel Dashboard 删除原项目，避免残留 PAT 暴露面
- 本地自测 POST：`curl -X POST https://aio-likes.<你的子域>.workers.dev`，返回的 `count` 应递增（注意前端有 24h 防重复点赞，清 localStorage 的 `aio_liked_at` 可重置）
