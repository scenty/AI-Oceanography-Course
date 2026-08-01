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
│   └── like.mjs                   # ⚠️ 旧 Vercel 点赞函数（已停用，保留作参考）
├── cloudflare/
│   └── like-worker.js             # ⚠️ 旧 Cloudflare Worker（已停用，保留作参考）
├── edge-functions/
│   └── api/
│       └── like.js                # EdgeOne Pages Edge Function：点赞计数器（当前使用）
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
| 点赞后端 | 腾讯 EdgeOne Pages Edge Function |

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

前端只调用自有的 **腾讯 EdgeOne Pages Edge Function** 接口（`edge-functions/api/like.js`），GitHub PAT 等密钥留在服务端，不会打进前端包。

│   └── like-worker.js             # ⚠️ 旧 Cloudflare Worker（已停用，保留作参考）
> 前两者因国内网络无法稳定访问 `*.vercel.app` / `*.workers.dev`（DNS 污染）而弃用，代码保留作参考。
> 现网接口：`https://likes.aioceanography.top/api/like`

### 1. 创建 GitHub PAT（如已有可跳过）

GitHub → **Settings → Developer settings → Personal access tokens**：
- Classic token：勾选 `repo`（含 contents 读写）
- Fine-grained token：只授权 `AI-Oceanography-Course` 仓库，开 **Contents: Read and write**

### 2. 部署 EdgeOne Pages 函数

#### 导入仓库

1. 打开 [腾讯云 EdgeOne Pages 控制台](https://console.cloud.tencent.com/edgeone/pages)（需实名认证，免费额度足够）
2. **创建项目 → 关联 GitHub 仓库**，授权后选择 `scenty/AI-Oceanography-Course`
3. 平台会自动识别 `edge-functions/` 目录：`edge-functions/api/like.js` → 路由 `/api/like`

#### 配置环境变量

项目 → **项目设置 → 环境变量**：

| 变量 | 值 |
|------|-----|
| `LIKES_GITHUB_PAT` | 第 1 步申请的 GitHub PAT |
| `LIKES_GH_OWNER` | `scenty` |
| `LIKES_GH_REPO` | `AI-Oceanography-Course` |

可选：`LIKES_GH_BRANCH`（默认 `main`）、`LIKES_JSON_PATH`（默认 `public/likes.json`）

#### 绑定自定义域名（必须）

⚠️ EdgeOne 默认分配的 `*.edgeone.cool` 域名**只是 3 小时有效的预览链接**（控制台"预览"按钮续期），不能用于生产。必须绑定自有域名：

1. 准备一个域名（本项目用 `aioceanography.top`）
2. 首次添加域名时按提示做**归属权验证**：在域名解析商处添加指定的 TXT 记录
3. 项目加速区域设为 **全球可用区（不含中国大陆）**——此区域绑定自定义域名无需 ICP 备案
4. 添加自定义域名 `likes.aioceanography.top`，按提示在解析商处加 **CNAME** 记录指向 EdgeOne 目标值
5. 等待证书自动签发（几分钟到半小时）；签好前浏览器会报 HTTPS 隐私错误，属正常

#### 验证

```bash
curl https://likes.aioceanography.top/api/like
# 应返回 {"count":N,"ok":true}
curl -X POST https://likes.aioceanography.top/api/like
# count 应递增
```

排错：
- `{"ok":false,"reason":"not_configured"}`：环境变量没配或名字打错，改完需重新部署
- `502`：PAT 权限不足、过期，或 `LIKES_GH_OWNER` / `LIKES_GH_REPO` 写错
- `401 UNAUTHORIZED`：访问的是 `*.edgeone.cool` 预览域名且凭证过期，应改用自定义域名

### 3. 配置前端构建

在 GitHub 仓库 **Settings → Secrets and variables → Actions** 中，把 `VITE_LIKES_API_URL` 设为：

- `VITE_LIKES_API_URL` = `https://likes.aioceanography.top/api/like`（注意带 `/api/like`）

然后 push 到 `main` 或在 Actions 页面手动 **Re-run** 最近一次 Deploy 工作流。本地开发可复制 `.env.example` 为 `.env.local`。

> 点赞写入会更新 `public/likes.json`；工作流已对它 `paths-ignore`，避免每次点赞触发整站重部署。

### 4. 说明

- 前端有 24h 防重复点赞（localStorage `aio_liked_at`），清除该键可重置
│   └── like-worker.js             # ⚠️ 旧 Cloudflare Worker（已停用，保留作参考）
