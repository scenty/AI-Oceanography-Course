# AGENTS.md — AI-Oceanography-Course

> 本文件面向 AI 编码助手。假设阅读者对该项目一无所知，所有信息均基于当前代码库的实际内容，不做推测。

---

## 1. 项目概述

本项目是 **《人工智能海洋学》** 课程的单页宣传与教学资源网站，授课教师为 **卢文芳**（中山大学海洋科学学院）。网站采用 React + TypeScript + Vite 构建，以深色海洋主题为视觉风格，部署在 **GitHub Pages** 上。

主要功能模块：
- **Hero 首屏**：全屏粒子动画背景 + 课程标题 + 点赞动效
- **课程介绍**：课程定位、特色与考核方式
- **课程内容**：10 章理论课（Tab 切换，部分章节折叠隐藏）
- **编程实践**：5 章实验课（可下载 `.py` / `.ipynb`，部分内嵌 JupyterLite）
- **对外教学**：AIO5 论坛、MARINE Summer School 等活动资料
- **授课教师**：教师名片与联系方式
- **页脚**：快速导航、外部链接、点赞数展示

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | React 19 + TypeScript ~5.9 |
| 构建工具 | Vite 7 |
| UI 组件 | shadcn/ui（new-york 风格，slate 基础色）+ Radix UI 底层 |
| 样式 | Tailwind CSS 3.4 + PostCSS + Autoprefixer |
| 动画 | Framer Motion + Canvas 原生粒子系统 |
| 图标 | lucide-react |
| 表单 | react-hook-form + zod |
| 图表 | recharts |
| 笔记本引擎 | JupyterLite（Pyodide 内核） |
| 部署 | GitHub Pages（GitHub Actions） |
| 点赞后端 | Vercel Edge Function（`api/like.mjs`） |

---

## 3. 目录结构

```
AI-Oceanography-Course/
├── .github/workflows/deploy.yml   # CI/CD：构建 + 部署到 GitHub Pages
├── api/
│   └── like.mjs                   # Vercel Edge Function：点赞计数器
├── public/                        # 静态资源（构建时会复制到 dist）
│   ├── images/                    # 课程配图（神经网络、CNN、Transformer 等）
│   ├── files/                     # 教学大纲、PDF 讲义
│   ├── external-teaching/         # 对外教学活动照片与大纲
│   ├── coding/                    # 学生编程练习（.py 文件，含 TODO 空待填写）
│   ├── notebook/                  # Jupyter 笔记本（学生版，部分空白）
│   └── likes.json                 # 点赞数数据源（GitHub API 直接读写）
├── src/
│   ├── main.tsx                   # 入口：创建 React Root，渲染 page.tsx
│   ├── page.tsx                   # 实际根组件：全局 likes 状态管理 + 所有 Section 组合
│   ├── index.css                  # 全局样式：Tailwind 指令、CSS 变量、深色主题、自定义动画
│   ├── App.tsx / App.css          # ⚠️ 废弃的 Vite 样板代码，未被引用
│   ├── components/
│   │   ├── Navbar.tsx             # 顶部固定导航栏（滚动变色 + 移动端汉堡菜单）
│   │   ├── ParticleBackground.tsx # Canvas 粒子背景（60 粒子，鼠标排斥）
│   │   ├── ScrollReveal.tsx       # Framer Motion 滚动进入动画包装器
│   │   └── ui/                    # 50+ shadcn/ui 基础组件（button、card、tabs、accordion 等）
│   ├── sections/                  # 页面区块组件（按垂直滚动顺序排列）
│   │   ├── Hero.tsx
│   │   ├── About.tsx
│   │   ├── CourseContent.tsx
│   │   ├── Labs.tsx
│   │   ├── ExternalTeaching.tsx
│   │   ├── Instructor.tsx
│   │   ├── Footer.tsx
│   │   ├── Syllabus.tsx           # ⚠️ 已存在但未被 page.tsx 引用
│   │   └── Resources.tsx          # ⚠️ 已存在但未被 page.tsx 引用
│   ├── data/
│   │   └── courseContent.ts       # 超大数据文件：10 章课程内容 + 实验代码模板
│   ├── hooks/
│   │   └── use-mobile.ts          # useIsMobile()（768px 断点）
│   └── lib/
│       ├── utils.ts               # cn() 工具 + getImagePath()（适配 GitHub Pages 子路径）
│       └── likesRemote.ts         # 点赞读写：localStorage + 可选 GitHub 远程同步
├── AI-Killing/                    # 独立 Python 研究脚本（与主站无集成）
│   ├── AIculting.py               # 智能斩杀线理论模拟（numpy + matplotlib）
│   ├── fig1_three_stage.png
│   ├── fig2_continuous_culling.png
│   ├── DEPLOY.md                  # 实际是主站的 GitHub Pages 部署说明（误放在此）
│   ├── TROUBLESHOOTING.md         # 部署故障排查（误放在此）
│   └── tech-spec.md               # 旧版技术规格（基于 Next.js，与当前 Vite 结构不符）
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── eslint.config.js
├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
├── components.json                # shadcn/ui 配置
├── requirements-jupyterlite.txt   # JupyterLite 构建依赖
└── package.json
```

---

## 4. 构建与开发命令

```bash
# 安装依赖
npm ci

# 本地开发（http://127.0.0.1:5173）
npm run dev

# 类型检查 + 生产构建（输出到 dist/）
npm run build

# 预览生产构建
npm run preview

# 代码检查
npm run lint
```

**JupyterLite 构建（仅在 CI 中自动执行）：**
```bash
pip install -r requirements-jupyterlite.txt
jupyter lite build --contents public/notebook --output-dir public/jupyterlite
```

---

## 5. 代码规范与约定

### 5.1 命名规范
- **组件文件**：PascalCase（`Navbar.tsx`、`CourseContent.tsx`）
- **Section 文件**：PascalCase，使用描述性名词（`Hero`、`About`、`ExternalTeaching`）
- **Hook 文件**：kebab-case（`use-mobile.ts`），导出 camelCase（`useIsMobile`）
- **UI 基础组件**：kebab-case 文件名（`accordion.tsx`），PascalCase 导出
- **数据常量**：camelCase（`courseChapters`、`labExercises`）
- **工具函数**：camelCase（`getImagePath`、`readLikesLocalStorage`）

### 5.2 路径别名
- `@/` 映射到 `./src`，在 `vite.config.ts` 和 `tsconfig.json` 中均已配置。
- 导入示例：`import { Navbar } from '@/components/Navbar'`

### 5.3 样式约定
- **全局主题**：深色模式，通过 `src/index.css` 中的 CSS 变量（HSL）定义。
- **Tailwind 扩展主题**：`tailwind.config.js` 中定义了 `colors`（primary、secondary、muted 等）、`borderRadius`、`fontFamily`（Inter / Noto Sans SC / Noto Serif SC）、自定义 `keyframes`（float、pulse-glow）。
- **常用类名模式**：
  - 卡片背景：`bg-[#0f172a]/80` 或 `bg-slate-900/50`
  - 边框：`border-slate-800`
  - 文字渐变：`text-gradient`（自定义 utility）
  - 玻璃态：`backdrop-blur-md`
- **shadcn/ui 组件**：统一放在 `src/components/ui/`，通过 `npx shadcn add <组件>` 安装。

### 5.4 TypeScript 严格配置
- `strict: true`
- `noUnusedLocals: true`
- `noUnusedParameters: true`
- `verbatimModuleSyntax: true`
- `noUncheckedSideEffectImports: true`

> 注意：由于 `noUnusedLocals` 和 `noUnusedParameters` 开启，任何未使用的变量或参数都会导致构建失败。

---

## 6. 架构要点

### 6.1 单页应用（SPA）
- 所有内容在一个长滚动页中，通过 `id` 锚点进行导航（`#hero`、`#about`、`#course-content` 等）。
- `page.tsx` 是唯一的页面组件，按顺序组装所有 Section。

### 6.2 数据组织
- **课程核心数据**：`src/data/courseContent.ts`（~1,475 行），包含 10 章理论内容 + 实验代码模板。
- **实验数据**：`src/sections/Labs.tsx` 内部内联定义。
- **对外教学数据**：`src/sections/ExternalTeaching.tsx` 内部内联定义。
- 部分章节标记了 `contentHidden: true`（第 4–10 章），只显示标题，内容折叠。

### 6.3 静态资源与 GitHub Pages 适配
- 生产环境基础路径为 `/AI-Oceanography-Course/`（`vite.config.ts` 中配置）。
- 所有图片/文件引用应通过 `getImagePath()`（`src/lib/utils.ts`）或 `import.meta.env.BASE_URL` 处理，避免子路径下 404。
- `public/coding/` 和 `public/notebook/` 下的文件直接在网页上提供下载链接。

### 6.4 点赞系统（Likes）
- **前端**：`page.tsx` 管理 `likes` 状态，`likesRemote.ts` 负责 localStorage 读写和可选的远程同步。
- **远程写入**：通过 `VITE_LIKES_API_URL` 指向的 POST 接口（Vercel Edge Function）。
- **远程读取**：直接从 GitHub raw 地址拉取 `public/likes.json`。
- **后端**：`api/like.mjs` 使用 GitHub Contents API 对 `public/likes.json` 进行乐观并发更新（最多重试 6 次）。

---

## 7. 环境变量

| 变量 | 来源 | 说明 |
|------|------|------|
| `VITE_GH_OWNER` | GitHub Actions (`github.repository_owner`) | GitHub 仓库所有者 |
| `VITE_GH_REPO` | GitHub Actions (`github.event.repository.name`) | 仓库名 |
| `VITE_GH_BRANCH` | GitHub Actions (`github.ref_name`) | 分支名（默认 main） |
| `VITE_LIKES_API_URL` | GitHub Secrets | 点赞 POST 接口地址 |

> 仅限以 `VITE_` 开头的变量才能在客户端代码中通过 `import.meta.env` 访问。

**后端 Edge Function 环境变量（部署在 Vercel）：**
- `LIKES_GITHUB_PAT`：具有 `contents:write` 权限的 GitHub Personal Access Token
- `LIKES_GH_OWNER`、`LIKES_GH_REPO`
- `LIKES_GH_BRANCH`（默认 `main`）
- `LIKES_JSON_PATH`（默认 `public/likes.json`）

---

## 8. 部署流程

### 8.1 GitHub Actions 工作流（`.github/workflows/deploy.yml`）

**触发条件：** `main` 分支的 `push` 或手动触发 (`workflow_dispatch`)

**Build 阶段：**
1. Checkout 代码
2. 安装 Node.js 20（缓存 npm）
3. 安装 Python 3.11
4. **构建 JupyterLite**：将 `public/notebook` 预置到 `public/jupyterlite`
5. `npm ci` 安装依赖
6. `npm run build` 构建 Vite（注入上述环境变量）
7. 上传 `dist/` 为 Pages artifact

**Deploy 阶段：**
- 使用 `actions/deploy-pages@v4` 部署到 GitHub Pages

### 8.2 部署前检查清单
- 仓库 **Settings > Pages** 的 Source 必须选择 **GitHub Actions**（不是 Branch）
- 仓库 **Settings > Actions > General > Workflow permissions** 需选择 **Read and write permissions**
- `vite.config.ts` 中的 `base` 路径需与仓库名一致（当前为 `/AI-Oceanography-Course/`）
- 若使用自定义域名，需将 `base` 改为 `'/'`

---

## 9. 测试策略

**当前项目未配置任何自动化测试框架。**（无 Jest、Vitest、Cypress、Playwright 等）

如需添加测试，建议：
- 单元测试：Vitest（与 Vite 同生态）
- E2E 测试：Playwright

---

## 10. 已知问题与遗留代码

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/App.tsx` | ❌ 废弃 | Vite 创建时的默认计数器样板，未被任何文件引用 |
| `src/App.css` | ❌ 废弃 | 同上，未被引用 |
| `src/sections/Syllabus.tsx` | ⚠️ 未使用 | 已开发完成但未被 `page.tsx` 导入渲染 |
| `src/sections/Resources.tsx` | ⚠️ 未使用 | 同上 |
| `AI-Killing/tech-spec.md` | ⚠️ 过期 | 描述的是 Next.js 项目结构，与当前 Vite 结构不符 |
| `AI-Killing/DEPLOY.md` | ⚠️ 错位 | 实际描述的是主站部署流程，不应放在 `AI-Killing/` 目录下 |

---

## 11. 安全注意事项

1. **GitHub PAT 保护**：`LIKES_GITHUB_PAT` 必须保存在 Vercel 环境变量或 GitHub Secrets 中，**切勿硬编码到仓库**。
2. **CORS 配置**：`api/like.mjs` 中 CORS 设置为 `*`，允许任何来源访问。如需限制，应修改为特定的域名白名单。
3. **公开仓库风险**：`public/likes.json` 作为数据存储，其修改历史完全公开在 Git 提交记录中。
4. **JupyterLite 安全**：JupyterLite 在浏览器端运行 Pyodide，学生代码在本地执行，不存在服务端代码注入风险，但需注意 Notebook 中是否包含恶意 JavaScript。

---

## 12. AI-Killing 目录说明

`AI-Killing/` 是一个**独立的 Python 研究脚本目录**，与主站前端代码无任何集成：
- `AIculting.py`：基于 numpy + matplotlib 的“智能斩杀线理论”数值模拟与可视化脚本。
- 运行后会生成 `fig1_three_stage.png` 和 `fig2_continuous_culling.png`。
- 该目录下的 `DEPLOY.md`、`TROUBLESHOOTING.md`、`tech-spec.md` 似乎是**误放或历史遗留**的主站文档，其内容描述的是主 React 应用，而非 Python 脚本本身。
- 修改主站代码时，**通常无需 touching 此目录**。
