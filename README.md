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
│   └── like.mjs                   # Vercel Edge Function：点赞计数器后端
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
| 点赞后端 | Vercel Edge Function |

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
