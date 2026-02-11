# 《人工智能海洋学》课程网站技术规格文档

---

## 1. 组件清单

### shadcn/ui 组件
| 组件名 | 用途 | 定制需求 |
|--------|------|----------|
| Button | 按钮交互 | 深海蓝配色 |
| Card | 内容卡片 | 半透明背景 |
| Tabs | 资源标签切换 | 深色主题 |
| Accordion | 课程大纲展开 | 深色主题 |
| Badge | 难度标签 | 多色状态 |
| Separator | 分隔线 | 默认 |
| ScrollArea | 滚动区域 | 自定义滚动条 |

### 自定义组件
| 组件名 | 用途 | 复杂度 |
|--------|------|--------|
| ParticleBackground | 粒子背景效果 | 高 |
| Navbar | 导航栏 | 中 |
| HeroSection | 英雄区 | 高 |
| SyllabusTimeline | 课程大纲时间线 | 中 |
| LabCard | 编程练习卡片 | 低 |
| ResourceTabs | 资源标签页 | 中 |
| Footer | 页脚 | 低 |

---

## 2. 动画实现规划

| 动画效果 | 实现库 | 实现方式 | 复杂度 |
|----------|--------|----------|--------|
| 页面加载淡入 | Framer Motion | AnimatePresence + motion.div | 中 |
| 粒子背景 | Canvas API | 原生Canvas绘制 | 高 |
| 滚动触发显示 | Framer Motion | useInView + motion | 中 |
| 导航栏滚动变化 | React Hook | useScroll + CSS transition | 低 |
| 卡片Hover效果 | CSS/Tailwind | hover: 类 | 低 |
| 按钮Hover效果 | CSS/Tailwind | hover: 类 + transition | 低 |
| 时间线动画 | Framer Motion | staggerChildren | 中 |
| 标签切换动画 | Framer Motion | AnimatePresence | 中 |

---

## 3. 项目文件结构

```
app/
├── app/
│   ├── sections/
│   │   ├── Hero.tsx           # 英雄区
│   │   ├── About.tsx          # 课程介绍
│   │   ├── Syllabus.tsx       # 课程大纲
│   │   ├── Labs.tsx           # 编程练习
│   │   ├── Resources.tsx      # 课程资源
│   │   ├── Instructor.tsx     # 教师信息
│   │   └── Footer.tsx         # 页脚
│   ├── components/
│   │   ├── Navbar.tsx         # 导航栏
│   │   ├── ParticleBackground.tsx  # 粒子背景
│   │   ├── LabCard.tsx        # 练习卡片
│   │   ├── SyllabusItem.tsx   # 大纲项目
│   │   └── ScrollReveal.tsx   # 滚动显示包装器
│   ├── hooks/
│   │   └── useScrollPosition.ts  # 滚动位置hook
│   ├── page.tsx               # 主页面
│   ├── layout.tsx             # 根布局
│   └── globals.css            # 全局样式
├── components/ui/             # shadcn/ui 组件
├── public/
│   └── images/                # 图片资源
├── lib/
│   └── utils.ts               # 工具函数
├── next.config.js
├── tailwind.config.ts
└── package.json
```

---

## 4. 依赖清单

### 核心依赖
```json
{
  "dependencies": {
    "next": "^14",
    "react": "^18",
    "react-dom": "^18",
    "framer-motion": "^11",
    "lucide-react": "^0.400",
    "class-variance-authority": "^0.7",
    "clsx": "^2",
    "tailwind-merge": "^2"
  }
}
```

### 开发依赖
- TypeScript
- Tailwind CSS
- ESLint
- PostCSS

---

## 5. 关键技术实现

### 粒子背景
```typescript
// 使用Canvas实现
- 粒子数量: 60
- 粒子颜色: rgba(59, 130, 246, 0.5)
- 连接距离: 120px
- 连接颜色: rgba(59, 130, 246, 0.15)
- 移动速度: 0.3-0.8 px/frame
- 鼠标交互: 粒子轻微避让
```

### 滚动动画
```typescript
// Framer Motion useInView
- threshold: 0.1
- triggerOnce: true
- 动画: opacity 0→1, y 30→0
- 持续时间: 0.6s
- 缓动: easeOut
```

### 导航栏
```typescript
// 滚动检测
- 检测阈值: 50px
- 背景变化: transparent → rgba(2, 6, 23, 0.95)
- 过渡时间: 300ms
```

---

## 6. 性能优化

### 动画性能
- 使用 `transform` 和 `opacity` 进行动画
- 添加 `will-change: transform, opacity`
- 使用 `requestAnimationFrame` 进行Canvas动画

### 加载优化
- 图片懒加载
- 组件代码分割
- 字体预加载

### 可访问性
- 支持 `prefers-reduced-motion`
- 语义化HTML结构
- 键盘导航支持
