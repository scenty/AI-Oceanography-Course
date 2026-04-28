import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  FileCode, 
  Calculator, 
  Brain,
  Database,
  Waves,
  Cpu,
  Play,
  Download,
  ExternalLink
} from 'lucide-react';

const difficultyLabels = {
  beginner: { label: '入门', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
  intermediate: { label: '进阶', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  advanced: { label: '高级', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
};

type LabDifficulty = keyof typeof difficultyLabels;
type LabMaterial = { label: string; href: string; kind: 'py' | 'ipynb' | 'other' };

type LabItem = {
  id: string;
  title: string;
  description: string;
  difficulty: LabDifficulty;
  topics: string[];
  notebook?: string;
  materials: LabMaterial[];
  note?: string;
};

type Chapter = {
  id: string;
  title: string;
  icon: typeof Calculator;
  summary: string;
  items: LabItem[];
};

const chapters: Chapter[] = [
  {
    id: 'Coding1',
    title: 'Coding1 - 基础',
    icon: Calculator,
    summary: 'Python / NumPy / Matplotlib 基础与向量化思维（课前热身）',
    items: [
      {
        id: 'Code1.0',
        title: '基本操作（50 Quizzes）',
        description: '复习 Python 与 NumPy/Matplotlib 核心操作，为后续练习打基础。',
        difficulty: 'beginner',
        topics: ['Python', 'NumPy', 'Matplotlib', '向量化'],
        notebook: 'Student_Notebook.ipynb',
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding1/Code1.0 - 基本操作.py', kind: 'py' },
          { label: '练习 Notebook（.ipynb）', href: '/notebook/Student_Notebook.ipynb', kind: 'ipynb' },
        ],
        note: '在线运行：先下载 Notebook，再在下方 JupyterLite 里 File → Open 打开。',
      },
      {
        id: 'Code1.1',
        title: 'NumPy 基础练习（Q）',
        description: '数组创建/索引/切片/广播等基础练习（素材待同步）。',
        difficulty: 'beginner',
        topics: ['NumPy', '索引切片', '广播'],
        materials: [],
        note: '该文件当前未成功同步到仓库（SynologyDrive 占位符/网络导致读取失败）。',
      },
      {
        id: 'Code1.2',
        title: 'NumPy 进阶练习（Q）',
        description: '更高阶的 NumPy 操作与向量化练习（素材待同步）。',
        difficulty: 'intermediate',
        topics: ['NumPy', '向量化', '性能'],
        materials: [],
        note: '该文件当前未成功同步到仓库（SynologyDrive 占位符/网络导致读取失败）。',
      },
    ],
  },
  {
    id: 'Coding2',
    title: 'Coding2 - 数据下载和分析',
    icon: Waves,
    summary: 'ERA5 数据下载、读取与基础处理（xarray / netCDF）',
    items: [
      {
        id: 'Code2.0',
        title: '下载数据及处理',
        description: '读取风/浪数据并完成基础处理（包含 TODO）。',
        difficulty: 'intermediate',
        topics: ['xarray', 'netCDF', '可视化'],
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding2/Code2. 下载数据及处理.py', kind: 'py' },
        ],
      },
      {
        id: 'get_data',
        title: 'get_data（CDS API 下载脚本）',
        description: '通过 CDS API 下载 ERA5 月平均数据（需配置 cdsapi key）。',
        difficulty: 'intermediate',
        topics: ['cdsapi', '数据下载', 'ERA5'],
        materials: [{ label: '脚本（.py）', href: '/coding/coding2/get_data.py', kind: 'py' }],
      },
    ],
  },
  {
    id: 'Coding3',
    title: 'Coding3 - 简单的线性模型',
    icon: Brain,
    summary: '从格点搜索到梯度下降：线性模型优化入门',
    items: [
      {
        id: 'Code3.1-Q1',
        title: 'Q1 GridSearch',
        description: '用格点搜索寻找最优参数（含可视化）。',
        difficulty: 'intermediate',
        topics: ['线性回归', '格点搜索', '代价函数'],
        notebook: 'L3_1_LinearModel.ipynb',
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding3/Code3.1 - 线性模型及其优化 - Q1 - GridSearch.py', kind: 'py' },
          { label: '练习 Notebook（.ipynb）', href: '/notebook/L3_1_LinearModel.ipynb', kind: 'ipynb' },
        ],
        note: 'Notebook 版本包含更适合在线运行的分步结构；.py 版本用于本地/Spyder。',
      },
      {
        id: 'Code3.1-Q2',
        title: 'Q2 GradientDescent',
        description: '用梯度下降最小化损失函数（含可视化）。',
        difficulty: 'intermediate',
        topics: ['梯度下降', '优化', '可视化'],
        notebook: 'L3_1_LinearModel.ipynb',
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding3/Code3.1 - 线性模型及其优化 - Q2 - GradientDescent.py', kind: 'py' },
          { label: '练习 Notebook（.ipynb）', href: '/notebook/L3_1_LinearModel.ipynb', kind: 'ipynb' },
        ],
      },
    ],
  },
  {
    id: 'Coding4',
    title: 'Coding4 - 手搓NN',
    icon: Cpu,
    summary: '手写前向/反向传播，理解 NN 的核心计算图',
    items: [
      {
        id: 'Code4.1-Q',
        title: '手搓 NN 模型及其优化（Q）',
        description: '补全激活函数、前向传播、反向传播并训练。',
        difficulty: 'advanced',
        topics: ['反向传播', '梯度', '训练循环'],
        notebook: 'L4_1_ManualNN.ipynb',
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding4/Code4.1 - 手搓NN模型及其优化 - Q.py', kind: 'py' },
          { label: '练习 Notebook（.ipynb）', href: '/notebook/L4_1_ManualNN.ipynb', kind: 'ipynb' },
        ],
      },
      {
        id: 'L4.2',
        title: '向量化 NN（Notebook）',
        description: '批量前向/反向传播向量化，实现更高效训练。',
        difficulty: 'advanced',
        topics: ['向量化', '批量梯度', '性能优化'],
        notebook: 'L4_2_VectorizedNN.ipynb',
        materials: [
          { label: '练习 Notebook（.ipynb）', href: '/notebook/L4_2_VectorizedNN.ipynb', kind: 'ipynb' },
        ],
      },
    ],
  },
  {
    id: 'Coding5',
    title: 'Coding5 - Pytorch',
    icon: Database,
    summary: 'PyTorch 张量、DataLoader 与训练流程',
    items: [
      {
        id: 'Code5.1-Q',
        title: 'PyTorch 基础练习（Q）',
        description: '张量操作、基础训练循环（含 TODO）。',
        difficulty: 'intermediate',
        topics: ['Tensor', 'autograd', '基础训练'],
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding5/Code5.1 - PyTorch基础练习 - Q.py', kind: 'py' },
        ],
      },
      {
        id: 'Code5.2-Q',
        title: 'PyTorch 进阶练习（Q）',
        description: '更完整的模型训练与工程化要点（含 TODO）。',
        difficulty: 'intermediate',
        topics: ['GPU', '保存加载', '训练技巧'],
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding5/Code5.2 - PyTorch进阶练习 - Q.py', kind: 'py' },
        ],
      },
      {
        id: 'Code5.3',
        title: '手搓 NN（PyTorch 版）',
        description: '将手搓 NN 迁移到 PyTorch 版本训练流程。',
        difficulty: 'intermediate',
        topics: ['nn.Module', '优化器', '训练循环'],
        materials: [
          { label: '练习代码（.py）', href: '/coding/coding5/Code5.3 - 手搓NN模型及其优化 - PyTorch版.py', kind: 'py' },
        ],
      },
      {
        id: 'Code5.4',
        title: 'show_dataloader',
        description: 'DataLoader 使用演示与数据批处理流程。',
        difficulty: 'beginner',
        topics: ['Dataset', 'DataLoader'],
        materials: [
          { label: '示例代码（.py）', href: '/coding/coding5/Code5.4 - PyTorch_show_dataloader.py', kind: 'py' },
        ],
      },
    ],
  },
];

export function Labs() {
  const [expandedLab, setExpandedLab] = useState<string | null>(null);
  const [activeChapter, setActiveChapter] = useState<string>(chapters[0].id);

  const itemIndex = useMemo(() => {
    const m = new Map<string, LabItem>();
    for (const ch of chapters) {
      for (const item of ch.items) m.set(item.id, item);
    }
    return m;
  }, []);

  const activeItem = expandedLab ? itemIndex.get(expandedLab) ?? null : null;

  return (
    <section id="labs" className="relative py-24 bg-[#020617]">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_70%,rgba(6,182,212,0.08),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full">
              编程练习
            </span>
          </ScrollReveal>
          
          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              动手实践，深入理解
            </h2>
          </ScrollReveal>
          
          <ScrollReveal delay={0.2}>
            <p className="text-slate-400 max-w-2xl mx-auto">
              从基础练习到深度学习应用，通过编程练习逐步掌握人工智能海洋学的核心技能
            </p>
          </ScrollReveal>
        </div>

        {/* Category Tabs */}
        <ScrollReveal delay={0.3}>
          <Tabs value={activeChapter} onValueChange={setActiveChapter} className="w-full">
            <TabsList className="flex flex-wrap justify-center gap-2 bg-transparent mb-8 h-auto">
              {chapters.map((chapter) => (
                <TabsTrigger
                  key={chapter.id}
                  value={chapter.id}
                  className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50 border border-slate-800 bg-[#0f172a]/80 text-slate-400 px-4 py-2 rounded-lg transition-all flex items-center gap-2"
                >
                  <chapter.icon className="w-4 h-4" />
                  <span className="text-sm">{chapter.title}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {chapters.map((chapter) => {
              return (
                <TabsContent key={chapter.id} value={chapter.id} className="mt-0">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                  >
                    <div className="mb-6 text-center">
                      <p className="text-slate-400 max-w-3xl mx-auto">{chapter.summary}</p>
                    </div>
                    <div className="grid md:grid-cols-2 gap-4">
                      {chapter.items.map((lab, index) => (
                        <motion.div
                          key={lab.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.4, delay: index * 0.05 }}
                        >
                          <Card 
                            className={`bg-[#0f172a]/80 border-slate-800 hover:border-cyan-500/50 transition-all duration-300 cursor-pointer ${
                              expandedLab === lab.id ? 'border-cyan-500/50' : ''
                            }`}
                            onClick={() => setExpandedLab(expandedLab === lab.id ? null : lab.id)}
                          >
                            <CardContent className="p-5">
                              {/* Header */}
                              <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-3">
                                  <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                                    <FileCode className="w-5 h-5 text-cyan-400" />
                                  </div>
                                  <div>
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs font-mono text-slate-500">{lab.id}</span>
                                      <h4 className="text-white font-medium">{lab.title}</h4>
                                    </div>
                                    <p className="text-slate-400 text-sm">{lab.description}</p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <Badge 
                                    variant="outline"
                                    className={difficultyLabels[lab.difficulty].color}
                                  >
                                    {difficultyLabels[lab.difficulty].label}
                                  </Badge>
                                </div>
                              </div>

                              {/* Topics */}
                              <div className="flex flex-wrap gap-1.5 mb-3">
                                {lab.topics.map((topic) => (
                                  <span
                                    key={topic}
                                    className="px-2 py-0.5 text-xs text-slate-500 bg-slate-800/80 rounded"
                                  >
                                    {topic}
                                  </span>
                                ))}
                              </div>

                              {/* Expanded Content */}
                              <AnimatePresence>
                                {expandedLab === lab.id && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.3 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="pt-4 border-t border-slate-800">
                                      {lab.note && (
                                        <p className="text-slate-500 text-sm mb-3">{lab.note}</p>
                                      )}

                                      {lab.materials.length === 0 ? (
                                        <div className="p-4 rounded-lg border border-dashed border-slate-700 text-slate-400 text-sm bg-slate-900/30">
                                          素材待同步（代码 / Notebook）
                                        </div>
                                      ) : (
                                        <div className="space-y-2">
                                          {lab.materials.map((m) => (
                                            <a
                                              key={m.href}
                                              href={`${import.meta.env.BASE_URL}${m.href.startsWith('/') ? m.href.slice(1) : m.href}`}
                                              download={m.kind === 'ipynb' ? (lab.notebook ?? undefined) : undefined}
                                              className="flex items-center justify-between gap-3 px-4 py-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors no-underline"
                                              onClick={(e) => e.stopPropagation()}
                                            >
                                              <span className="text-slate-200 text-sm">{m.label}</span>
                                              <span className="text-slate-500 text-xs flex items-center gap-2">
                                                {m.kind === 'ipynb' ? 'Notebook' : m.kind === 'py' ? 'Python' : '文件'}
                                                <Download className="w-4 h-4" />
                                              </span>
                                            </a>
                                          ))}
                                        </div>
                                      )}

                                      {lab.notebook && (
                                        <p className="mt-3 text-xs text-cyan-400/90">
                                          下方已嵌入 JupyterLite（官方在线环境），建议先下载本练习的 Notebook，再在 JupyterLite 中
                                          通过 File → Open 打开运行。
                                        </p>
                                      )}

                                      <div className="mt-4 flex flex-wrap gap-2">
                                        {lab.notebook && (
                                          <a
                                            href={`${import.meta.env.BASE_URL}notebook/${lab.notebook}`}
                                            download={lab.notebook}
                                            onClick={(e) => e.stopPropagation()}
                                            className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-300 rounded-lg text-sm hover:bg-slate-700 transition-colors no-underline"
                                          >
                                            <FileCode className="w-4 h-4" />
                                            下载 Notebook
                                          </a>
                                        )}
                                        <a
                                          href="https://jupyterlite.github.io/demo/lab/index.html?theme=JupyterLab%20Dark"
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          onClick={(e) => e.stopPropagation()}
                                          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg text-sm hover:bg-cyan-500/30 transition-colors no-underline"
                                        >
                                          <Play className="w-4 h-4" />
                                          打开在线 JupyterLite
                                          <ExternalLink className="w-4 h-4" />
                                        </a>
                                      </div>
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </CardContent>
                          </Card>
                        </motion.div>
                      ))}
                    </div>

                    {/* 支持 Notebook 的练习展开时，全宽嵌入 JupyterLite */}
                    {activeItem?.notebook && (
                      <motion.div
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-8 w-full space-y-3"
                      >
                        <div className="flex items-center gap-2 text-sm text-cyan-400">
                          <Play className="w-4 h-4 flex-shrink-0" />
                          <span>在线运行（JupyterLite 官方 Demo，可在浏览器内执行 Python / Notebook）</span>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-slate-700 bg-slate-900/50">
                          <iframe
                            title="JupyterLite 在线 Notebook"
                            src="https://jupyterlite.github.io/demo/lab/index.html?theme=JupyterLab%20Dark"
                            className="w-full border-0"
                            style={{ height: 'min(72vh, 680px)' }}
                            sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"
                            allowFullScreen
                          />
                        </div>
                        <p className="text-xs text-slate-500">
                          首次加载会下载 Python 运行环境（约数十秒）。请先下载本练习 Notebook，然后在左侧菜单 File → Open 打开运行。
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                </TabsContent>
              );
            })}
          </Tabs>
        </ScrollReveal>
      </div>
    </section>
  );
}
