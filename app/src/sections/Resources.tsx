// Resources section with tabs for different resource types
import { motion, AnimatePresence } from 'framer-motion';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  FileText, 
  Code, 
  BookOpen, 
  ExternalLink,
  Download,
  Database,
  Globe
} from 'lucide-react';

const lectures = [
  { id: 1, title: '人工智能概述', file: 'Lect.1 人工智能海洋学课程课件-人工智能概述.pdf' },
  { id: 2, title: '海洋大数据简介', file: 'Lect.2 人工智能海洋学课程课件-海洋大数据简介.pdf' },
  { id: 3, title: '神经网络基础', file: 'Lect.3 人工智能海洋学课程课件-神经网络基础.pdf' },
  { id: 4, title: 'PyTorch基础', file: 'Lect.4 人工智能海洋学课程课件-PyTorch.pdf' },
  { id: 5, title: '深度学习', file: 'Lect.5 人工智能海洋学课程课件-深度学习.pdf' },
  { id: 6, title: '海洋回归问题 - 重建', file: 'Lect.6 人工智能海洋学课程课件-海洋回归问题 重建问题.pdf' },
  { id: 7, title: '海洋时序问题', file: 'Lect.7 人工智能海洋学课程课件-海洋回归问题 时间序列问题.pdf' },
  { id: 8, title: '海洋预测 - 注意力机制', file: 'Lect.8 人工智能海洋学课程课件-海洋回归问题 注意力机制.pdf' },
  { id: 9, title: '海洋识别问题', file: 'Lect.9 人工智能海洋学课程课件-海洋识别问题.pdf' },
  { id: 11, title: '物理约束神经网络', file: 'Lect.11 人工智能海洋学课程课件-物理约束.pdf' },
];

const codeExamples = [
  { id: 'L1.0', title: '基本操作', file: 'Lect1.0 - 基本操作.py' },
  { id: 'L1.1', title: 'NumPy基础练习', file: 'Lect1.1 - numpy基础练习.py' },
  { id: 'L1.2', title: '其他工具基础', file: 'Lect1.2 - 其他工具基础.py' },
  { id: 'L1.3', title: 'PyTorch基础', file: 'Lect1.3 - Pytorch基础.py' },
  { id: 'L1.4', title: 'NumPy进阶练习', file: 'Lect1.4 - numpy进阶练习.py' },
  { id: 'L2.1', title: 'NumPy地转差分', file: 'Lec2.1  - numpy地转差分.py' },
  { id: 'L3.1', title: '线性模型及其优化', file: 'Lect3.1 - 线性模型及其优化.py' },
  { id: 'L4.1', title: '手搓NN模型', file: 'Lect4.1 - 手搓NN模型及其优化 - 本.py' },
  { id: 'L4.2', title: '手搓NN模型-向量化', file: 'Lect4.2 - 手搓NN模型及其优化 - vectorize.py' },
  { id: 'L5.1', title: 'PyTorch基础练习', file: 'Lect5.1 - Pytorch基础练习.py' },
  { id: 'LeNet', title: 'LeNet手写字体识别', file: 'LeNet.py' },
];

const references = [
  {
    title: '教材与专著',
    items: [
      { name: '《人工智能海洋学》', author: '董昌明等', type: '教材' },
      { name: 'Artificial Intelligence Oceanography', author: 'Xiaofeng Li, Fan Wang等', type: '专著' },
    ],
  },
  {
    title: '在线资源',
    items: [
      { name: 'Deep Learning with PyTorch', url: 'https://isip.piconepress.com/courses/temple/ece_4822/resources/books/Deep-Learning-with-PyTorch.pdf', type: '电子书' },
      { name: '深入浅出PyTorch', url: 'https://datawhalechina.github.io/thorough-pytorch/', type: '中文教程' },
      { name: 'PyTorch官方文档', url: 'https://pytorch.org/docs/', type: '文档' },
    ],
  },
];

const dataPlatforms = [
  { name: 'Copernicus Marine Service', url: 'https://data.marine.copernicus.eu/', description: '欧盟哥白尼海洋服务' },
  { name: 'NDBC', url: 'https://www.ndbc.noaa.gov/', description: '美国国家数据浮标中心' },
  { name: 'JMA', url: 'https://www.data.jma.go.jp/', description: '日本气象厅' },
  { name: '国家海洋科学数据中心', url: 'https://mds.nmdis.org.cn/', description: '中国海洋数据中心' },
];

export function Resources() {
  return (
    <section id="resources" className="relative py-24 bg-[#020617]">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.05),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-blue-400 bg-blue-400/10 border border-blue-400/20 rounded-full">
              课程资源
            </span>
          </ScrollReveal>
          
          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              丰富的学习资源
            </h2>
          </ScrollReveal>
          
          <ScrollReveal delay={0.2}>
            <p className="text-slate-400 max-w-2xl mx-auto">
              课件资料、代码示例、参考教材和数据平台，助力你的学习之旅
            </p>
          </ScrollReveal>
        </div>

        {/* Tabs */}
        <ScrollReveal delay={0.3}>
          <Tabs defaultValue="lectures" className="w-full">
            <TabsList className="grid w-full max-w-2xl mx-auto grid-cols-4 bg-[#0f172a] border border-slate-800 mb-8">
              <TabsTrigger value="lectures" className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400">
                <FileText className="w-4 h-4 mr-2" />
                课件
              </TabsTrigger>
              <TabsTrigger value="code" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
                <Code className="w-4 h-4 mr-2" />
                代码
              </TabsTrigger>
              <TabsTrigger value="references" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
                <BookOpen className="w-4 h-4 mr-2" />
                参考资料
              </TabsTrigger>
              <TabsTrigger value="platforms" className="data-[state=active]:bg-green-500/20 data-[state=active]:text-green-400">
                <Database className="w-4 h-4 mr-2" />
                数据平台
              </TabsTrigger>
            </TabsList>

            <AnimatePresence mode="wait">
              <TabsContent value="lectures" className="mt-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-[#0f172a]/80 border-slate-800">
                    <CardContent className="p-6">
                      <div className="grid sm:grid-cols-2 gap-4">
                        {lectures.map((lecture) => (
                          <div
                            key={lecture.id}
                            className="flex items-center gap-4 p-4 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors group"
                          >
                            <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                              <span className="text-sm font-mono text-blue-400">{lecture.id}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-white font-medium truncate">{lecture.title}</h4>
                              <p className="text-slate-500 text-sm truncate">{lecture.file}</p>
                            </div>
                            <button className="p-2 rounded-lg bg-blue-500/10 text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-blue-500/20">
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </TabsContent>

              <TabsContent value="code" className="mt-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-[#0f172a]/80 border-slate-800">
                    <CardContent className="p-6">
                      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {codeExamples.map((code) => (
                          <div
                            key={code.id}
                            className="flex items-center gap-3 p-4 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors group"
                          >
                            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
                              <Code className="w-4 h-4 text-cyan-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-white font-medium text-sm truncate">{code.title}</h4>
                              <p className="text-slate-500 text-xs truncate">{code.file}</p>
                            </div>
                            <button className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-cyan-500/20">
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </TabsContent>

              <TabsContent value="references" className="mt-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="grid md:grid-cols-2 gap-6">
                    {references.map((section) => (
                      <Card key={section.title} className="bg-[#0f172a]/80 border-slate-800">
                        <CardContent className="p-6">
                          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                            <BookOpen className="w-5 h-5 text-purple-400" />
                            {section.title}
                          </h3>
                          <div className="space-y-3">
                            {section.items.map((item) => (
                              <div
                                key={item.name}
                                className="flex items-start justify-between gap-4 p-3 rounded-lg bg-slate-800/50"
                              >
                                <div>
                                  <h4 className="text-white text-sm">{item.name}</h4>
                                  {'author' in item && (
                                    <p className="text-slate-500 text-xs">{item.author}</p>
                                  )}
                                  <span className="inline-block mt-1 px-2 py-0.5 text-xs text-purple-400 bg-purple-500/10 rounded">
                                    {item.type}
                                  </span>
                                </div>
                                {'url' in item && (
                                  <a
                                    href={item.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="p-2 rounded-lg bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-colors"
                                  >
                                    <ExternalLink className="w-4 h-4" />
                                  </a>
                                )}
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </motion.div>
              </TabsContent>

              <TabsContent value="platforms" className="mt-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-[#0f172a]/80 border-slate-800">
                    <CardContent className="p-6">
                      <div className="grid sm:grid-cols-2 gap-4">
                        {dataPlatforms.map((platform) => (
                          <a
                            key={platform.name}
                            href={platform.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-4 p-4 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors group"
                          >
                            <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center flex-shrink-0">
                              <Globe className="w-5 h-5 text-green-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-white font-medium truncate">{platform.name}</h4>
                              <p className="text-slate-500 text-sm truncate">{platform.description}</p>
                            </div>
                            <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-green-400 transition-colors" />
                          </a>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </TabsContent>
            </AnimatePresence>
          </Tabs>
        </ScrollReveal>
      </div>
    </section>
  );
}
