import { useState } from 'react';
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
  Code,
  ChevronDown,
  ChevronUp,
  Play,
  Lightbulb
} from 'lucide-react';
import { labExercises } from '@/data/courseContent';

const difficultyLabels = {
  beginner: { label: '入门', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
  intermediate: { label: '进阶', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  advanced: { label: '高级', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
};

const categories = [
  { id: '基础练习', icon: Calculator, color: 'blue' },
  { id: '海洋数据处理', icon: Waves, color: 'cyan' },
  { id: '神经网络基础', icon: Brain, color: 'purple' },
  { id: 'PyTorch实践', icon: Database, color: 'pink' },
  { id: '深度学习应用', icon: Cpu, color: 'orange' },
];

export function Labs() {
  const [expandedLab, setExpandedLab] = useState<string | null>(null);

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
              从Python基础到深度学习应用，通过15个编程练习逐步掌握人工智能海洋学的核心技能
            </p>
          </ScrollReveal>
        </div>

        {/* Category Tabs */}
        <ScrollReveal delay={0.3}>
          <Tabs defaultValue="基础练习" className="w-full">
            <TabsList className="flex flex-wrap justify-center gap-2 bg-transparent mb-8 h-auto">
              {categories.map((category) => (
                <TabsTrigger
                  key={category.id}
                  value={category.id}
                  className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50 border border-slate-800 bg-[#0f172a]/80 text-slate-400 px-4 py-2 rounded-lg transition-all flex items-center gap-2"
                >
                  <category.icon className="w-4 h-4" />
                  <span className="text-sm">{category.id}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {categories.map((category) => {
              const categoryLabs = labExercises.filter(lab => lab.category === category.id);
              
              return (
                <TabsContent key={category.id} value={category.id} className="mt-0">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                  >
                    <div className="grid md:grid-cols-2 gap-4">
                      {categoryLabs.map((lab, index) => (
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
                                  {expandedLab === lab.id ? (
                                    <ChevronUp className="w-4 h-4 text-slate-500" />
                                  ) : (
                                    <ChevronDown className="w-4 h-4 text-slate-500" />
                                  )}
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
                                      {lab.codeTemplate && (
                                        <div className="mb-4">
                                          <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                                            <Code className="w-3 h-3" />
                                            <span>代码模板</span>
                                          </div>
                                          <pre className="text-xs text-slate-400 bg-slate-900/80 p-4 rounded-lg overflow-x-auto font-mono leading-relaxed">
                                            <code>{lab.codeTemplate}</code>
                                          </pre>
                                        </div>
                                      )}

                                      {lab.hints && (
                                        <div>
                                          <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                                            <Lightbulb className="w-3 h-3" />
                                            <span>提示</span>
                                          </div>
                                          <ul className="space-y-1">
                                            {lab.hints.map((hint, i) => (
                                              <li key={i} className="text-sm text-slate-400 flex items-start gap-2">
                                                <span className="text-cyan-500">•</span>
                                                {hint}
                                              </li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}

                                      <div className="mt-4 flex gap-2">
                                        <button className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg text-sm hover:bg-cyan-500/30 transition-colors">
                                          <Play className="w-4 h-4" />
                                          运行代码
                                        </button>
                                        <button className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-300 rounded-lg text-sm hover:bg-slate-700 transition-colors">
                                          <FileCode className="w-4 h-4" />
                                          查看完整代码
                                        </button>
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
