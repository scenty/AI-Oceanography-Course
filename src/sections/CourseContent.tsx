// Course Content Section with chapter tabs and detailed sections
import { motion } from 'framer-motion';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BookOpen, 
  Clock, 
  ChevronRight, 
  Code, 
  Lightbulb,
  Image as ImageIcon
} from 'lucide-react';
import { courseChapters } from '@/data/courseContent';

export function CourseContent() {
  return (
    <section id="course-content" className="relative py-24 bg-[#020617]">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_30%,rgba(59,130,246,0.08),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-blue-400 bg-blue-400/10 border border-blue-400/20 rounded-full">
              课程内容
            </span>
          </ScrollReveal>
          
          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              系统学习人工智能海洋学
            </h2>
          </ScrollReveal>
          
          <ScrollReveal delay={0.2}>
            <p className="text-slate-400 max-w-2xl mx-auto">
              基于PDF课件内容，按章节详细讲解人工智能与海洋科学的交叉知识
            </p>
          </ScrollReveal>
        </div>

        {/* Chapter Tabs */}
        <ScrollReveal delay={0.3}>
          <Tabs defaultValue={courseChapters[0].id} className="w-full">
            <TabsList className="flex flex-wrap justify-start gap-2 bg-transparent mb-8 h-auto">
              {courseChapters.map((chapter) => (
                <TabsTrigger
                  key={chapter.id}
                  value={chapter.id}
                  className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400 data-[state=active]:border-blue-500/50 border border-slate-800 bg-[#0f172a]/80 text-slate-400 px-4 py-2 rounded-lg transition-all"
                >
                  <span className="text-xs font-mono mr-2 opacity-60">{chapter.id}</span>
                  <span className="text-sm">{chapter.title}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {courseChapters.map((chapter) => (
              <TabsContent key={chapter.id} value={chapter.id} className="mt-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                >
                  {/* Chapter Header */}
                  <Card className="bg-[#0f172a]/80 border-slate-800 mb-8">
                    <CardContent className="p-6">
                      <div className="flex flex-wrap items-center gap-4 mb-4">
                        <Badge 
                          variant={chapter.type === 'theory' ? 'default' : 'secondary'}
                          className={chapter.type === 'theory' 
                            ? 'bg-blue-500/20 text-blue-400 border-blue-500/30' 
                            : 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
                          }
                        >
                          {chapter.type === 'theory' ? '理论课' : '实践课'}
                        </Badge>
                        <div className="flex items-center gap-2 text-slate-500 text-sm">
                          <Clock className="w-4 h-4" />
                          <span>{chapter.hours} 课时</span>
                        </div>
                      </div>
                      <h3 className="text-2xl font-serif font-bold text-white mb-2">
                        {chapter.title}
                      </h3>
                      <p className="text-slate-400 text-sm mb-2">{chapter.subtitle}</p>
                      <p className="text-slate-300">{chapter.description}</p>
                    </CardContent>
                  </Card>

                  {/* Sections Grid */}
                  <div className="grid lg:grid-cols-2 gap-6">
                    {chapter.sections.map((section, index) => (
                      <motion.div
                        key={section.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: index * 0.1 }}
                      >
                        <Card 
                          className="h-full bg-[#0f172a]/80 border-slate-800 hover:border-blue-500/50 transition-all duration-300 overflow-hidden group"
                        >
                          <CardContent className="p-0">
                            {/* Image Section */}
                            {section.image && (
                              <div className="relative h-48 overflow-hidden">
                                <img
                                  src={section.image}
                                  alt={section.title}
                                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0f172a] to-transparent" />
                              </div>
                            )}
                            
                            {/* Content Section */}
                            <div className="p-6">
                              <div className="flex items-center gap-2 mb-3">
                                {section.image ? (
                                  <ImageIcon className="w-4 h-4 text-blue-400" />
                                ) : section.codeExample ? (
                                  <Code className="w-4 h-4 text-cyan-400" />
                                ) : (
                                  <BookOpen className="w-4 h-4 text-purple-400" />
                                )}
                                <h4 className="text-white font-semibold">{section.title}</h4>
                              </div>
                              
                              <p className="text-slate-400 text-sm leading-relaxed mb-4">
                                {section.description}
                              </p>

                              {/* Key Points */}
                              <div className="space-y-2">
                                <div className="flex items-center gap-2 text-xs text-slate-500">
                                  <Lightbulb className="w-3 h-3" />
                                  <span>核心要点</span>
                                </div>
                                <ul className="space-y-1.5">
                                  {section.keyPoints.map((point, i) => (
                                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                      <ChevronRight className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                                      <span>{point}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>

                              {/* Code Example Preview */}
                              {section.codeExample && (
                                <div className="mt-4 pt-4 border-t border-slate-800">
                                  <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                                    <Code className="w-3 h-3" />
                                    <span>代码示例</span>
                                  </div>
                                  <pre className="text-xs text-slate-400 bg-slate-900/50 p-3 rounded-lg overflow-x-auto">
                                    <code>{section.codeExample.slice(0, 200)}...</code>
                                  </pre>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              </TabsContent>
            ))}
          </Tabs>
        </ScrollReveal>
      </div>
    </section>
  );
}
