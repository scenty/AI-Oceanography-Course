import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { BookOpen, Code, Target, GraduationCap } from 'lucide-react';

const features = [
  {
    icon: BookOpen,
    title: '理论讲授',
    description: '2/3课时用于理论讲授，涵盖人工智能基础与海洋学应用',
  },
  {
    icon: Code,
    title: '编程实践',
    description: '1/3课时用于Python编程实现，实践出真知',
  },
  {
    icon: Target,
    title: '课程目标',
    description: '能够使用AI工具进行海洋科学研究，阅读理解AI文献',
  },
  {
    icon: GraduationCap,
    title: '考核方式',
    description: '大作业形式，综合运用所学知识解决实际问题',
  },
];

export function About() {
  return (
    <section id="about" className="relative py-24 bg-[#020617]">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_50%,rgba(59,130,246,0.08),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <div>
            <ScrollReveal>
              <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-blue-400 bg-blue-400/10 border border-blue-400/20 rounded-full">
                课程简介
              </span>
            </ScrollReveal>
            
            <ScrollReveal delay={0.1}>
              <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-6">
                探索人工智能与海洋科学的
                <span className="text-gradient"> 交叉前沿</span>
              </h2>
            </ScrollReveal>
            
            <ScrollReveal delay={0.2}>
              <p className="text-slate-400 text-lg leading-relaxed mb-6">
                本课程系统介绍人工智能技术在海洋科学研究中的应用，包括神经网络基础、
                深度学习方法（CNN、RNN、注意力机制）以及海洋回归问题、识别问题等典型应用案例。
              </p>
            </ScrollReveal>
            
            <ScrollReveal delay={0.3}>
              <p className="text-slate-400 leading-relaxed mb-8">
                编程语言采用 Python - PyTorch，前置课程需要数据分析及统计、数值模型基础。
                通过理论讲授与编程实践相结合的方式，帮助学生掌握使用AI工具进行海洋科学研究的能力。
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <div className="space-y-4">
                <h3 className="text-white font-semibold mb-3">参考教材</h3>
                <div className="flex items-start gap-3">
                  <div className="w-1 h-1 rounded-full bg-blue-500 mt-2.5" />
                  <p className="text-slate-400">《人工智能海洋学》，董昌明等编</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-1 h-1 rounded-full bg-cyan-500 mt-2.5" />
                  <p className="text-slate-400">Artificial Intelligence Oceanography, Xiaofeng Li、Fan Wang等编</p>
                </div>
              </div>
            </ScrollReveal>
          </div>

          {/* Right Content - Feature Cards */}
          <div className="grid sm:grid-cols-2 gap-4">
            {features.map((feature, index) => (
              <ScrollReveal key={feature.title} delay={0.1 * (index + 1)}>
                <Card className="bg-[#0f172a]/80 border-slate-800 hover:border-blue-500/50 transition-all duration-300 hover:-translate-y-1 group">
                  <CardContent className="p-6">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4 group-hover:bg-blue-500/20 transition-colors">
                      <feature.icon className="w-6 h-6 text-blue-400" />
                    </div>
                    <h3 className="text-white font-semibold mb-2">{feature.title}</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">{feature.description}</p>
                  </CardContent>
                </Card>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
