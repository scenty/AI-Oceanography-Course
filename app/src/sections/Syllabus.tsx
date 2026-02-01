import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Database, 
  Network, 
  Flame, 
  Layers, 
  Waves,
  Cpu,
  Target,
  Microscope
} from 'lucide-react';

interface SyllabusItem {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: React.ElementType;
  topics: string[];
  hours: number;
  type: 'theory' | 'practice';
}

const syllabusData: SyllabusItem[] = [
  {
    id: '01',
    title: '人工智能概述',
    subtitle: 'AI Overview',
    description: '人工智能发展历程、DeepSeek技术突破、AI在海洋学中的应用方向',
    icon: Brain,
    topics: ['AI发展历程', 'DeepSeek', 'AI应用方向', '海洋大数据'],
    hours: 1,
    type: 'theory',
  },
  {
    id: '02',
    title: '海洋大数据简介',
    subtitle: 'Ocean Big Data',
    description: '大数据概况、海洋数据发展历程、数据来源与特征、常用数据平台',
    icon: Database,
    topics: ['大数据5V特征', '海洋观测技术', '数据来源', 'Copernicus平台'],
    hours: 1,
    type: 'theory',
  },
  {
    id: '03',
    title: '神经网络基础',
    subtitle: 'Neural Networks',
    description: '机器学习基础、梯度下降算法、神经网络结构与反向传播',
    icon: Network,
    topics: ['机器学习', '梯度下降', '激活函数', '反向传播', '特征缩放', '正则化'],
    hours: 6,
    type: 'theory',
  },
  {
    id: '04',
    title: 'PyTorch基础',
    subtitle: 'PyTorch Basics',
    description: 'Tensor操作、自动微分、数据加载、模型构建与训练',
    icon: Flame,
    topics: ['Tensor基础', '自动微分', 'DataLoader', 'nn.Module', '优化器'],
    hours: 1,
    type: 'practice',
  },
  {
    id: '05',
    title: '深度学习',
    subtitle: 'Deep Learning',
    description: '卷积神经网络CNN、循环神经网络RNN、注意力机制',
    icon: Layers,
    topics: ['CNN', '卷积操作', '池化层', 'LeNet', 'RNN', 'LSTM', 'Attention'],
    hours: 5,
    type: 'theory',
  },
  {
    id: '06',
    title: '海洋回归问题 - 重建',
    subtitle: 'Ocean Regression - Reconstruction',
    description: '深海遥感、U-Net网络、热含量重建、超分辨率重建',
    icon: Waves,
    topics: ['深海遥感', 'U-Net', 'OHC重建', '超分辨率', 'Earthformer'],
    hours: 2,
    type: 'theory',
  },
  {
    id: '07',
    title: '海洋时序问题',
    subtitle: 'Ocean Time Series',
    description: '时间序列预测、LSTM应用、ConvLSTM、时空预测',
    icon: Cpu,
    topics: ['时间序列', 'LSTM预测', 'ConvLSTM', '时空预测', 'ENSO预报'],
    hours: 1,
    type: 'theory',
  },
  {
    id: '08',
    title: '海洋预测问题 - 注意力',
    subtitle: 'Ocean Prediction - Attention',
    description: 'Transformer架构、自注意力机制、时空注意力、波浪预测',
    icon: Target,
    topics: ['Transformer', 'Self-Attention', 'ViT', '时空注意力', '波浪预测'],
    hours: 1,
    type: 'theory',
  },
  {
    id: '09',
    title: '海洋识别问题',
    subtitle: 'Ocean Recognition',
    description: '目标检测、语义分割、涡旋识别、R-CNN与YOLO',
    icon: Microscope,
    topics: ['目标检测', '语义分割', '涡旋识别', 'R-CNN', 'YOLO', 'U-Net分割'],
    hours: 1,
    type: 'theory',
  },
];

export function Syllabus() {
  return (
    <section id="syllabus" className="relative py-24 bg-[#020617]">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_30%,rgba(139,92,246,0.08),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-purple-400 bg-purple-400/10 border border-purple-400/20 rounded-full">
              课程大纲
            </span>
          </ScrollReveal>
          
          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              系统学习人工智能海洋学
            </h2>
          </ScrollReveal>
          
          <ScrollReveal delay={0.2}>
            <p className="text-slate-400 max-w-2xl mx-auto">
              总计26次课程，涵盖人工智能基础理论、深度学习方法与海洋学典型应用
            </p>
          </ScrollReveal>
        </div>

        {/* Stats */}
        <ScrollReveal delay={0.3}>
          <div className="flex flex-wrap justify-center gap-8 mb-16">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">26</div>
              <div className="text-slate-500 text-sm">总课时</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-cyan-400">16</div>
              <div className="text-slate-500 text-sm">理论讲授</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400">9</div>
              <div className="text-slate-500 text-sm">编程实践</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-pink-400">1</div>
              <div className="text-slate-500 text-sm">课程展示</div>
            </div>
          </div>
        </ScrollReveal>

        {/* Syllabus Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {syllabusData.map((item, index) => (
            <ScrollReveal key={item.id} delay={0.1 * (index + 1)}>
              <Card className="h-full bg-[#0f172a]/80 border-slate-800 hover:border-blue-500/50 transition-all duration-300 hover:-translate-y-1 group overflow-hidden">
                <CardContent className="p-0">
                  {/* Header */}
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                        <item.icon className="w-6 h-6 text-blue-400" />
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant={item.type === 'theory' ? 'default' : 'secondary'}
                          className={item.type === 'theory' 
                            ? 'bg-blue-500/20 text-blue-400 border-blue-500/30' 
                            : 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
                          }
                        >
                          {item.type === 'theory' ? '理论' : '实践'}
                        </Badge>
                        <span className="text-xs text-slate-500">{item.hours}课时</span>
                      </div>
                    </div>
                    
                    <div className="flex items-baseline gap-2 mb-2">
                      <span className="text-sm font-mono text-slate-500">{item.id}</span>
                      <h3 className="text-white font-semibold">{item.title}</h3>
                    </div>
                    <p className="text-xs text-slate-500 mb-3">{item.subtitle}</p>
                    <p className="text-slate-400 text-sm leading-relaxed mb-4">
                      {item.description}
                    </p>
                  </div>
                  
                  {/* Topics */}
                  <div className="px-6 pb-6">
                    <div className="flex flex-wrap gap-2">
                      {item.topics.map((topic) => (
                        <span
                          key={topic}
                          className="px-2 py-1 text-xs text-slate-400 bg-slate-800/80 rounded-md"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
