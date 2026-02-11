import { motion } from 'framer-motion';
import { ChevronDown, BookOpen, Code } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getImagePath } from '@/lib/utils';
import { ParticleBackground } from '@/components/ParticleBackground';

export function Hero() {
  const scrollToSection = (href: string) => {
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section
      id="hero"
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat z-0"
        style={{
          backgroundImage: `url(${getImagePath('/images/hero-bg.jpg')})`,
        }}
      />
      
      {/* Gradient Overlay - 降低透明度让气泡更明显 */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#020617]/40 via-[#020617]/30 to-[#020617]/80 z-[1]" />
      
      {/* Radial Glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_30%,rgba(59,130,246,0.15),transparent_50%)] z-[2]" />

      {/* Particle Background - 放在 Hero 内部以确保正确的堆叠顺序 */}
      <div className="absolute inset-0 z-[5] overflow-hidden">
        <ParticleBackground />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: [0.165, 0.84, 0.44, 1] }}
        >
          <span className="inline-block px-4 py-1.5 mb-6 text-sm font-medium text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full">
            2024-2025学年 春季学期
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: [0.165, 0.84, 0.44, 1] }}
          className="font-serif text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-4 tracking-tight"
        >
          人工智能海洋学
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4, ease: [0.165, 0.84, 0.44, 1] }}
          className="text-xl sm:text-2xl text-slate-300 mb-2 font-light"
        >
          Artificial Intelligence Oceanography
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5, ease: [0.165, 0.84, 0.44, 1] }}
          className="flex items-center justify-center gap-2 text-slate-400 mb-8"
        >
          <span className="w-8 h-px bg-slate-600" />
          <span>中山大学</span>
          <span className="w-8 h-px bg-slate-600" />
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6, ease: [0.165, 0.84, 0.44, 1] }}
          className="text-slate-400 mb-10 max-w-2xl mx-auto"
        >
          授课教师：卢文芳 副教授 · 海洋科学学院
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7, ease: [0.165, 0.84, 0.44, 1] }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Button
            size="lg"
            className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-6 text-lg rounded-xl transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-blue-500/25"
            onClick={() => scrollToSection('#about')}
          >
            <BookOpen className="w-5 h-5 mr-2" />
            开始学习
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="border-slate-600 text-slate-300 hover:bg-white/5 hover:text-white px-8 py-6 text-lg rounded-xl transition-all"
            onClick={() => scrollToSection('#syllabus')}
          >
            <Code className="w-5 h-5 mr-2" />
            查看课程大纲
          </Button>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          className="flex flex-col items-center text-slate-500 cursor-pointer"
          onClick={() => scrollToSection('#about')}
        >
          <span className="text-xs mb-2">向下滚动</span>
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </motion.div>
    </section>
  );
}
