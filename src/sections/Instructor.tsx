import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Mail, MapPin, Clock, Building } from 'lucide-react';

export function Instructor() {
  return (
    <section id="instructor" className="relative py-24 bg-[#020617]">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_70%,rgba(139,92,246,0.08),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-pink-400 bg-pink-400/10 border border-pink-400/20 rounded-full">
              教师信息
            </span>
          </ScrollReveal>
          
          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              授课教师
            </h2>
          </ScrollReveal>
        </div>

        {/* Instructor Card */}
        <ScrollReveal delay={0.2}>
          <Card className="max-w-3xl mx-auto bg-[#0f172a]/80 border-slate-800 overflow-hidden">
            <CardContent className="p-0">
              <div className="grid md:grid-cols-[200px_1fr]">
                {/* Avatar Section */}
                <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 p-8 flex items-center justify-center">
                  <div className="w-32 h-32 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                    <span className="text-4xl font-serif font-bold text-white">卢</span>
                  </div>
                </div>
                
                {/* Info Section */}
                <div className="p-8">
                  <h3 className="text-2xl font-serif font-bold text-white mb-1">
                    卢文芳
                  </h3>
                  <p className="text-blue-400 mb-6">副教授</p>
                  
                  <div className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                        <Building className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">所属单位</p>
                        <p className="text-white">中山大学 海洋科学学院</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
                        <Mail className="w-4 h-4 text-cyan-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">电子邮箱</p>
                        <a 
                          href="mailto:luwf6@mail.sysu.edu.cn" 
                          className="text-white hover:text-cyan-400 transition-colors"
                        >
                          luwf6@mail.sysu.edu.cn
                        </a>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                        <MapPin className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">办公地点</p>
                        <p className="text-white">海琴三号楼 A526</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-pink-500/10 flex items-center justify-center flex-shrink-0">
                        <Clock className="w-4 h-4 text-pink-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">上课时间</p>
                        <p className="text-white">周三 7-8节 · 周五 5-6节</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6 pt-6 border-t border-slate-800">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-green-500/10 flex items-center justify-center flex-shrink-0">
                        <MapPin className="w-4 h-4 text-green-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">上课地点</p>
                        <p className="text-white">教学大楼珠海 A305</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </ScrollReveal>
      </div>
    </section>
  );
}
