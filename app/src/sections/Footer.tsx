import { Waves, Mail, MapPin, ExternalLink } from 'lucide-react';

const quickLinks = [
  { label: '课程介绍', href: '#about' },
  { label: '课程大纲', href: '#syllabus' },
  { label: '编程练习', href: '#labs' },
  { label: '课程资源', href: '#resources' },
];

const resources = [
  { label: '课件下载', href: '#resources' },
  { label: '代码示例', href: '#resources' },
  { label: '参考资料', href: '#resources' },
  { label: '数据平台', href: '#resources' },
];

export function Footer() {
  const scrollToSection = (href: string) => {
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <footer className="relative bg-[#0a0f1c] border-t border-slate-800">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(59,130,246,0.05),transparent_50%)]" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid md:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <a href="#hero" className="flex items-center gap-2 mb-4">
              <Waves className="w-8 h-8 text-blue-500" />
              <span className="font-serif text-xl font-semibold text-white">
                AI Ocean
              </span>
            </a>
            <p className="text-slate-400 mb-6 max-w-md">
              人工智能海洋学课程，探索AI与海洋科学的交叉前沿。
              从神经网络基础到深度学习应用，系统掌握智能海洋数据分析方法。
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Mail className="w-4 h-4" />
                <span>luwf6@mail.sysu.edu.cn</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <MapPin className="w-4 h-4" />
                <span>中山大学 海洋科学学院</span>
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">快速导航</h3>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection(link.href);
                    }}
                    className="text-slate-400 hover:text-blue-400 transition-colors text-sm"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold mb-4">课程资源</h3>
            <ul className="space-y-2">
              {resources.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection(link.href);
                    }}
                    className="text-slate-400 hover:text-blue-400 transition-colors text-sm"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-16 pt-8 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-slate-500 text-sm">
            © 2025 中山大学 人工智能海洋学课程. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <a
              href="https://www.sysu.edu.cn"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-white transition-colors text-sm flex items-center gap-1"
            >
              中山大学
              <ExternalLink className="w-3 h-3" />
            </a>
            <span className="text-slate-700">·</span>
            <a
              href="https://marine.sysu.edu.cn"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-white transition-colors text-sm flex items-center gap-1"
            >
              海洋科学学院
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
