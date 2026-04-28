import { motion } from 'framer-motion';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { CalendarDays, MapPin, Users, FileText, Download, Image as ImageIcon } from 'lucide-react';

type TeachingFile = { title: string; desc: string; href: string };
type TeachingPhoto = { src: string; alt: string };

type TeachingActivity = {
  id: string;
  summaryTitle: string;
  title: string;
  speaker: string;
  date: string;
  location: string;
  participants?: string;
  outline?: TeachingFile;
  ppt?: TeachingFile;
  extras?: TeachingFile[];
  photos?: TeachingPhoto[];
};

const activities: TeachingActivity[] = [
  {
    id: 'aio5',
    summaryTitle: 'AIO5 论坛培训活动',
    title: 'AI海洋学中的大数据构建和智能参数化问题（兼谈AI海洋学的底层逻辑）',
    speaker: '卢文芳（中山大学海洋科学学院）',
    date: '2026-04-24',
    location: '国家超算济南中心',
    participants: '160+',
    outline: {
      title: '培训大纲（DOCX）',
      desc: '授课安排与内容纲要',
      href: '/external-teaching/training-outline.docx',
    },
    ppt: {
      title: '课件（PDF）',
      desc: '授课PPT导出版，共55页',
      href: '/files/Lu-AIO5授课.pdf',
    },
    photos: [
      { src: '/external-teaching/photos/aio5-1.png', alt: '论坛授课照片 1' },
      { src: '/external-teaching/photos/aio5-2.png', alt: '论坛授课照片 2' },
      { src: '/external-teaching/photos/aio5-3.png', alt: '论坛授课照片 3' },
      { src: '/external-teaching/photos/aio5-4.png', alt: '论坛授课照片 4' },
      { src: '/external-teaching/photos/aio5-5.png', alt: '论坛授课照片 5' },
      { src: '/external-teaching/photos/aio5-6.png', alt: '论坛授课照片 6' },
    ],
  },
  {
    id: 'marine-2025',
    summaryTitle: 'MARINE Summer School',
    title: 'AI Oceanography（Part 1–3: General / Gradient Descent & NN / LLM with your science）',
    speaker: 'Wenfang Lu（SYSU）',
    date: '2025-07-10',
    location: '珠海',
    participants: '40（来自韩国、斯里兰卡等国家）',
    ppt: {
      title: '课件（PDF）',
      desc: 'MARINE Summer School 2025（英文版，78页）',
      href: '/files/AIO_lecture_Lu-FIN.pdf',
    },
    photos: [
      { src: '/external-teaching/photos/marine-2025-1.png', alt: 'MARINE Summer School 授课合影' },
      { src: '/external-teaching/photos/marine-2025-2.png', alt: 'MARINE Summer School 授课现场' },
    ],
  },
  {
    id: 'activity-3',
    summaryTitle: '外部授课（待补充）',
    title: '待补充',
    speaker: '卢文芳',
    date: '待补充',
    location: '待补充',
    participants: '待补充',
    photos: [],
  },
];

function FileRow({ file }: { file: TeachingFile }) {
  return (
    <div className="flex items-center justify-between gap-4 p-4 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
      <div className="min-w-0">
        <div className="text-white font-medium">{file.title}</div>
        <div className="text-slate-500 text-sm truncate">{file.desc}</div>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        <a href={file.href} target="_blank" rel="noopener noreferrer">
          <Button
            variant="secondary"
            className="bg-blue-500/10 text-blue-300 border border-blue-500/20 hover:bg-blue-500/20"
          >
            打开
          </Button>
        </a>
        <a href={file.href} download>
          <Button
            variant="secondary"
            className="bg-slate-900/40 text-slate-200 border border-slate-700 hover:bg-slate-900/60"
          >
            <Download className="w-4 h-4 mr-2" />
            下载
          </Button>
        </a>
      </div>
    </div>
  );
}

export function ExternalTeaching() {
  return (
    <section id="external-teaching" className="relative py-24 bg-[#020617]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_30%,rgba(59,130,246,0.08),transparent_55%)]" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <ScrollReveal>
            <span className="inline-block px-3 py-1 mb-4 text-sm font-medium text-blue-400 bg-blue-400/10 border border-blue-400/20 rounded-full">
              外部授课
            </span>
          </ScrollReveal>

          <ScrollReveal delay={0.1}>
            <h2 className="font-serif text-3xl sm:text-4xl font-bold text-white mb-4">
              培训活动与授课资料
            </h2>
          </ScrollReveal>

          <ScrollReveal delay={0.2}>
            <p className="text-slate-400 max-w-3xl mx-auto">
              这里汇总卢文芳教授参与的外部授课活动。点击条目展开，可查看时间、地点、参与人数，以及大纲、课件与照片等素材。
            </p>
          </ScrollReveal>
        </div>

        <ScrollReveal delay={0.25}>
          <Card className="bg-[#0f172a]/80 border-slate-800 overflow-hidden">
            <CardContent className="p-0">
              <Accordion type="multiple" className="px-6">
                {activities.map((a) => {
                  const photoCount = a.photos ? a.photos.length : 0;
                  const files: TeachingFile[] = [
                    ...(a.outline ? [a.outline] : []),
                    ...(a.ppt ? [a.ppt] : []),
                    ...(a.extras ? a.extras : []),
                  ];

                  return (
                    <AccordionItem key={a.id} value={a.id} className="border-slate-800">
                      <AccordionTrigger className="text-slate-200 hover:text-white hover:no-underline">
                        <div className="flex items-start justify-between gap-4 w-full pr-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-500/10 flex-shrink-0">
                                <FileText className="w-4 h-4 text-blue-400" />
                              </span>
                              <div className="min-w-0">
                                <div className="text-base font-semibold text-white truncate">{a.summaryTitle}</div>
                                <div className="text-sm text-slate-400 truncate">{a.title}</div>
                              </div>
                            </div>
                            <div className="mt-2 text-xs text-slate-500">
                              {a.speaker}
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <div className="text-sm text-slate-300">{a.date}</div>
                            <div className="text-xs text-slate-500 truncate max-w-[12rem]">{a.location}</div>
                          </div>
                        </div>
                      </AccordionTrigger>

                      <AccordionContent className="pb-6">
                        <div className="grid lg:grid-cols-12 gap-6">
                          <div className="lg:col-span-5">
                            <div className="grid sm:grid-cols-3 gap-3">
                              <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2">
                                <CalendarDays className="w-4 h-4 text-slate-400" />
                                <span className="text-slate-300 text-sm">{a.date}</span>
                              </div>
                              <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2 sm:col-span-2">
                                <MapPin className="w-4 h-4 text-slate-400" />
                                <span className="text-slate-300 text-sm truncate">{a.location}</span>
                              </div>
                              <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2 sm:col-span-3">
                                <Users className="w-4 h-4 text-slate-400" />
                                <span className="text-slate-300 text-sm">参与人数：{a.participants || '待补充'}</span>
                              </div>
                            </div>

                            <div className="mt-6 space-y-3">
                              {files.length === 0 ? (
                                <div className="p-4 rounded-lg border border-dashed border-slate-700 text-slate-400 text-sm bg-slate-900/30">
                                  素材待补充（大纲 / PPT / 照片等）
                                </div>
                              ) : (
                                files.map((f) => <FileRow key={f.href} file={f} />)
                              )}
                            </div>
                          </div>

                          <div className="lg:col-span-7">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-white font-semibold flex items-center gap-2">
                                <ImageIcon className="w-5 h-5 text-cyan-400" />
                                现场照片
                              </h4>
                              <span className="text-slate-500 text-sm">{photoCount} 张</span>
                            </div>

                            {photoCount === 0 ? (
                              <div className="p-4 rounded-lg border border-dashed border-slate-700 text-slate-400 text-sm bg-slate-900/30">
                                暂无照片素材
                              </div>
                            ) : (
                              <motion.div
                                initial={{ opacity: 0, y: 12 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.35 }}
                                className="grid grid-cols-2 md:grid-cols-3 gap-3"
                              >
                                {(a.photos || []).map((p) => (
                                  <a
                                    key={p.src}
                                    href={p.src}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="group relative overflow-hidden rounded-lg border border-slate-800 bg-slate-900/40"
                                  >
                                    <img
                                      src={p.src}
                                      alt={p.alt}
                                      className="w-full h-32 sm:h-36 md:h-40 object-cover transition-transform duration-500 group-hover:scale-105"
                                      loading="lazy"
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-[#0f172a]/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </a>
                                ))}
                              </motion.div>
                            )}
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  );
                })}
              </Accordion>
            </CardContent>
          </Card>
        </ScrollReveal>
      </div>
    </section>
  );
}

