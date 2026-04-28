import { Navbar } from '@/components/Navbar';
import { Hero } from '@/sections/Hero';
import { About } from '@/sections/About';
import { CourseContent } from '@/sections/CourseContent';
import { Labs } from '@/sections/Labs';
import { ExternalTeaching } from '@/sections/ExternalTeaching';
import { Instructor } from '@/sections/Instructor';
import { Footer } from '@/sections/Footer';
import { useEffect, useState } from 'react';

function App() {
  const [likes, setLikes] = useState(0);

  useEffect(() => {
    const raw = window.localStorage.getItem('aio_likes');
    const n = raw ? Number(raw) : 0;
    setLikes(Number.isFinite(n) ? n : 0);
  }, []);

  useEffect(() => {
    window.localStorage.setItem('aio_likes', String(likes));
  }, [likes]);

  const addLike = () => setLikes((v) => v + 1);

  return (
    <div className="relative min-h-screen bg-[#020617]">
      {/* Navigation */}
      <Navbar />
      
      {/* Main Content */}
      <main className="relative z-10">
        <Hero onLike={addLike} />
        <About />
        <CourseContent />
        <Labs />
        <ExternalTeaching />
        <Instructor />
      </main>
      
      {/* Footer */}
      <Footer likes={likes} />
    </div>
  );
}

export default App;
