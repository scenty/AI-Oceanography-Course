import { Navbar } from '@/components/Navbar';
import { Hero } from '@/sections/Hero';
import { About } from '@/sections/About';
import { CourseContent } from '@/sections/CourseContent';
import { Labs } from '@/sections/Labs';
import { ExternalTeaching } from '@/sections/ExternalTeaching';
import { Instructor } from '@/sections/Instructor';
import { Footer } from '@/sections/Footer';
import { useEffect, useState } from 'react';
import {
  getLikesApiPostUrl,
  getLikesJsonRawUrl,
  readLikesLocalStorage,
  writeLikesLocalStorage,
} from '@/lib/likesRemote';

function App() {
  const [likes, setLikes] = useState(0);

  useEffect(() => {
    setLikes(readLikesLocalStorage());
    const rawUrl = getLikesJsonRawUrl();
    if (!rawUrl) return;
    const bust = `?t=${Date.now()}`;
    fetch(`${rawUrl}${bust}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data || typeof data.count !== 'number') return;
        const c = Math.max(0, Math.floor(data.count));
        setLikes(c);
        writeLikesLocalStorage(c);
      });
  }, []);

  useEffect(() => {
    writeLikesLocalStorage(likes);
  }, [likes]);

  const addLike = () => {
    setLikes((v) => v + 1);
    const postUrl = getLikesApiPostUrl();
    if (!postUrl) return;
    fetch(postUrl, { method: 'POST' })
      .then((r) => {
        if (!r.ok) {
          setLikes((v) => Math.max(0, v - 1));
          return null;
        }
        return r.json();
      })
      .then((data) => {
        if (data && typeof data.count === 'number') {
          const c = Math.max(0, Math.floor(data.count));
          setLikes(c);
          writeLikesLocalStorage(c);
        }
      });
  };

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
