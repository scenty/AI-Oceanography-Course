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
  fetchRemoteLikes,
  incrementRemoteLikes,
  hasLikedRecently,
  markLiked,
  clearLiked,
  readLikesLocalStorage,
} from '@/lib/likesRemote';

function App() {
  const [likes, setLikes] = useState(0);
  const [hasLiked, setHasLiked] = useState(false);

  useEffect(() => {
    setHasLiked(hasLikedRecently());
    setLikes(readLikesLocalStorage());
    fetchRemoteLikes().then((count) => setLikes(count));
  }, []);

  const addLike = async () => {
    if (hasLiked) return;

    setLikes((v) => v + 1);
    setHasLiked(true);
    markLiked();

    const newCount = await incrementRemoteLikes();
    if (newCount !== null) {
      setLikes(newCount);
      return;
    }

    // 远程写入失败：回滚乐观更新，允许用户重试
    setLikes((v) => Math.max(0, v - 1));
    setHasLiked(false);
    clearLiked();
  };

  return (
    <div className="relative min-h-screen bg-[#020617]">
      {/* Navigation */}
      <Navbar />

      {/* Main Content */}
      <main className="relative z-10">
        <Hero onLike={addLike} likes={likes} hasLiked={hasLiked} />
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
