import { Navbar } from '@/components/Navbar';
import { Hero } from '@/sections/Hero';
import { About } from '@/sections/About';
import { CourseContent } from '@/sections/CourseContent';
import { Labs } from '@/sections/Labs';
import { Resources } from '@/sections/Resources';
import { Instructor } from '@/sections/Instructor';
import { Footer } from '@/sections/Footer';

function App() {
  return (
    <div className="relative min-h-screen bg-[#020617]">
      {/* Navigation */}
      <Navbar />
      
      {/* Main Content */}
      <main className="relative z-10">
        <Hero />
        <About />
        <CourseContent />
        <Labs />
        <Resources />
        <Instructor />
      </main>
      
      {/* Footer */}
      <Footer />
    </div>
  );
}

export default App;
