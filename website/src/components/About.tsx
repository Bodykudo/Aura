'use client';

import AboutCards from './AboutCards';
import AboutTitle from './AboutTitle';
import { HeroHighlight } from './ui/hero-highlight';

export default function About() {
  return (
    <div id="about" className="bg-black">
      <AboutTitle />
      <AboutCards />
    </div>
  );
}
