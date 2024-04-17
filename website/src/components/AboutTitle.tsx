import { motion } from 'framer-motion';
import { Highlight } from './ui/hero-highlight';

export default function AboutTitle() {
  return (
    <motion.h1
      initial={{
        opacity: 0,
        y: 20,
      }}
      animate={{
        opacity: 1,
        y: [20, -5, 0],
      }}
      transition={{
        duration: 0.5,
        ease: [0.4, 0.0, 0.2, 1],
      }}
      className="text-2xl px-4 md:text-4xl lg:text-5xl font-bold text-white max-w-4xl leading-relaxed lg:leading-snug text-center mx-auto py-10"
    >
      With{' '}
      <span className="bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 animate-gradient text-transparent bg-clip-text">
        Aura
      </span>
      , every pixel tells a story. Each image is not just a picture, but a{' '}
      <Highlight className="text-white">
        playground for exploration and learning.
      </Highlight>
    </motion.h1>
  );
}
