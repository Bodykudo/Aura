'use client';

import { motion } from 'framer-motion';

export default function DownloadHeader() {
  return (
    <motion.div
      initial={{
        opacity: 0,
        y: 20,
      }}
      animate={{
        opacity: 1,
        y: 0,
      }}
      transition={{
        duration: 1,
      }}
      className="div px-4 sm:px-0"
    >
      <h2 className="text-center text-xl md:text-4xl font-bold text-white">
        <span className="bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 animate-gradient text-transparent bg-clip-text">
          Aura
        </span>{' '}
        is available everywhere on the globe.
      </h2>
      <p className="text-center text-base md:text-lg font-normal text-neutral-200 max-w-md mt-2 mx-auto">
        Experience the power of Aura from anywhere in the world. Our platform is
        accessible to everyone, no matter where you are.
      </p>
    </motion.div>
  );
}
