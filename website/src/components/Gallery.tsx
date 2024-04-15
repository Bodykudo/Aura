'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from 'react-compare-slider';

export default function SkeletonFour() {
  const first = {
    initial: {
      x: 20,
      rotate: -5,
    },
    hover: {
      x: 0,
      rotate: 0,
    },
  };
  const second = {
    initial: {
      x: -20,
      rotate: 5,
    },
    hover: {
      x: 0,
      rotate: 0,
    },
  };

  return (
    <motion.div
      initial="initial"
      animate="animate"
      whileHover="hover"
      className="flex flex-1 w-full h-full min-h-[6rem] bg-dot-white/[0.2] flex-row space-x-2"
    >
      <motion.div
        variants={first}
        className="h-full w-1/3 rounded-2xl p-4 bg-black border-white/[0.1] border flex flex-col items-center justify-center"
      >
        <ReactCompareSlider
          itemOne={<ReactCompareSliderImage src="/low-pass-before.jpg" />}
          itemTwo={<ReactCompareSliderImage src="/low-pass-after.jpg" />}
        />
        <p className="sm:text-sm text-xs text-center font-semibold text-neutral-500 mt-4">
          Low-Pass Filter
        </p>
      </motion.div>
      <motion.div className="h-full relative z-20 w-1/3 rounded-2xl p-4 bg-black border-white/[0.1] border flex flex-col items-center justify-center">
        <Image
          src="/hybrid.jpg"
          alt="avatar"
          height="564"
          width="564"
          className="h-full"
        />
        <p className="sm:text-sm text-xs text-center font-semibold text-neutral-500 mt-4">
          Hybrid Image
        </p>
      </motion.div>
      <motion.div
        variants={second}
        className="h-full w-1/3 rounded-2xl p-4 bg-black border-white/[0.1] border flex flex-col items-center justify-center"
      >
        <ReactCompareSlider
          itemOne={<ReactCompareSliderImage src="/high-pass-before.jpg" />}
          itemTwo={<ReactCompareSliderImage src="/high-pass-after.jpg" />}
        />
        <p className="sm:text-sm text-xs text-center font-semibold text-neutral-500 mt-4">
          High-Pass Filter
        </p>
      </motion.div>
    </motion.div>
  );
}
