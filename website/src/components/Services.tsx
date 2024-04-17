import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from 'react-compare-slider';

import { BentoGrid, BentoGridItem } from './ui/bento-grid';
import HistogramChart from './HistogramChart';

import { histogramBefore, histogramAfter } from '@/lib/histogram';

import {
  IconArrowWaveRightUp,
  IconBoxAlignRightFilled,
  IconBoxAlignTopLeft,
  IconClipboardCopy,
  IconFileBroken,
  IconSignature,
  IconTableColumn,
} from '@tabler/icons-react';
import { TypewriterEffect } from './ui/typewriter-effect';
import SkeletonFour from './Gallery';

const words = [
  {
    text: 'Explore',
  },
  {
    text: 'awesome',
  },
  {
    text: 'stuff',
  },
  {
    text: 'with',
  },
  {
    text: 'Aura',
    className:
      'bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 animate-gradient text-transparent bg-clip-text',
  },
  { text: '.' },
];

const items = [
  {
    title: 'Add Noise to Images',
    description: 'Create different types of noise in images.',
    header: (
      <ReactCompareSlider
        className="h-60 md:h-full"
        itemOne={
          <ReactCompareSliderImage
            className="object-top md:object-center"
            src="/noise-before.jpg"
          />
        }
        itemTwo={<ReactCompareSliderImage src="/noise-after.jpg" />}
      />
    ),
    icon: <IconClipboardCopy className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Filter Noisy Images',
    description: 'Remove noise from images using Aura filters.',
    header: (
      <ReactCompareSlider
        itemOne={<ReactCompareSliderImage src="/filter-before.jpg" />}
        itemTwo={<ReactCompareSliderImage src="/filter-after.jpg" />}
      />
    ),
    icon: <IconFileBroken className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Detect Edges in Images',
    description: 'Identify edges in images using Aura edge detection.',
    header: (
      <ReactCompareSlider
        itemOne={<ReactCompareSliderImage src="/edges-before.jpg" />}
        itemTwo={<ReactCompareSliderImage src="/edges-after.jpg" />}
      />
    ),
    icon: <IconSignature className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Transform Image Histograms',
    description:
      'Equalization, normalization, grayscaling, all histogram transformations are available in Aura toolkit.',
    header: (
      <div className="flex flex-col md:flex-row overflow-hidden items-center justify-center gap-1 h-60">
        <ReactCompareSlider
          itemOne={<ReactCompareSliderImage src="/equalization-before.png" />}
          itemTwo={<ReactCompareSliderImage src="/equalization-after.jpg" />}
        />
        <div className="flex flex-col sm:flex-row md:flex-col items-center justify-center w-full h-full">
          <HistogramChart data={histogramBefore} />
          <HistogramChart data={histogramAfter} />
        </div>
      </div>
    ),
    icon: <IconTableColumn className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Apply Thresholding to Images',
    description: 'Apply thresholding to images using Aura tools.',
    header: (
      <ReactCompareSlider
        itemOne={<ReactCompareSliderImage src="/threshold-before.jpg" />}
        itemTwo={<ReactCompareSliderImage src="/threshold-after.jpg" />}
      />
    ),
    icon: <IconArrowWaveRightUp className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Detect Boundaries in Images',
    description: 'Identify boundaries in images using Active Contour.',
    header: (
      <ReactCompareSlider
        itemOne={<ReactCompareSliderImage src="/contour-before.jpg" />}
        itemTwo={<ReactCompareSliderImage src="/contour-after.jpg" />}
      />
    ),
    icon: <IconBoxAlignTopLeft className="h-4 w-4 text-neutral-500" />,
  },
  {
    // title: 'Detect Boundaries in Images',
    // description: 'Identify boundaries in images using Active Contour.',
    header: <SkeletonFour />,
    // icon: <IconBoxAlignRightFilled className="h-4 w-4 text-neutral-500" />,
  },
];

export default function Services() {
  return (
    <div
      id="features"
      className="flex flex-col w-full bg-black bg-grid-white/[0.2] relative items-center justify-center"
    >
      <div className="absolute pointer-events-none inset-0 flex items-center justify-center bg-black [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"></div>
      <div className="mt-4 py-4 z-10">
        <TypewriterEffect words={words} />
      </div>
      <BentoGrid className="max-w-4xl mx-auto z-10">
        {items.map((item, i) => (
          <BentoGridItem
            key={i}
            title={item.title}
            description={item.description}
            header={item.header}
            icon={item.icon}
            className={i === 3 || i === 6 ? 'md:col-span-2' : ''}
          />
        ))}
      </BentoGrid>
    </div>
  );
}
