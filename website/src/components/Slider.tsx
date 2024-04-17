'use client';

import { cn } from '@/lib/utils';
import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from 'react-compare-slider';

interface SliderProps {
  className?: string;
  imageOneSrc: string;
  imageTwoSrc: string;
}

export default function Slider({
  className,
  imageOneSrc,
  imageTwoSrc,
}: SliderProps) {
  return (
    <ReactCompareSlider
      className={cn('h-60 md:h-full', className)}
      itemOne={
        <ReactCompareSliderImage
          className="object-top md:object-center"
          src={imageOneSrc}
        />
      }
      itemTwo={<ReactCompareSliderImage src={imageTwoSrc} />}
    />
  );
}
