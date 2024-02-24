import { AudioLines, BarChartBig, Blend, Columns2, FilterX, ScanIcon } from 'lucide-react';

export const routes = [
  {
    label: 'Noise',
    icon: AudioLines,
    href: '/noise',
    color: 'text-violet-500',
    bgColor: 'bg-violet-500/10'
  },
  {
    label: 'Filters',
    icon: FilterX,
    href: '/filters',
    color: 'text-pink-700',
    bgColor: 'bg-pink-700/10'
  },
  {
    label: 'Edge Detection',
    icon: ScanIcon,
    href: '/edge',
    color: 'text-orange-700',
    bgColor: 'bg-orange-700/10'
  },
  {
    label: 'Histogram, Normalization, & Equalization',
    icon: BarChartBig,
    href: '/histogram',
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10'
  },
  {
    label: 'Thresholding',
    icon: Columns2,
    href: '/thresholding',
    color: 'text-green-700',
    bgColor: 'bg-green-700/10'
  },
  {
    label: 'Hybrid Image',
    icon: Blend,
    href: '/hybrid',
    color: 'text-blue-700',
    bgColor: 'bg-blue-700/10'
  }
];
