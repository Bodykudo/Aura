import {
  ArrowDownRightFromSquare,
  AudioLines,
  BarChartBig,
  Blend,
  BringToFront,
  CircleDotDashed,
  ClipboardX,
  Columns2,
  FilterX,
  PieChart,
  ScanIcon,
  Wand2
} from 'lucide-react';

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
  },
  {
    label: 'Feature Extraction',
    icon: Wand2,
    href: '/hough',
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
    comingSoon: true
  },
  {
    label: 'Active Contours',
    icon: CircleDotDashed,
    href: '/contours',
    color: 'text-cyan-700',
    bgColor: 'bg-cyan-700/10',
    comingSoon: true
  },
  {
    label: 'Corner Detection',
    icon: ArrowDownRightFromSquare,
    href: '/corner',
    color: 'text-rose-600',
    bgColor: 'bg-rose-600/10',
    comingSoon: true
  },
  {
    label: 'Image Matching',
    icon: BringToFront,
    href: '/matching',
    color: 'text-sky-400',
    bgColor: 'bg-sky-400/10',
    comingSoon: true
  },
  {
    label: 'SIFT Descriptors',
    icon: ClipboardX,
    href: '/sift',
    color: 'text-teal-600',
    bgColor: 'bg-teal-600/10',
    comingSoon: true
  },
  {
    label: 'Segmentation',
    icon: PieChart,
    href: '/segmentation',
    color: 'text-red-800',
    bgColor: 'bg-red-800/10',
    comingSoon: true
  }
];
