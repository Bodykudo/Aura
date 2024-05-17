import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';
import { IconHome, IconMessage, IconUser } from '@tabler/icons-react';
import { FloatingNav } from '@/components/ui/floating-navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Aura - Image Processing Toolkit',
  description:
    'Aura: Comprehensive image processing toolkit designed to provide a wide range of image manipulation capabilities.',
  generator: 'Next.js',
  authors: {
    name: 'Abdallah Magdy',
    url: 'https://www.linkedin.com/in/abdallahmagdy',
  },
  keywords: [
    'image processing',
    'image manipulation',
    'image filters',
    'image effects',
    'edge detection',
    'thresholding',
    'pca',
    'face detection',
    'face recognition',
    'corner detection',
    'image matching',
    'sgementation',
  ],
  openGraph: {
    title: 'Aura - Image Processing Toolkit',
    description:
      'Aura: Comprehensive image processing toolkit designed to provide a wide range of image manipulation capabilities.',
    type: 'website',
    images: [`${process.env.NEXT_PUBLIC_APP_URL}/mockup.png`],
    url: process.env.NEXT_PUBLIC_APP_URL,
  },
  twitter: {
    title: 'Aura - Image Processing Toolkit',
    description:
      'Aura: Comprehensive image processing toolkit designed to provide a wide range of image manipulation capabilities.',
    card: 'summary_large_image',
    creator: 'a_m_s666',
    images: [`${process.env.NEXT_PUBLIC_APP_URL}/mockup.png`],
  },
  viewport:
    'minimum-scale=1, initial-scale=1, width=device-width, shrink-to-fit=no, viewport-fit=cover',
  icons: [
    { rel: 'apple-touch-icon', url: 'icons/icon-128x128.png' },
    { rel: 'icon', url: 'icons/icon-128x128.png' },
  ],
};

const navItems = [
  {
    name: 'Home',
    link: 'home',
    icon: <IconHome className="h-4 w-4 text-white" />,
  },
  {
    name: 'About',
    link: 'about',
    icon: <IconUser className="h-4 w-4 text-white" />,
  },
  {
    name: 'Features',
    link: 'features',
    icon: <IconMessage className="h-4 w-4 text-white" />,
  },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body id="home" className={cn('overflow-x-hidden', inter.className)}>
        <FloatingNav navItems={navItems} />
        {children}
      </body>
    </html>
  );
}
