import { ContainerScroll } from './ui/container-scroll-animation';

import Image from 'next/image';

export default function Hero() {
  return (
    <div className="flex flex-col w-full bg-black bg-grid-white/[0.2] relative items-center justify-center">
      <div className="absolute pointer-events-none inset-0 flex items-center justify-center bg-black [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"></div>
      <ContainerScroll
        titleComponent={
          <>
            <h1 className="text-4xl font-semibold text-white">
              Image Processing Toolkit <br />
              <span className="text-4xl md:text-[6rem] font-bold mt-1 leading-none">
                Introducing{' '}
                <span className="bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 animate-gradient text-transparent bg-clip-text">
                  Aura
                </span>
              </span>
            </h1>
          </>
        }
      >
        <Image
          src={`/aura.png`}
          alt="hero"
          height={1860}
          width={3308}
          className="mx-auto rounded-2xl object-cover h-full object-left-top"
          draggable={false}
        />
      </ContainerScroll>
    </div>
  );
}
