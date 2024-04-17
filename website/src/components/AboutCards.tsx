import AboutCard from './AboutCard';
import { CanvasRevealEffect } from './ui/canvas-reveal-effect';

const AceternityIcon = () => {
  return (
    <svg
      width="66"
      height="65"
      viewBox="0 0 66 65"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-10 w-10 text-white group-hover/canvas-card:text-white "
    >
      <path
        d="M8 8.05571C8 8.05571 54.9009 18.1782 57.8687 30.062C60.8365 41.9458 9.05432 57.4696 9.05432 57.4696"
        stroke="currentColor"
        strokeWidth="15"
        strokeMiterlimit="3.86874"
        strokeLinecap="round"
        style={{ mixBlendMode: 'darken' }}
      />
    </svg>
  );
};

const aboutContent = [
  {
    title: 'Cutting-Edge Technology',
    description:
      'Aura utilizes the latest advancements in image processing technology to provide a robust and efficient toolkit. Experience the power of modern tech with Aura.',
    icon: <AceternityIcon />,
    animationSpeed: 5.1,
    containerClassName: 'bg-emerald-900',
  },
  {
    title: 'Educational Focus',
    description:
      'Aura is designed with students in mind. Our toolkit simplifies complex image processing techniques, making them accessible for learning and exploration.',
    icon: <AceternityIcon />,
    animationSpeed: 3,
    containerClassName: 'bg-black',
    colors: [
      [236, 72, 153],
      [232, 121, 249],
    ],
    dotSize: 2,
  },
  {
    title: 'Versatile Applications',
    description:
      'From edge detection to segmentation, Aura offers a wide range of image processing tools. Discover the potential of image processing in various fields with Aura.',
    icon: <AceternityIcon />,
    animationSpeed: 3,
    containerClassName: 'bg-sky-600',
    colors: [[125, 211, 252]],
  },
];

export default function AboutCards() {
  return (
    <div className="pb-10 flex flex-col lg:flex-row items-center justify-center w-full gap-4 mx-auto px-4">
      {aboutContent.map((content, index) => (
        <AboutCard
          key={index}
          title={content.title}
          description={content.description}
          icon={content.icon}
        >
          <CanvasRevealEffect
            animationSpeed={content.animationSpeed}
            containerClassName={content.containerClassName}
            colors={content.colors}
            dotSize={content.dotSize}
          />
          {index === 1 && (
            <div className="absolute inset-0 [mask-image:radial-gradient(400px_at_center,white,transparent)] bg-black/90" />
          )}
        </AboutCard>
      ))}
    </div>
  );
}
