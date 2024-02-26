import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { CircleDotDashed } from 'lucide-react';

function Contours() {
  return (
    <div>
      <Heading
        title="Active Contours"
        description="Apply active contours to an image to detect edges."
        icon={CircleDotDashed}
        iconColor="text-cyan-700"
        bgColor="bg-cyan-700/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/contours')({
  component: Contours
});
