import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { Wand2 } from 'lucide-react';

function Hough() {
  return (
    <div>
      <Heading
        title="Feature Extraction"
        description="Apply Hough to an image to detect lines and circles."
        icon={Wand2}
        iconColor="text-amber-500"
        bgColor="bg-amber-500/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/hough')({
  component: Hough
});
