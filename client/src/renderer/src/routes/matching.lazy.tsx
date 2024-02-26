import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { BringToFront } from 'lucide-react';

function Matching() {
  return (
    <div>
      <Heading
        title="Image Matching"
        description="Apply image matching to an image using several algorithms."
        icon={BringToFront}
        iconColor="text-sky-400"
        bgColor="bg-sky-400/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/matching')({
  component: Matching
});
