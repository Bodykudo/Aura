import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { ScanIcon } from 'lucide-react';

function Edge() {
  return (
    <div>
      <Heading
        title="Edge Detection"
        description="Detect edges in an image using various algorithms."
        icon={ScanIcon}
        iconColor="text-orange-700"
        bgColor="bg-orange-700/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/edge')({
  component: Edge
});
