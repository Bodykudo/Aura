import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { PieChart } from 'lucide-react';

function Segmentation() {
  return (
    <div>
      <Heading
        title="Segmentation"
        description="Apply segmentation to an image using several algorithms."
        icon={PieChart}
        iconColor="text-red-800"
        bgColor="bg-red-800/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/segmentation')({
  component: Segmentation
});
