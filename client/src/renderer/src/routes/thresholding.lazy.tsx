import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { Columns2 } from 'lucide-react';

function Thresholding() {
  return (
    <div>
      <Heading
        title="Thresholding"
        description="Apply thresholding to an image to segment it into regions."
        icon={Columns2}
        iconColor="text-green-700"
        bgColor="bg-green-700/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/thresholding')({
  component: Thresholding
});
