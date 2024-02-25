import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { BarChartBig } from 'lucide-react';

function Histogram() {
  return (
    <div>
      <Heading
        title="Histogram, Normalization, & Equalization"
        description="View and manipulate the histogram of an image, and apply normalization and equalization."
        icon={BarChartBig}
        iconColor="text-emerald-500"
        bgColor="bg-emerald-500/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/histogram')({
  component: Histogram
});
