import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { ArrowDownRightFromSquare } from 'lucide-react';

function Corner() {
  return (
    <div>
      <Heading
        title="Corner Detection"
        description="Apply corner detection to an image to detect corners using Harris and Lambda Detectors."
        icon={ArrowDownRightFromSquare}
        iconColor="text-rose-600"
        bgColor="bg-rose-600/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/corner')({
  component: Corner
});
