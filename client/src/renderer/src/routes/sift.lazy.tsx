import Heading from '@renderer/components/Heading';
import { createLazyFileRoute } from '@tanstack/react-router';
import { ClipboardX } from 'lucide-react';

function SIFT() {
  return (
    <div>
      <Heading
        title="SIFT"
        description="Apply SIFT to an image to detect keypoints and match features."
        icon={ClipboardX}
        iconColor="text-teal-600"
        bgColor="bg-teal-600/10"
      />
      <div className="px-4 lg:px-8"></div>
    </div>
  );
}

export const Route = createLazyFileRoute('/sift')({
  component: SIFT
});
