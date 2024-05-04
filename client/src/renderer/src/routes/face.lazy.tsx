import { createLazyFileRoute } from '@tanstack/react-router';
import { ScanFace } from 'lucide-react';

import Heading from '@renderer/components/Heading';

function Face() {
  return (
    <div>
      <Heading
        title="Face Detection & Recognition"
        description="Detect and recognize faces in an image"
        icon={ScanFace}
        iconColor="text-blue-600"
        bgColor="bg-blue-500/10"
      />
      <div className="px-4 lg:px-8">
        <div className="mb-4"></div>
        <div className="flex flex-col md:flex-row gap-4 w-full"></div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/face')({
  component: Face
});
