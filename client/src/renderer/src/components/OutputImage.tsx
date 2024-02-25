import { Card } from './ui/card';

import useGlobalState from '@renderer/hooks/useGlobalState';

interface OutputImageProps {
  index: number;
}

export default function OutputImage({ index }: OutputImageProps) {
  const { processedImagesURLs } = useGlobalState();

  return (
    <div className="flex-1">
      <Card className="w-full h-72 lg:h-80 xl:h-96 p-2">
        <div
          className="h-full w-full bg-center bg-no-repeat bg-contain"
          style={
            processedImagesURLs[index]
              ? {
                  backgroundImage: `url(${processedImagesURLs[index]})`
                }
              : {}
          }
        />
      </Card>
    </div>
  );
}
