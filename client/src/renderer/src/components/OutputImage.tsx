import { Card } from './ui/card';

import useGlobalState from '@renderer/hooks/useGlobalState';

interface OutputImageProps {
  index: number;
  placeholder?: string;
}

export default function OutputImage({ index, placeholder }: OutputImageProps) {
  const { processedImagesURLs } = useGlobalState();

  return (
    <div className="flex-1">
      <Card className="w-full h-72 lg:h-80 xl:h-96 p-2">
        <div className="h-full w-full relative overflow-hidden">
          {processedImagesURLs[index] ? (
            <img
              src={`data:image/jpg;base64,${processedImagesURLs[index]}`}
              className="absolute h-full top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
              alt="uploaded image"
            />
          ) : (
            placeholder && (
              <img
                src={placeholder}
                className="absolute h-full top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
                alt="placeholder"
              />
            )
          )}
        </div>
      </Card>
    </div>
  );
}
