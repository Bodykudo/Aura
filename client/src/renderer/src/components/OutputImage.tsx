import useGlobalState from '@renderer/hooks/useGlobalState';
import { Card } from './ui/card';

export default function OutputImage() {
  const { processedImageURL } = useGlobalState();
  return (
    <div className="flex-1">
      <Card className="w-full h-72 lg:h-80 xl:h-96 p-2">
        <div
          className="h-full w-full bg-center bg-no-repeat bg-contain"
          style={
            processedImageURL
              ? {
                  backgroundImage: `url(${processedImageURL})`
                }
              : {}
          }
        />
      </Card>
    </div>
  );
}
