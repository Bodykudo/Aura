import { Card } from './ui/card';

import useGlobalState from '@renderer/hooks/useGlobalState';
import { Progress } from './ui/progress';
import { Button } from './ui/button';
import { Download } from 'lucide-react';
import { useEffect, useState } from 'react';

interface OutputImageProps {
  index: number;
  placeholder?: string;
  extra?: string;
}

export default function OutputImage({ index, placeholder, extra }: OutputImageProps) {
  const { processedImagesURLs, isProcessing } = useGlobalState();
  const [progressValue, setProgressValue] = useState(0);

  const startSimulateProgress = () => {
    setProgressValue(0);
    const interval = setInterval(() => {
      setProgressValue((value) => {
        if (value >= 90) {
          clearInterval(interval);
          return value;
        }
        return value + 10;
      });
    }, 500);

    return interval;
  };

  useEffect(() => {
    if (isProcessing) {
      const interval = startSimulateProgress();
      return () => clearInterval(interval);
    }
    return;
  }, [isProcessing]);

  return (
    <div className="flex-1 flex flex-col gap-2">
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

      <div className="flex items-center justify-between gap-2">
        <Button
          disabled={!processedImagesURLs[index] || isProcessing}
          onClick={() => {
            const link = document.createElement('a');
            link.href = `data:image/jpg;base64,${processedImagesURLs[index]}`;
            link.download = `output_${index + 1}.jpg`;
            link.click();
          }}
          className="w-[200px]"
        >
          <Download className="mr-2 w-5 h-5" />
          Download
        </Button>
        {extra && <span>{extra}</span>}
        {isProcessing && <Progress className="max-w-[250px] h-2" value={progressValue} />}
      </div>
    </div>
  );
}
