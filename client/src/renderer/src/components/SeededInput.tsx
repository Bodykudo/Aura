import { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle } from 'react-konva';

import { Button } from './ui/button';
import useGlobalState from '@renderer/hooks/useGlobalState';

interface SeededInputProps {
  imageUrl: string;
  dots: { x: number; y: number }[];
  setDots: (dots: { x: number; y: number }[]) => void;
}

export default function SeededInput({ imageUrl, dots, setDots }: SeededInputProps) {
  const { setUploadedImageURL, setFileId, setProcessedImageURL } = useGlobalState();

  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const parentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const image = new window.Image();

    image.onload = () => {
      setImage(image);
      if (parentRef.current) {
        const parentWidth = parentRef.current.offsetWidth;
        const parentHeight = parentRef.current.offsetHeight;
        const scale = Math.min(parentWidth / image.width, parentHeight / image.height);
        setImageSize({ width: image.width * scale, height: image.height * scale });

        if (layerRef.current) {
          layerRef.current.x((parentWidth - image.width * scale) / 2);
          layerRef.current.y((parentHeight - image.height * scale) / 2);
        }
      }
    };

    image.src = imageUrl;
  }, [imageUrl]);

  const handleOnClick = (event: any) => {
    if (event.evt.ctrlKey) {
      const stage = stageRef.current!.getStage();
      const scale = stage.scaleX();
      const position = stage.getPointerPosition()!;
      const x = (position.x - layerRef.current!.x()) / scale;
      const y = (position.y - layerRef.current!.y()) / scale;

      setDots([...dots, { x, y }]);
    }
  };

  return (
    <div className="flex flex-col gap-2 flex-1">
      <div
        ref={parentRef}
        className="relative flex-1 overflow-hidden rounded-md outline-dashed outline-2 outline-border"
      >
        <Stage
          width={parentRef.current ? parentRef.current.offsetWidth : 0}
          height={parentRef.current ? parentRef.current.offsetHeight : 0}
          ref={stageRef}
          onClick={handleOnClick}
        >
          <Layer ref={layerRef}>
            {image && (
              <KonvaImage image={image} width={imageSize.width} height={imageSize.height} />
            )}
            {dots.map((dot, i) => (
              <Circle
                key={i}
                x={dot.x}
                y={dot.y}
                radius={5}
                onClick={() => {
                  setDots(dots.filter((_, index) => index !== i));
                }}
                fill="red"
              />
            ))}
          </Layer>
        </Stage>
      </div>
      <Button
        onClick={() => {
          setUploadedImageURL(0, null);
          setFileId(0, null);
          setProcessedImageURL(0, null);
        }}
        className="w-[200px]"
      >
        Reset
      </Button>
    </div>
  );
}
