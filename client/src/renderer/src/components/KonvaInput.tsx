import { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Circle, Image as KonvaImage } from 'react-konva';

import { Button } from './ui/button';
import useGlobalState from '@renderer/hooks/useGlobalState';

interface KonvaInputProps {
  imageUrl: string;
  setCenterX: (x: number) => void;
  setCenterY: (y: number) => void;
  setRadius: (radius: number) => void;
}

export default function KonvaInput({
  imageUrl,
  setCenterX,
  setCenterY,
  setRadius
}: KonvaInputProps) {
  const { setUploadedImageURL, setFileId, setProcessedImageURL } = useGlobalState();

  const [circle, setCircle] = useState<{
    x: number;
    y: number;
    radius: number;
  } | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const parentRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

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

  const handleStageMouseDown = () => {
    const stage = stageRef.current!.getStage();
    const scale = stage.scaleX();
    const position = stage.getPointerPosition()!;
    const x = (position.x - layerRef.current!.x()) / scale;
    const y = (position.y - layerRef.current!.y()) / scale;

    setCircle({
      x,
      y,
      radius: 0
    });
    isDragging.current = true;
  };

  const handleStageMouseMove = () => {
    if (isDragging.current && circle) {
      const stage = stageRef.current!.getStage();
      const scale = stage.scaleX();
      const position = stage.getPointerPosition()!;
      const x = (position.x - layerRef.current!.x()) / scale;
      const y = (position.y - layerRef.current!.y()) / scale;

      const newRadius = Math.sqrt(Math.pow(x - circle.x, 2) + Math.pow(y - circle.y, 2));

      setCircle({
        ...circle,
        radius: newRadius
      });
    }
  };

  const handleStageMouseUp = () => {
    isDragging.current = false;
    if (circle) {
      const newCircle = {
        ...circle,
        radius: Math.max(circle.radius, 10)
      };
      setCircle(newCircle);

      let imageCenterX = (circle.x / imageSize.width) * image!.width;
      let imageCenterY = (circle.y / imageSize.height) * image!.height;
      const imageRadius = (circle.radius / imageSize.width) * image!.width;

      imageCenterX =
        imageCenterX < 0 ? 0 : imageCenterX > image!.width ? image!.width : imageCenterX;
      imageCenterY =
        imageCenterY < 0 ? 0 : imageCenterY > image!.height ? image!.height : imageCenterY;

      setCenterX(imageCenterX);
      setCenterY(imageCenterY);
      setRadius(imageRadius);
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
          onMouseDown={handleStageMouseDown}
          onMouseMove={handleStageMouseMove}
          onMouseUp={handleStageMouseUp}
        >
          <Layer ref={layerRef}>
            {image && (
              <KonvaImage image={image} width={imageSize.width} height={imageSize.height} />
            )}
            {circle && (
              <Circle
                x={circle.x}
                y={circle.y}
                radius={circle.radius}
                stroke="red"
                strokeWidth={2}
                dash={[5, 5]}
              />
            )}
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
