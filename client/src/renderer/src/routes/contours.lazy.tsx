import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { CircleDotDashed } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import KonvaInput from '@renderer/components/KonvaInput';
import OutputImage from '@renderer/components/OutputImage';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
import { Input } from '@renderer/components/ui/input';
import { Label } from '@renderer/components/ui/label';
import { Button } from '@renderer/components/ui/button';

import useGlobalState from '@renderer/hooks/useGlobalState';
import useHandleProcessing from '@renderer/hooks/useHandleProcessing';

import placeholder from '@renderer/assets/placeholder2.png';

const contoursSchema = z.object({
  iterations: z.number(),
  points: z.number(),
  windowSize: z.number(),
  alpha: z.number(),
  beta: z.number(),
  gamma: z.number(),
  centerX: z.number(),
  centerY: z.number(),
  radius: z.number()
});

const inputs = [
  { label: 'Iterations', name: 'iterations', min: 10, max: 5000, step: 10 },
  { label: 'Points', name: 'points', min: 10, max: 1000, step: 5 },
  { label: 'Window Size', name: 'windowSize', min: 3, max: 21, step: 2 },
  { label: 'α', name: 'alpha', min: 0.1, max: 15, step: 0.1 },
  { label: 'β', name: 'beta', min: 0.1, max: 15, step: 0.1 },
  { label: 'Γ', name: 'gamma', min: 0.1, max: 15, step: 0.1 }
];

function Contours() {
  const ipcRenderer = window.ipcRenderer;

  const {
    filesIds,
    uploadedImagesURLs,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing,
    reset
  } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Acitve contours couldn't be applied to your image. Please try again."
  });

  const [perimeter, setPerimeter] = useState<number | null>(null);
  const [area, setArea] = useState<number | null>(null);

  const form = useForm({
    resolver: zodResolver(contoursSchema),
    defaultValues: {
      iterations: 300,
      points: 70,
      windowSize: 11,
      alpha: 10,
      beta: 3,
      gamma: 1,
      centerX: 0,
      centerY: 0,
      radius: 0
    }
  });

  const setCenterX = (x: number) => form.setValue('centerX', x);
  const setCenterY = (y: number) => form.setValue('centerY', y);
  const setRadius = (r: number) => form.setValue('radius', r);

  useEffect(() => {
    reset();
  }, []);

  useEffect(() => {
    if (data) {
      if (data.image) {
        setProcessedImageURL(0, data.image);
      }
      if (data.perimeter) {
        const outputPerimeter: number = data.perimeter;
        setPerimeter(outputPerimeter);
      }
      if (data.area) {
        const outputArea: number = data.area;
        setArea(outputArea);
      }
    }
  }, [data]);

  const onSubmit = (data: z.infer<typeof contoursSchema>) => {
    const body = {
      centerX: data.centerX,
      centerY: data.centerY,
      radius: data.radius,
      iterations: data.iterations,
      points: data.points,
      windowSize: data.windowSize,
      alpha: data.alpha,
      beta: data.beta,
      gamma: data.gamma
    };

    setPerimeter(null);
    setArea(null);
    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/active-contour/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Active Contours"
        description="Apply active contours to an image to detect edges."
        icon={CircleDotDashed}
        iconColor="text-cyan-700"
        bgColor="bg-cyan-700/10"
      />
      <div className="px-4 lg:px-8">
        <div className="mb-4">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="flex flex-wrap gap-4 justify-between items-end"
            >
              <div className="flex flex-wrap gap-2">
                {inputs.map((input) => {
                  return (
                    <FormField
                      key={input.name}
                      name={input.name}
                      render={({ field }) => (
                        <FormItem className="w-[150px]">
                          <Label htmlFor={input.name}>{input.label}</Label>
                          <FormControl className="p-2">
                            <Input
                              type="number"
                              disabled={isProcessing}
                              id={input.name}
                              min={input.min}
                              max={input.max}
                              step={input.step}
                              {...field}
                              onChange={(e) => field.onChange(Number(e.target.value))}
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                  );
                })}
              </div>
              <Button disabled={!filesIds[0] || isProcessing} className="capitalize" type="submit">
                Apply Active Contours
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          {uploadedImagesURLs[0] ? (
            <KonvaInput
              imageUrl={uploadedImagesURLs[0]}
              setCenterX={setCenterX}
              setCenterY={setCenterY}
              setRadius={setRadius}
            />
          ) : (
            <Dropzone index={0} />
          )}
          <OutputImage
            index={0}
            extra={
              perimeter && area
                ? `Perimeter: ${perimeter.toFixed(2)}, Area: ${area.toFixed(2)}`
                : undefined
            }
            placeholder={placeholder}
          />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/contours')({
  component: Contours
});
