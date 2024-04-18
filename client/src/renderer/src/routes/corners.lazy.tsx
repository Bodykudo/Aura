import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { ArrowDownRightFromSquare } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
import { Input } from '@renderer/components/ui/input';
import { Label } from '@renderer/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@renderer/components/ui/select';
import { Button } from '@renderer/components/ui/button';

import useGlobalState from '@renderer/hooks/useGlobalState';
import { useToast } from '@renderer/components/ui/use-toast';

import placeholder from '@renderer/assets/placeholder.png';

const cornersSchema = z.object({
  type: z.enum(['harris', 'lambda', 'both']).nullable(),
  blockSize: z.number(),
  kernelSize: z.number(),
  k: z.number(),
  threshold: z.number(),
  maxCorners: z.number(),
  qualityLevel: z.number(),
  minDistance: z.number()
});

const cornersOptions = [
  { label: 'Harris Corner Detector', value: 'harris' },
  { label: 'Lambda Corner Detector', value: 'lambda' },
  { label: 'Harris & Lambda Hybrid Detector', value: 'both' }
];

const inputs = [
  {
    value: 'harris',
    inputs: [
      { label: 'Block Size', name: 'blockSize', min: 1, max: 30, step: 1 },
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 21, step: 2 },
      { label: 'K', name: 'k', min: 0, max: 1, step: 0.01 },
      { label: 'Threshold', name: 'threshold', min: 0, max: 1, step: 0.01 }
    ]
  },
  {
    value: 'lambda',
    inputs: [
      { label: 'Max Corners', name: 'maxCorners', min: 1, max: 100, step: 1 },
      { label: 'Quality Level', name: 'qualityLevel', min: 0, max: 1, step: 0.01 },
      { label: 'Min Distance', name: 'minDistance', min: 1, max: 1000, step: 1 }
    ]
  },
  {
    value: 'both',
    inputs: [
      { label: 'Block Size', name: 'blockSize', min: 1, max: 30, step: 1 },
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 21, step: 2 },
      { label: 'K', name: 'k', min: 0, max: 1, step: 0.01 },
      { label: 'Threshold', name: 'threshold', min: 0, max: 1, step: 0.01 },
      { label: 'Max Corners', name: 'maxCorners', min: 1, max: 100, step: 1 },
      { label: 'Quality Level', name: 'qualityLevel', min: 0, max: 1, step: 0.01 },
      { label: 'Min Distance', name: 'minDistance', min: 1, max: 1000, step: 1 }
    ]
  }
];

function Corners() {
  const ipcRenderer = (window as any).ipcRenderer;

  const {
    filesIds,
    setFileId,
    setUploadedImageURL,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing
  } = useGlobalState();

  const [elapsedTime, setElapsedTime] = useState<number | null>(null);

  const form = useForm<z.infer<typeof cornersSchema>>({
    resolver: zodResolver(cornersSchema),
    defaultValues: {
      blockSize: 2,
      kernelSize: 3,
      k: 0.04,
      threshold: 0.01,
      maxCorners: 25,
      qualityLevel: 0.01,
      minDistance: 10
    }
  });

  const { toast } = useToast();

  useEffect(() => {
    setIsProcessing(false);
    setFileId(0, null);
    setUploadedImageURL(0, null);
    setProcessedImageURL(0, null);
  }, []);

  useEffect(() => {
    const imageReceivedListener = (event: any) => {
      if (event.data.image) {
        setProcessedImageURL(0, event.data.image);
      }
      if (event.data.time) {
        setElapsedTime(event.data.time);
      }
      setIsProcessing(false);
    };
    ipcRenderer.on('image:received', imageReceivedListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  useEffect(() => {
    const imageErrorListener = () => {
      toast({
        title: 'Something went wrong',
        description: "Your image couldn't be processed, please try again later.",
        variant: 'destructive'
      });
      setIsProcessing(false);
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  const onSubmit = (data: z.infer<typeof cornersSchema>) => {
    const body = {
      type: data.type,
      blockSize: data.blockSize,
      kernelSize: data.kernelSize,
      k: data.k,
      threshold: data.threshold,
      maxCorners: data.maxCorners,
      qualityLevel: data.qualityLevel,
      minDistance: data.minDistance
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/corners/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Corners Detection"
        description="Apply corner detection to an image to detect corners using Harris and Lambda Detectors."
        icon={ArrowDownRightFromSquare}
        iconColor="text-rose-600"
        bgColor="bg-rose-600/10"
      />
      <div className="px-4 lg:px-8">
        <div className="mb-4">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="flex flex-wrap gap-4 justify-between items-end"
            >
              <div className="flex flex-wrap gap-2">
                <FormField
                  control={form.control}
                  name="type"
                  render={({ field }) => (
                    <FormItem className="w-[250px] mr-4">
                      <Label htmlFor="type">Detection Algorithm</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="type">
                          <SelectTrigger>
                            <SelectValue placeholder="Select detection algorithm" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Algorithms</SelectLabel>
                            {cornersOptions.map((option) => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />

                <div className="flex flex-wrap gap-2">
                  {inputs.find((input) => input.value === form.watch('type')) &&
                    inputs
                      .find((input) => input.value === form.watch('type'))
                      ?.inputs.map((input) => {
                        return (
                          <FormField
                            key={input.label}
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
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Detect Corners
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          <Dropzone index={0} />
          <OutputImage
            index={0}
            extra={elapsedTime ? `Elapsed time: ${elapsedTime.toFixed(2)}s` : undefined}
            placeholder={placeholder}
          />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/corners')({
  component: Corners
});
