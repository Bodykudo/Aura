import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PieChart } from 'lucide-react';

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
import SeededInput from '@renderer/components/SeededInput';

const filtersSchema = z.object({
  type: z.enum(['kmeans', 'meanShift', 'agglomerative', 'regionGrowing']).nullable(),
  k: z.number(),
  maxIterations: z.number(),
  windowSize: z.number(),
  threshold: z.number(),
  clustersNumber: z.number(),
  colorThreshold: z.number()
});

const filtersOptions = [
  { label: 'K-Mean Clustering', value: 'kmeans' },
  { label: 'Mean Shift Segmentation', value: 'meanShift' },
  { label: 'Agglomerative Segmentation', value: 'agglomerative' },
  { label: 'Region Growing', value: 'regionGrowing' }
];

const inputs = [
  {
    value: 'kmeans',
    inputs: [
      { label: 'K', name: 'k', min: 1, max: 13, step: 2 },
      { label: 'Max Iterations', name: 'maxIterations', min: 1, max: 300, step: 1 }
    ]
  },
  {
    value: 'meanShift',
    inputs: [
      { label: 'Window Size', name: 'windowSize', min: 1, max: 100, step: 1 },
      { label: 'Threshold', name: 'threshold', min: 1, max: 100, step: 1 }
    ]
  },
  {
    value: 'agglomerative',
    inputs: [
      { label: 'Clusters Number', name: 'clustersNumber', min: 1, max: 13, step: 2 },
      { label: 'Color Threshold', name: 'colorThreshold', min: 1, max: 50, step: 1 }
    ]
  },
  {
    value: 'regionGrowing',
    inputs: [
      { label: 'Threshold', name: 'threshold', min: 1, max: 250, step: 1 },
    ]
  }
];

function Segmentation() {
  const ipcRenderer = (window as any).ipcRenderer;

  const {
    filesIds,
    uploadedImagesURLs,
    setFileId,
    setUploadedImageURL,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing
  } = useGlobalState();

  const [seededPoints, setSeededPoints] = useState<{ x: number; y: number }[]>([]);

  const form = useForm<z.infer<typeof filtersSchema>>({
    resolver: zodResolver(filtersSchema),
    defaultValues: {
      k: 5,
      maxIterations: 100,
      windowSize: 30,
      threshold: 100,
      clustersNumber: 7,
      colorThreshold: 5
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
        description: "Your image couldn't be segmented, please try again later.",
        variant: 'destructive'
      });
      setIsProcessing(false);
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  const onSubmit = (data: z.infer<typeof filtersSchema>) => {
    const body = {
      type: data.type,
      k: data.k,
      maxIterations: data.maxIterations,
      windowSize: data.windowSize,
      threshold: data.threshold,
      clustersNumber: data.clustersNumber,
      colorThreshold: data.colorThreshold,
      seedPoints: seededPoints
    };

    console.log(body);

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/segmentation/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Segmentation"
        description="Apply segmentation to an image using several algorithms."
        icon={PieChart}
        iconColor="text-pink-700"
        bgColor="bg-pink-700/10"
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
                      <Label htmlFor="segmentationMethod">Segmentation Method</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="segmentationMethod">
                          <SelectTrigger>
                            <SelectValue placeholder="Select a method" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Method</SelectLabel>
                            {filtersOptions.map((option) => (
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
                Segment Image
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          {uploadedImagesURLs[0] && form.watch('type') === 'regionGrowing' ? (
            <SeededInput
              imageUrl={uploadedImagesURLs[0]}
              dots={seededPoints}
              setDots={setSeededPoints}
            />
          ) : (
            <Dropzone index={0} />
          )}
          <OutputImage index={0} placeholder={placeholder} />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/filters')({
  component: Segmentation
});
