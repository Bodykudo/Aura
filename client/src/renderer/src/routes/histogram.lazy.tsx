import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { BarChartBig } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
import HistogramChart from '@renderer/components/HistogramChart';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
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

import placeholder from '@renderer/assets/placeholder3.png';

const noiseSchema = z.object({
  type: z.enum(['grayscale', 'normalization', 'equalization']).nullable(),
  minWidth: z.number(),
  maxWidth: z.number()
});

const typesOptions = [
  { label: 'Grayscale', value: 'grayscale' },
  { label: 'Normalization', value: 'normalization' },
  { label: 'Equalization', value: 'equalization' }
];

function Histogram() {
  const ipcRenderer = (window as any).ipcRenderer;
  const {
    filesIds,
    setFileId,
    setUploadedImageURL,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing
  } = useGlobalState();

  const form = useForm<z.infer<typeof noiseSchema>>({
    resolver: zodResolver(noiseSchema),
    defaultValues: {
      minWidth: 120,
      maxWidth: 200
    }
  });

  const [originalHistogram, setOriginalHistogram] = useState([]);
  const [originalCdf, setOriginalCdf] = useState([]);

  const [transformedHistogram, setTransformedHistogram] = useState([]);
  const [transformedCdf, setTransformedCdf] = useState([]);

  const { toast } = useToast();

  useEffect(() => {
    setIsProcessing(false);
    setFileId(0, null);
    setUploadedImageURL(0, null);
    setProcessedImageURL(0, null);
  }, []);

  useEffect(() => {
    const imageReceivedListener = (event: any) => {
      console.log(event.data);
      if (event.data.image) {
        setProcessedImageURL(0, event.data.image);
      }
      if (event.data.histogram) {
        const original = event.data.histogram.original;
        if (original) {
          setOriginalHistogram(original.histogram);
          setOriginalCdf(original.cdf);
        }

        const transformed = event.data.histogram.transformed;
        if (transformed) {
          setTransformedHistogram(transformed.histogram);
          setTransformedCdf(transformed.cdf);
        }
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
        description: "Your image couldn't be processed, please try again.",
        variant: 'destructive'
      });
      setIsProcessing(false);
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  const onSubmit = (data: z.infer<typeof noiseSchema>) => {
    const body = {
      type: data.type
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/histogram/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Histogram, Normalization, & Equalization"
        description="View and manipulate the histogram of an image, and apply normalization and equalization."
        icon={BarChartBig}
        iconColor="text-emerald-500"
        bgColor="bg-emerald-500/10"
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
                      <Label htmlFor="transformationType">Transform Image</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="transformationType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select transformation type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Transformation</SelectLabel>
                            {typesOptions.map((option) => (
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
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Transform Image
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          <Dropzone index={0} />
          <OutputImage index={0} placeholder={placeholder} />
        </div>
        {originalHistogram.length > 0 && (
          <div className="flex flex-col md:flex-row gap-4 w-full mt-8 mb-4">
            <div className="flex flex-1 flex-col items-center gap-2">
              <h2 className="text-xl font-bold">Origina Histogram</h2>
              <HistogramChart data={originalHistogram} />
            </div>
            <div className="flex flex-1 flex-col items-center gap-2">
              <h2 className="text-xl font-bold">Transformed Histogram</h2>
              <HistogramChart data={transformedHistogram} />
            </div>
          </div>
        )}
        {transformedHistogram.length > 0 && (
          <div className="flex flex-col md:flex-row gap-4 w-full mt-8 mb-4">
            <div className="flex flex-1 flex-col items-center gap-2">
              <h2 className="text-xl font-bold">Original CDF</h2>
              <HistogramChart data={originalCdf} />
            </div>
            <div className="flex flex-1 flex-col items-center gap-2">
              <h2 className="text-xl font-bold">Transformed CDF</h2>
              <HistogramChart data={transformedCdf} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/histogram')({
  component: Histogram
});
