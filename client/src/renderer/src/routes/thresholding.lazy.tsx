import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Columns2 } from 'lucide-react';

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
import useHandleProcessing from '@renderer/hooks/useHandleProcessing';

import placeholder from '@renderer/assets/placeholder.png';

const thresholdingSchema = z.object({
  type: z.enum(['optimal', 'otsu', 'spectral']).nullable(),
  scope: z.enum(['global', 'local']).nullable(),
  threshold: z.number(),
  windowSize: z.number(),
  offset: z.number()
});

const thresholdingOptions = [
  { label: 'Optimal Thresholding', value: 'optimal' },
  { label: 'Otsu Thresholding', value: 'otsu' },
  { label: 'Spectral Thresholding', value: 'spectral' }
];

function Thresholding() {
  const ipcRenderer = (window as any).ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Thresholding couldn't be applied to your image. Please try again."
  });

  const form = useForm<z.infer<typeof thresholdingSchema>>({
    resolver: zodResolver(thresholdingSchema),
    defaultValues: {
      threshold: 127,
      windowSize: 11,
      offset: 0
    }
  });

  useEffect(() => {
    reset();
  }, []);

  useEffect(() => {
    if (data && data.image) {
      setProcessedImageURL(0, data.image);
    }
  }, [data]);

  const onSubmit = (data: z.infer<typeof thresholdingSchema>) => {
    const body = {
      type: data.type,
      scope: data.scope,
      windowSize: data.windowSize,
      offset: data.offset
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/thresholding/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Thresholding"
        description="Apply thresholding to an image to segment it into regions."
        icon={Columns2}
        iconColor="text-green-700"
        bgColor="bg-green-700/10"
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
                      <Label htmlFor="thresholdingType">Thrsholding Type</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="thresholdingType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select thresholding type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Thresholding</SelectLabel>
                            {thresholdingOptions.map((option) => (
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
                  {form.watch('type') && (
                    <FormField
                      control={form.control}
                      name="scope"
                      render={({ field }) => (
                        <FormItem className="w-[250px]">
                          <Label htmlFor="scope">Thrsholding Scope</Label>
                          <Select disabled={isProcessing} onValueChange={field.onChange}>
                            <FormControl id="scope">
                              <SelectTrigger>
                                <SelectValue placeholder="Select thresholding scope" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectGroup>
                                <SelectLabel>Scope</SelectLabel>
                                <SelectItem value="global">Global Thresholding</SelectItem>
                                <SelectItem value="local">Local Thresholding</SelectItem>
                              </SelectGroup>
                            </SelectContent>
                          </Select>
                        </FormItem>
                      )}
                    />
                  )}
                  {form.watch('scope') === 'local' && (
                    <>
                      <FormField
                        control={form.control}
                        name="windowSize"
                        render={({ field }) => (
                          <FormItem className="w-[150px]">
                            <Label htmlFor="windowSize">Window Size</Label>
                            <Input
                              id="windowSize"
                              type="number"
                              disabled={isProcessing}
                              min={3}
                              max={51}
                              step={2}
                              {...field}
                              onChange={(e) => field.onChange(Number(e.target.value))}
                            />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="offset"
                        render={({ field }) => (
                          <FormItem className="w-[150px]">
                            <Label htmlFor="offset">Offset</Label>
                            <Input
                              id="offset"
                              type="number"
                              disabled={isProcessing}
                              min={0}
                              max={10}
                              step={1}
                              {...field}
                              onChange={(e) => field.onChange(Number(e.target.value))}
                            />
                          </FormItem>
                        )}
                      />
                    </>
                  )}
                </div>
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Apply Thresholding
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          <Dropzone index={0} />
          <OutputImage index={0} placeholder={placeholder} />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/thresholding')({
  component: Thresholding
});
