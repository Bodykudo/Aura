import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { BringToFront } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
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
import useHandleProcessing from '@renderer/hooks/useHandleProcessing';

import placeholder from '@renderer/assets/placeholder.png';

const matchingSchema = z.object({
  type: z.enum(['ssd', 'ncc']).nullable()
});

const matchingOptions = [
  { label: 'Sum of Squared Difference', value: 'ssd' },
  { label: 'Normalized Cross Correlation', value: 'ncc' }
];

function Matching() {
  const ipcRenderer = (window as any).ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Your images couldn't be matched. Please try again."
  });

  const [elapsedTime, setElapsedTime] = useState<number | null>(null);

  const form = useForm<z.infer<typeof matchingSchema>>({
    resolver: zodResolver(matchingSchema)
  });

  useEffect(() => {
    reset();
  }, []);

  useEffect(() => {
    if (data) {
      if (data.image) {
        setProcessedImageURL(0, data.image);
      }
      if (data.time) {
        setElapsedTime(data.time);
      }
    }
  }, [data]);

  const onSubmit = (data: z.infer<typeof matchingSchema>) => {
    const body = {
      type: data.type,
      originalImageId: filesIds[0],
      templateImageId: filesIds[1]
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: '/api/matching'
    });
  };

  return (
    <div>
      <Heading
        title="Image Matching"
        description="Apply image matching to an image using several algorithms."
        icon={BringToFront}
        iconColor="text-sky-400"
        bgColor="bg-sky-400/10"
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
                      <Label htmlFor="type">Matching Algorithm</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="type">
                          <SelectTrigger>
                            <SelectValue placeholder="Select matching algorithm" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Algorithms</SelectLabel>
                            {matchingOptions.map((option) => (
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
                Match Images
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col md:flex-row gap-4 w-full">
            <div className="flex flex-col gap-1 w-full">
              <p className="font-medium text-xl">Image</p>
              <Dropzone index={0} />
            </div>
            <div className="flex flex-col gap-1 w-full">
              <p className="font-medium text-xl">Template</p>
              <Dropzone index={1} />
            </div>
          </div>
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

export const Route = createLazyFileRoute('/matching')({
  component: Matching
});
