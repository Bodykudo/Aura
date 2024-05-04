import { useEffect, useState } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { ClipboardX } from 'lucide-react';

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

const siftSchema = z.object({
  mode: z.enum(['keypoints', 'matching']).nullable(),
  sigma: z.number(),
  numIntervals: z.number(),
  assumedBlur: z.number(),
  type: z.enum(['ssd', 'ncc']),
  numMatches: z.number()
});

const modesOptions = [
  { label: 'Detect Keypoints', value: 'keypoints' },
  { label: 'Match Images Features', value: 'matching' }
];

const inputs = [
  {
    value: 'keypoints',
    inputs: [
      { label: 'Sigma', name: 'sigma', min: 0.1, max: 5, step: 0.1 },
      { label: 'Number of Intervals', name: 'numIntervals', min: 1, max: 5, step: 1 },
      { label: 'Assumed Blur', name: 'assumedBlur', min: 0.01, max: 1, step: 0.01 }
    ]
  },
  {
    value: 'matching',
    inputs: [
      {
        label: 'Matching Algorithm',
        name: 'type',
        type: 'select',
        placeholder: 'Select matching algorithm',
        options: [
          { label: 'Sum of Squared Difference', value: 'ssd' },
          { label: 'Normalized Cross Correlation', value: 'ncc' }
        ]
      },
      { label: 'Number of Matches', name: 'numMatches', min: 1, max: 500, step: 1 }
    ]
  }
];

function SIFT() {
  const ipcRenderer = window.ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Your images couldn't be processed. Please try again."
  });

  const form = useForm<z.infer<typeof siftSchema>>({
    resolver: zodResolver(siftSchema),
    defaultValues: {
      mode: 'keypoints',
      sigma: 1.6,
      numIntervals: 3,
      assumedBlur: 0.5,
      type: 'ncc',
      numMatches: 30
    }
  });

  const [elapsedTime, setElapsedTime] = useState<number | null>(null);

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

  const onSubmit = (data: z.infer<typeof siftSchema>) => {
    let body = {};
    setIsProcessing(true);

    if (data.mode === 'keypoints') {
      body = {
        sigma: data.sigma,
        numIntervals: data.numIntervals,
        assumedBlur: data.assumedBlur
      };

      ipcRenderer.send('process:image', {
        body,
        url: `/api/sift/keypoints/${filesIds[0]}`
      });
    }

    if (data.mode === 'matching') {
      const body = {
        type: data.type,
        originalImageId: filesIds[0],
        templateImageId: filesIds[1],
        numMatches: data.numMatches
      };

      ipcRenderer.send('process:image', {
        body,
        url: '/api/sift/matching'
      });
    }
  };

  return (
    <div>
      <Heading
        title="SIFT"
        description="Apply SIFT to an image to detect keypoints and match features."
        icon={ClipboardX}
        iconColor="text-teal-600"
        bgColor="bg-teal-600/10"
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
                  name="mode"
                  render={({ field }) => (
                    <FormItem className="w-[250px] mr-4">
                      <Label htmlFor="mode">SIFT Mode</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="mode">
                          <SelectTrigger>
                            <SelectValue placeholder="Select SIFT mode" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Modes</SelectLabel>

                            {modesOptions.map((option) => (
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
                  {inputs.find((input) => input.value === form.watch('mode')) &&
                    inputs
                      .find((input) => input.value === form.watch('mode'))
                      ?.inputs.map((input) => {
                        if (input.type === 'select') {
                          return (
                            <FormField
                              key={input.name}
                              name={input.name}
                              render={({ field }) => (
                                <FormItem className="w-[250px]">
                                  <Label htmlFor={input.name}>{input.label}</Label>
                                  <Select
                                    disabled={isProcessing}
                                    value={field.value ?? ''}
                                    onValueChange={field.onChange}
                                  >
                                    <FormControl id={input.name}>
                                      <SelectTrigger>
                                        <SelectValue placeholder={input.placeholder} />
                                      </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                      <SelectGroup>
                                        <SelectLabel>{input.label}</SelectLabel>

                                        {input.options?.map((option) => (
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
                          );
                        } else {
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
                        }
                      })}
                </div>
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                {form.watch('mode') === 'keypoints' ? 'Detect Keypoints' : 'Match Features'}
              </Button>
            </form>
          </Form>
        </div>
        {form.watch('mode') === 'keypoints' && (
          <div className="flex flex-col md:flex-row gap-4 w-full">
            <Dropzone index={0} />
            <OutputImage
              index={0}
              extra={elapsedTime ? `Elapsed time: ${elapsedTime.toFixed(2)}s` : undefined}
              placeholder={placeholder}
            />
          </div>
        )}
        {form.watch('mode') === 'matching' && (
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
        )}
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/sift')({
  component: SIFT
});
