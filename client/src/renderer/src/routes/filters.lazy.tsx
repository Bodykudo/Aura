import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { FilterX } from 'lucide-react';

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

const filtersSchema = z.object({
  type: z.enum(['average', 'gaussian', 'median', 'low', 'high']).nullable(),
  kernelSize: z.number(),
  sigma: z.number(),
  radius: z.number()
});

const filtersOptions = [
  { label: 'Average Filter', value: 'average' },
  { label: 'Gaussian Filter', value: 'gaussian' },
  { label: 'Median Filter', value: 'median' },
  { label: 'Low Pass Filter', value: 'low' },
  { label: 'High Pass Filter', value: 'high' }
];

const inputs = [
  {
    value: 'average',
    inputs: [{ label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 }]
  },
  {
    value: 'gaussian',
    inputs: [
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 },
      { label: 'Sigma', name: 'sigma', min: 0, max: 10, step: 0.1 }
    ]
  },
  {
    value: 'median',
    inputs: [{ label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 }]
  },
  {
    value: 'low',
    inputs: [
      {
        label: 'Filter Radius',
        name: 'radius',
        min: 1,
        max: 100,
        step: 1
      }
    ]
  },
  {
    value: 'high',
    inputs: [
      {
        label: 'Filter Radius',
        name: 'radius',
        min: 1,
        max: 100,
        step: 1
      }
    ]
  }
];

function Filters() {
  const ipcRenderer = window.ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Your image couldn't be filtered. Please try again."
  });

  const form = useForm<z.infer<typeof filtersSchema>>({
    resolver: zodResolver(filtersSchema),
    defaultValues: {
      kernelSize: 3,
      sigma: 1,
      radius: 20
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

  const onSubmit = (data: z.infer<typeof filtersSchema>) => {
    const body = {
      type: data.type,
      kernelSize: data.kernelSize,
      sigma: data.sigma,
      radius: data.radius
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/filter/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Filters"
        description="Apply filters to an image to enhance or remove certain features."
        icon={FilterX}
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
                      <Label htmlFor="filterType">Filter Type</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="filterType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select a filter" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Filters</SelectLabel>

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
                Apply Filter
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

export const Route = createLazyFileRoute('/filters')({
  component: Filters
});
