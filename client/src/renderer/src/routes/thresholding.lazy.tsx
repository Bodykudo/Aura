import { useEffect, useState } from 'react';
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
import { useToast } from '@renderer/components/ui/use-toast';

import placeholder from '@renderer/assets/placeholder.png';

const thresholdingSchema = z.object({
  type: z.enum(['local', 'global']).nullable(),
  kernelSize: z.number(),
  sigma: z.number()
});

const thresholdingOptions = [
  { label: 'Local Thresholding', value: 'local' },
  { label: 'Global Thresholding', value: 'global' }
];

const inputs = [
  {
    value: 'local',
    inputs: [{ label: 'Border Radius', name: 'kernelSize', min: 1, max: 9, step: 2 }]
  },
  {
    value: 'global',
    inputs: [
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 },
      { label: 'Sigma', name: 'sigma', min: 0, max: 10, step: 0.1 }
    ]
  }
];

function Thresholding() {
  const ipcRenderer = (window as any).ipcRenderer;

  const { filesIds, setUploadedImageURL, setProcessedImageURL } = useGlobalState();
  const [isProcessing, setIsProcessing] = useState(false);

  const form = useForm<z.infer<typeof thresholdingSchema>>({
    resolver: zodResolver(thresholdingSchema),
    defaultValues: {
      kernelSize: 3,
      sigma: 1
    }
  });

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
              // onSubmit={form.handleSubmit(onSubmit)}
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
                            <SelectValue placeholder="Select a filter" />
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
                  {inputs.find((input) => input.value === form.watch('type')) &&
                    inputs
                      .find((input) => input.value === form.watch('type'))
                      ?.inputs.map((input, index) => {
                        return (
                          <FormField
                            key={index}
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
      </div>{' '}
    </div>
  );
}

export const Route = createLazyFileRoute('/thresholding')({
  component: Thresholding
});
