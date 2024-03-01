import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

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
import { ScanIcon } from 'lucide-react';

import placeholder from '@renderer/assets/placeholder2.png';

const noiseSchema = z.object({
  type: z.enum(['sobel', 'roberts', 'perwitt', 'canny']).nullable(),
  direction: z.enum(['x', 'y', 'both']).nullable(),
  kernelSize: z.number(),
  sigma: z.number(),
  lowerThreshold: z.number(),
  upperThreshold: z.number()
});

const typesOptions = [
  { label: 'Sobel Detector', value: 'sobel' },
  { label: 'Roberts Detector', value: 'roberts' },
  { label: 'Perwitt Detector', value: 'perwitt' },
  { label: 'Canny Detector', value: 'canny' }
];

const inputs = [
  {
    value: 'sobel',
    inputs: [
      {
        label: 'Direction',
        name: 'direction',
        type: 'select',
        placeholder: 'Select direction',
        options: [
          { label: 'Horizontal', value: 'x' },
          { label: 'Vertical', value: 'y' },
          { label: 'Combined', value: 'both' }
        ]
      },
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 }
    ]
  },
  {
    value: 'roberts',
    inputs: [{ label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 }]
  },
  {
    value: 'perwitt',
    inputs: [{ label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 }]
  },
  {
    value: 'canny',
    inputs: [
      { label: 'Kernel Size', name: 'kernelSize', min: 1, max: 9, step: 2 },
      { label: 'Sigma', name: 'sigma', min: 0, max: 10, step: 0.1 },
      { label: 'Lower Threshold', name: 'lowerThreshold', min: 1, max: 200, step: 1 },
      { label: 'Upper Threshold', name: 'upperThreshold', min: 1, max: 200, step: 1 }
    ]
  }
];

function Edge() {
  const ipcRenderer = (window as any).ipcRenderer;
  const { setUploadedImageURL, setProcessedImageURL } = useGlobalState();
  // const downloadRef = useRef<HTMLAnchorElement | null>(null);

  const form = useForm<z.infer<typeof noiseSchema>>({
    resolver: zodResolver(noiseSchema),
    defaultValues: {
      direction: null,
      kernelSize: 3,
      sigma: 1,
      lowerThreshold: 30,
      upperThreshold: 150
    }
  });

  useEffect(() => {
    setUploadedImageURL(0, null);
    setProcessedImageURL(0, null);
  }, []);

  useEffect(() => {
    // ipcRenderer.on('upload:done', (event: any) => {
    //   console.log(event);
    //   if (event?.data) {
    //     console.log(event.data);
    //   }
    // });
    ipcRenderer.on('image:received', (event: any) => {
      console.log(event);
      console.log(event);
      setProcessedImageURL(0, event);
    });
  }, []);

  // const handleClick = () => {
  //   console.log('clicked');
  //   ipcRenderer.send('get:image');
  // };

  // const handleDownloadClick = () => {
  // if (processedImagesURLs && downloadRef.current) {
  //     downloadRef.current.click();
  //   }
  // };

  const onSubmit = (data: z.infer<typeof noiseSchema>) => {
    if (data.type === 'sobel' && !data.direction) {
      return;
    }
    console.log(data);
    // ipcRenderer.send('process:image', data);
  };

  return (
    <div>
      <Heading
        title="Edge Detection"
        description="Detect edges in an image using various algorithms."
        icon={ScanIcon}
        iconColor="text-orange-700"
        bgColor="bg-orange-700/10"
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
                      <Label htmlFor="noiseType">Edge Detector</Label>
                      <Select onValueChange={field.onChange}>
                        <FormControl id="noiseType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select detector algorithm" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Detectors</SelectLabel>

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

                <div className="flex flex-wrap gap-2">
                  {inputs.find((input) => input.value === form.watch('type')) &&
                    inputs
                      .find((input) => input.value === form.watch('type'))
                      ?.inputs.map((input) => {
                        if (input.type === 'select') {
                          return (
                            <FormField
                              key={input.name}
                              name={input.name}
                              render={({ field }) => (
                                <FormItem className="w-[200px]">
                                  <Label htmlFor={input.name}>{input.label}</Label>
                                  <Select value={field.value ?? ''} onValueChange={field.onChange}>
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
              <Button type="submit">Apply Filter</Button>
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

export const Route = createLazyFileRoute('/edge')({
  component: Edge
});
