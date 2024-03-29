import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { SketchPicker } from 'react-color';
import { Wand2 } from 'lucide-react';

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
import { Popover, PopoverContent, PopoverTrigger } from '@renderer/components/ui/popover';

import useGlobalState from '@renderer/hooks/useGlobalState';
import { useToast } from '@renderer/components/ui/use-toast';

import placeholder from '@renderer/assets/placeholder3.png';

const houghSchema = z.object({
  type: z.enum(['lines', 'circles','ellipses']).nullable(),
  rho: z.number(),
  theta: z.number(),
  threshold: z.number(),
  minRadius: z.number(),
  maxRadius: z.number(),
  color: z.string(),
  minMajoraxis: z.number()
});

const houghOptions = [
  { label: 'Lines Detection', value: 'lines' },
  { label: 'Circles Detection', value: 'circles' },{ label: 'Ellipses Detection', value: 'ellipses' }
];

const inputs = [
  {
    value: 'lines',
    inputs: [
      { label: 'Rho', name: 'rho', min: 1, max: 10, step: 1 },
      { label: 'Theta', name: 'theta', min: 1, max: 360, step: 1 },
      { label: 'Threshold', name: 'threshold', min: 1, max: 100, step: 1 }
    ]
  },
  {
    value: 'circles',
    inputs: [
      { label: 'Threshold', name: 'threshold', min: 0, max: 100, step: 1 },
      { label: 'Min Radius', name: 'minRadius', min: 0, max: 255, step: 1 },
      { label: 'Max Radius', name: 'maxRadius', min: 0, max: 512, step: 1 }
    ]
  },
  {
    value: 'ellipses',
    inputs: [
      { label: 'Threshold', name: 'threshold', min: 0, max: 100, step: 1 },
      { label: 'Min Major Axis', name: 'minMajoraxis', min: 0, max: 255, step: 1 }
    ]
  }

];

function Hough() {
  const ipcRenderer = (window as any).ipcRenderer;

  const {
    filesIds,
    setFileId,
    setUploadedImageURL,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing
  } = useGlobalState();

  const form = useForm({
    resolver: zodResolver(houghSchema),
    defaultValues: {
      type: null,
      rho: 1,
      theta: 180,
      threshold: 100,
      minRadius: 50,
      maxRadius: 300,
      color: '#ff0023',
      minMajoraxis:100
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
        description: "Thresholding couldn't be applied on your image, please try again later.",
        variant: 'destructive'
      });
      setIsProcessing(false);
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  const onSubmit = (data: z.infer<typeof houghSchema>) => {
    const body = {
      type: data.type,
      threshold: data.threshold,
      rho: data.rho,
      theta: data.theta,
      minRadius: data.minRadius,
      maxRadius: data.maxRadius,
      color: data.color,
      minMajoraxis: data.minMajoraxis
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/hough/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Object Detection"
        description="Apply Hough to an image to detect lines, circles, and ellipses."
        icon={Wand2}
        iconColor="text-amber-500"
        bgColor="bg-amber-500/10"
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
                      <Label htmlFor="type">Object Type</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="type">
                          <SelectTrigger>
                            <SelectValue placeholder="Select object type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Thresholding</SelectLabel>
                            {houghOptions.map((option) => (
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

                  {form.watch('type') && (
                    <FormField
                      name="color"
                      render={({ field }) => (
                        <FormItem className="w-[150px]">
                          <Label htmlFor="color">Color</Label>
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button
                                variant="outline"
                                id="color"
                                className="flex items-center gap-2 hover:bg-background"
                              >
                                <div
                                  className="w-6 h-6 rounded-sm border"
                                  style={{ backgroundColor: field.value }}
                                />
                                Pick a color
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent className="flex items-center justify-center bg-none border-0 shadow-0 w-fit h-fit p-0 bg-transparent inset-0">
                              <FormControl className="p-2">
                                <SketchPicker
                                  color={field.value}
                                  onChangeComplete={(color) => field.onChange(color.hex)}
                                />
                              </FormControl>
                            </PopoverContent>
                          </Popover>
                        </FormItem>
                      )}
                    />
                  )}
                </div>
              </div>
              <Button disabled={!filesIds[0] || isProcessing} className="capitalize" type="submit">
                Detect {form.watch('type') ? form.watch('type') : 'Objects'}
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

export const Route = createLazyFileRoute('/hough')({
  component: Hough
});
