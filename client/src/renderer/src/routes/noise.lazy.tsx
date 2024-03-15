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
import { AudioLines } from 'lucide-react';
import { useToast } from '@renderer/components/ui/use-toast';

const noiseSchema = z.object({
  type: z.enum(['uniform', 'gaussian', 'salt_and_pepper']).nullable(),
  noiseValue: z.number(),
  mean: z.number(),
  variance: z.number(),
  saltProbability: z.number(),
  pepperProbability: z.number()
});

const typesOptions = [
  { label: 'Uniform Noise', value: 'uniform' },
  { label: 'Gaussian Noise', value: 'gaussian' },
  { label: 'Salt & Pepper Filter', value: 'salt_and_pepper' }
];

const inputs = [
  {
    value: 'uniform',
    inputs: [{ label: 'Noise Value', name: 'noiseValue', min: 0, max: 255, step: 1 }]
  },
  {
    value: 'gaussian',
    inputs: [
      { label: 'Mean', name: 'mean', min: 0, max: 255, step: 1 },
      { label: 'Variance', name: 'variance', min: 0, max: 255, step: 1 }
    ]
  },
  {
    value: 'salt_and_pepper',
    inputs: [
      { label: 'Salt Probability', name: 'saltProbability', min: 0, max: 1, step: 0.01 },
      { label: 'Pepper Probability', name: 'pepperProbability', min: 0, max: 1, step: 0.01 }
    ]
  }
];

function Noise() {
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
      noiseValue: 50,
      mean: 20,
      variance: 5,
      saltProbability: 0.05,
      pepperProbability: 0.05
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
      type: data.type,
      noiseValue: data.noiseValue,
      mean: data.mean,
      variance: data.variance,
      saltProbability: data.saltProbability,
      pepperProbability: data.pepperProbability
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/noise/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Noise"
        description="Apply noise to an image to simulate real-world conditions."
        icon={AudioLines}
        iconColor="text-violet-500"
        bgColor="bg-violet-500/10"
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
                      <Label htmlFor="noiseType">Noise Type</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="noiseType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select noise type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Noise</SelectLabel>

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
                Apply Noise
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          <Dropzone index={0} />
          <OutputImage index={0} />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/noise')({
  component: Noise
});
