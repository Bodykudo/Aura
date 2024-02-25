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

const noiseSchema = z.object({
  type: z.enum(['uniform', 'gaussian', 'salt_pepper']).nullable(),
  noiseValue: z.number(),
  mean: z.number(),
  variance: z.number(),
  noiseProbability: z.number()
});

const typesOptions = [
  { label: 'Uniform Noise', value: 'uniform' },
  { label: 'Gaussian Noise', value: 'gaussian' },
  { label: 'Salt & Pepper Filter', value: 'salt_pepper' }
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
    value: 'salt_pepper',
    inputs: [{ label: 'Noise Probability', name: 'noiseProbability', min: 0, max: 1, step: 0.01 }]
  }
];

function Noise() {
  const ipcRenderer = (window as any).ipcRenderer;
  const { setProcessedImageURL } = useGlobalState();
  // const downloadRef = useRef<HTMLAnchorElement | null>(null);

  const form = useForm<z.infer<typeof noiseSchema>>({
    resolver: zodResolver(noiseSchema),
    defaultValues: {
      noiseValue: 50,
      mean: 20,
      variance: 5,
      noiseProbability: 0.07
    }
  });

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
    console.log(data);
    // ipcRenderer.send('process:image', data);
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
            <form onSubmit={form.handleSubmit(onSubmit)} className="flex justify-between items-end">
              <div className="flex gap-6">
                <FormField
                  control={form.control}
                  name="type"
                  render={({ field }) => (
                    <FormItem className="w-[250px]">
                      <Label htmlFor="noiseType">Noise Type</Label>
                      <Select
                        onValueChange={field.onChange}
                        value={field.value}
                        defaultValue={field.value}
                      >
                        <FormControl id="noiseType">
                          <SelectTrigger>
                            <SelectValue
                              placeholder="Select noise type"
                              defaultValue={field.value}
                            />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Noise Types</SelectLabel>

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

                <div className="flex gap-2">
                  {inputs.find((input) => input.value === form.watch('type')) &&
                    inputs
                      .find((input) => input.value === form.watch('type'))
                      ?.inputs.map((input, index) => {
                        return (
                          <FormField
                            key={index}
                            control={form.control}
                            name={input.name}
                            render={({ field }) => (
                              <FormItem className="w-[150px]">
                                <Label htmlFor={input.name}>{input.label}</Label>
                                <Input
                                  type="number"
                                  id={input.name}
                                  min={input.min}
                                  max={input.max}
                                  step={input.step}
                                  onChange={(e) => {
                                    field.onChange(Number(e.target.value));
                                  }}
                                  value={field.value}
                                  defaultValue={field.value}
                                />
                              </FormItem>
                            )}
                          />
                        );
                      })}
                </div>
              </div>
              <Button type="submit">Apply Noise</Button>
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
