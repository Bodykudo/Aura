import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Blend } from 'lucide-react';

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
import placeholder2 from '@renderer/assets/placeholder2.png';
import placeholder3 from '@renderer/assets/placeholder3.png';

const hybridSchema = z.object({
  firstImage: z.enum(['low', 'high']),
  secondImage: z.enum(['low', 'high']),
  filterRadius: z.number()
});

function Hybrid() {
  const ipcRenderer = (window as any).ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Your images couldn't be processed. Please try again."
  });

  const form = useForm<z.infer<typeof hybridSchema>>({
    resolver: zodResolver(hybridSchema),
    defaultValues: {
      firstImage: 'low',
      secondImage: 'high',
      filterRadius: 10
    }
  });

  useEffect(() => {
    reset();
  }, []);

  useEffect(() => {
    if (data) {
      if (data.filteredImage1) {
        setProcessedImageURL(0, data.filteredImage1);
      }

      if (data.filteredImage2) {
        setProcessedImageURL(1, data.filteredImage2);
      }

      if (data.hybridImage) {
        setProcessedImageURL(2, data.hybridImage);
      }
    }
  }, [data]);

  const onSubmit = (data: z.infer<typeof hybridSchema>) => {
    const body = {
      firstImageId: filesIds[0],
      secondImageId: filesIds[1],
      firstFilterType: data.firstImage,
      secondFilterType: data.secondImage,
      filterRadius: data.filterRadius
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/hybrid`
    });
  };

  return (
    <div>
      <Heading
        title="Hybrid Images"
        description="Apply frequency domain filters, and view the hybrid images."
        icon={Blend}
        iconColor="text-blue-700"
        bgColor="bg-blue-700/10"
      />
      <div className="px-4 lg:px-8">
        <div className="mb-4">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="flex flex-wrap gap-4 justify-between items-end"
            >
              <div className="flex flex-wrap gap-4">
                <FormField
                  control={form.control}
                  name="firstImage"
                  render={({ field }) => (
                    <FormItem className="w-[220px]">
                      <Label htmlFor="firstImageFilter">First Image Filter</Label>
                      <Select
                        value={field.value}
                        disabled={isProcessing}
                        onValueChange={(value) => {
                          if (value === 'low') {
                            form.setValue('secondImage', 'high');
                          } else {
                            form.setValue('secondImage', 'low');
                          }
                          field.onChange(value);
                        }}
                      >
                        <FormControl id="firstImageFilter">
                          <SelectTrigger>
                            <SelectValue placeholder="Select transformation type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Filters</SelectLabel>
                            <SelectItem value="low">Low Pass Filter</SelectItem>
                            <SelectItem value="high">High Pass Filter</SelectItem>
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="secondImage"
                  render={({ field }) => (
                    <FormItem className="w-[220px]">
                      <Label htmlFor="secondImageFilter">Second Image Filter</Label>
                      <Select
                        value={field.value}
                        disabled={isProcessing}
                        onValueChange={(value) => {
                          if (value === 'low') {
                            form.setValue('firstImage', 'high');
                          } else {
                            form.setValue('firstImage', 'low');
                          }
                          field.onChange(value);
                        }}
                      >
                        <FormControl id="secondImageFilter">
                          <SelectTrigger>
                            <SelectValue placeholder="Select transformation type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Filters</SelectLabel>
                            <SelectItem value="low">Low Pass Filter</SelectItem>
                            <SelectItem value="high">High Pass Filter</SelectItem>
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />

                <FormField
                  name="filterRadius"
                  render={({ field }) => (
                    <FormItem className="w-[220px]">
                      <Label htmlFor="filterRadius">Filter Radius</Label>
                      <FormControl className="p-2">
                        <Input
                          type="number"
                          disabled={isProcessing}
                          id="filterRadius"
                          min={0}
                          max={100}
                          step={1}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Mix Images
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex flex-col md:flex-row gap-4 w-full">
              <Dropzone index={0} />
              <OutputImage index={0} placeholder={placeholder} />
            </div>
            <div className="flex flex-col md:flex-row gap-4 w-full">
              <Dropzone index={1} />
              <OutputImage index={1} placeholder={placeholder2} />
            </div>
          </div>

          <div className="w-full md:w-[50%] mb-4 mx-auto">
            <OutputImage index={2} placeholder={placeholder3} />
          </div>
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/hybrid')({
  component: Hybrid
});
