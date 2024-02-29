import { zodResolver } from '@hookform/resolvers/zod';
import Dropzone from '@renderer/components/Dropzone';
import Heading from '@renderer/components/Heading';
import OutputImage from '@renderer/components/OutputImage';
import { Button } from '@renderer/components/ui/button';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
import { Input } from '@renderer/components/ui/input';
import { Label } from '@renderer/components/ui/label';
import useGlobalState from '@renderer/hooks/useGlobalState';
import { createLazyFileRoute } from '@tanstack/react-router';
import { Blend } from 'lucide-react';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

const hybridSchema = z.object({
  lowFrequencyFilterRadius: z.number(),
  highFrequencyFilterRadius: z.number()
});

function Hybrid() {
  const { setUploadedImageURL, setProcessedImageURL } = useGlobalState();

  const form = useForm<z.infer<typeof hybridSchema>>({
    resolver: zodResolver(hybridSchema),
    defaultValues: {
      lowFrequencyFilterRadius: 10,
      highFrequencyFilterRadius: 10
    }
  });

  useEffect(() => {
    setUploadedImageURL(0, null);
    setUploadedImageURL(1, null);
    setProcessedImageURL(0, null);
    setProcessedImageURL(1, null);
    setProcessedImageURL(2, null);
  }, []);

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
              onSubmit={form.handleSubmit((values) => console.log(values))}
              className="flex flex-wrap gap-4 justify-between items-end"
            >
              <div className="flex flex-wrap gap-4">
                <FormField
                  name="lowFrequencyFilterRadius"
                  render={({ field }) => (
                    <FormItem className="w-[220px]">
                      <Label htmlFor="lowFrequencyFilterRadius">Low Frequency Filter Radius</Label>
                      <FormControl className="p-2">
                        <Input
                          type="number"
                          id="lowFrequencyFilterRadius"
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
                <FormField
                  name="highFrequencyFilterRadius"
                  render={({ field }) => (
                    <FormItem className="w-[220px]">
                      <Label htmlFor="highFrequencyFilterRadius">
                        High Frequency Filter Radius
                      </Label>
                      <FormControl className="p-2">
                        <Input
                          type="number"
                          id="highFrequencyFilterRadius"
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
              <Button type="submit">Mix Images</Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex flex-col md:flex-row gap-4 w-full">
              <Dropzone index={0} />
              <OutputImage index={0} />
            </div>
            <div className="flex flex-col md:flex-row gap-4 w-full">
              <Dropzone index={1} />
              <OutputImage index={1} />
            </div>
          </div>

          <div className="w-full md:w-[50%] mb-4 mx-auto">
            <OutputImage index={2} />
          </div>
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/hybrid')({
  component: Hybrid
});
