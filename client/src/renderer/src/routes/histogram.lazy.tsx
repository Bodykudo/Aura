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
import { BarChartBig } from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  ResponsiveContainer
} from 'recharts';

const noiseSchema = z.object({
  type: z.enum(['grayscale', 'normalization', 'equalization']).nullable(),
  minWidth: z.number(),
  maxWidth: z.number()
});

const typesOptions = [
  { label: 'Grayscale', value: 'grayscale' },
  { label: 'Normalization', value: 'normalization' },
  { label: 'Equalization', value: 'equalization' }
];

const inputs = [
  {
    value: 'normalization',
    inputs: [
      { label: 'Min. Width', name: 'minWidth', min: 0, max: 400, step: 1 },
      { label: 'Max. Width', name: 'maxWidth', min: 0, max: 400, step: 1 }
    ]
  }
];

function Histogram() {
  const ipcRenderer = (window as any).ipcRenderer;
  const { setUploadedImageURL, setProcessedImageURL } = useGlobalState();
  // const downloadRef = useRef<HTMLAnchorElement | null>(null);

  const form = useForm<z.infer<typeof noiseSchema>>({
    resolver: zodResolver(noiseSchema),
    defaultValues: {
      minWidth: 120,
      maxWidth: 200
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
    console.log(data);
    // ipcRenderer.send('process:image', data);
  };

  return (
    <div>
      <Heading
        title="Histogram, Normalization, & Equalization"
        description="View and manipulate the histogram of an image, and apply normalization and equalization."
        icon={BarChartBig}
        iconColor="text-emerald-500"
        bgColor="bg-emerald-500/10"
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
                      <Label htmlFor="transformationType">Transform Image</Label>
                      <Select onValueChange={field.onChange}>
                        <FormControl id="transformationType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select transformation type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Transformation</SelectLabel>
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
              <Button type="submit">Apply Filter</Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full">
          <Dropzone index={0} />
          <OutputImage index={0} />
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full mt-8 mb-4">
          <div className="flex flex-1 flex-col items-center gap-2">
            <h2 className="text-xl font-bold">RGB Histogram</h2>
            <RGBHistogram />
          </div>
          <div className="flex flex-1 flex-col items-center gap-2">
            <h2 className="text-xl font-bold">RGB Histogram</h2>
            <RGBHistogram />
          </div>
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/histogram')({
  component: Histogram
});

const data = [
  { name: '0-15', red: 4000, green: 2400, blue: 2400 },
  { name: '16-31', red: 3000, green: 1398, blue: 2210 },
  { name: '32-47', red: 2000, green: 9800, blue: 2290 },
  { name: '48-63', red: 2780, green: 3908, blue: 2000 },
  { name: '64-79', red: 1890, green: 4800, blue: 2181 },
  { name: '80-95', red: 2390, green: 3800, blue: 2500 },
  { name: '96-111', red: 3490, green: 4300, blue: 2100 },
  { name: '112-127', red: 4000, green: 2400, blue: 2400 },
  { name: '128-143', red: 3000, green: 1398, blue: 2210 },
  { name: '144-159', red: 2000, green: 9800, blue: 2290 },
  { name: '160-175', red: 2780, green: 3908, blue: 2000 },
  { name: '176-191', red: 1890, green: 4800, blue: 2181 },
  { name: '192-207', red: 2390, green: 3800, blue: 2500 },
  { name: '208-223', red: 3490, green: 4300, blue: 2100 },
  { name: '224-239', red: 4000, green: 2400, blue: 2400 },
  { name: '240-255', red: 3000, green: 1398, blue: 2210 }
];

function RGBHistogram() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart
        width={600}
        height={300}
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Area type="bump" dataKey="red" stackId="1" stroke="red" fill="red" />
        <Area type="bump" dataKey="green" stackId="1" stroke="green" fill="green" />
        <Area type="bump" dataKey="blue" stackId="1" stroke="blue" fill="blue" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
