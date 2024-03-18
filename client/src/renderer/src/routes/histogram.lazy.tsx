import { useEffect, useState } from 'react';
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
import { useToast } from '@renderer/components/ui/use-toast';

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
      minWidth: 120,
      maxWidth: 200
    }
  });

  const [originalHistogram, setOriginalHistogram] = useState([]);
  const [newHistogram, setNewHistogram] = useState([]);
  const [originalCdf, setOriginalCdf] = useState([]);
  const [newCdf, setNewCdf] = useState([]);

  const { toast } = useToast();

  useEffect(() => {
    setIsProcessing(false);
    setFileId(0, null);
    setUploadedImageURL(0, null);
    setProcessedImageURL(0, null);
  }, []);

  useEffect(() => {
    const imageReceivedListener = (event: any) => {
      console.log(event.data);
      if (event.data.image) {
        setProcessedImageURL(0, event.data.image);
      }
      if (event.data.histogram) {
        const original = event.data.histogram.original;
        if (original) {
          setOriginalHistogram(original.histogram);
          setOriginalCdf(original.cdf);
        }

        const newData = event.data.histogram.new;
        if (newData) {
          setNewHistogram(newData.histogram);
          setNewCdf(newData.cdf);
        }
        // if (original) {
        //   const originalData = original.red.map((value: number, index: number) => {
        //     return {
        //       name: index,
        //       red: value,
        //       green: original.green[index],
        //       blue: original.blue[index]
        //     };
        //   });
        //   setOriginalHistogram(originalData);
        //   const originalCdfData = original.red.map((value: number, index: number) => {
        //     return {
        //       name: index,
        //       red: original.red.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0),
        //       green: original.green.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0),
        //       blue: original.blue.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0)
        //     };
        //   });
        //   setOriginalCdf(originalCdfData);
        // }
        // const newHist = event.data.histogram.new;
        // if (newHist) {
        //   const newData = newHist.red.map((value: number, index: number) => {
        //     return {
        //       name: index,
        //       red: value,
        //       green: newHist.green[index],
        //       blue: newHist.blue[index]
        //     };
        //   });
        //   // Find the last non-zero entry
        //   let lastIndex = newData.length - 1;
        //   while (
        //     lastIndex >= 0 &&
        //     newData[lastIndex].red === 0 &&
        //     newData[lastIndex].green === 0 &&
        //     newData[lastIndex].blue === 0
        //   ) {
        //     lastIndex--;
        //   }
        //   // Trim the zeros from the end
        //   const trimmedData = newData.slice(0, lastIndex + 1);
        //   setNewHistogram(trimmedData);
        //   const newCdfData = newHist.red.map((value: number, index: number) => {
        //     return {
        //       name: index,
        //       red: newHist.red.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0),
        //       green: newHist.green.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0),
        //       blue: newHist.blue.slice(0, index + 1).reduce((acc, curr) => acc + curr, 0)
        //     };
        //   });
        //   setNewCdf(newCdfData);
        // }
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
      type: data.type
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/histogram/${filesIds[0]}`
    });
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
            <RGBHistogram data={originalHistogram} />
          </div>
          <div className="flex flex-1 flex-col items-center gap-2">
            <h2 className="text-xl font-bold">RGB Histogram</h2>
            <RGBHistogram data={newHistogram} />
          </div>
        </div>
        <div className="flex flex-col md:flex-row gap-4 w-full mt-8 mb-4">
          <div className="flex flex-1 flex-col items-center gap-2">
            <h2 className="text-xl font-bold">RGB CDF</h2>
            <RGBHistogram data={originalCdf} />
          </div>
          <div className="flex flex-1 flex-col items-center gap-2">
            <h2 className="text-xl font-bold">RGB CDF</h2>
            <RGBHistogram data={newCdf} />
          </div>
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/histogram')({
  component: Histogram
});

const ddata = [
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

const cdfData = ddata.map((item, index) => {
  return {
    name: item.name,
    red: ddata.slice(0, index + 1).reduce((acc, curr) => acc + curr.red, 0),
    green: ddata.slice(0, index + 1).reduce((acc, curr) => acc + curr.green, 0),
    blue: ddata.slice(0, index + 1).reduce((acc, curr) => acc + curr.blue, 0)
  };
});

function RGBHistogram({ data, isCdf = false }: { data: any; isCdf?: boolean }) {
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
        <Area type="monotone" dataKey="red" stackId="1" stroke="red" fill="red" />
        <Area type="monotone" dataKey="green" stackId="1" stroke="green" fill="green" />
        <Area type="monotone" dataKey="blue" stackId="1" stroke="blue" fill="blue" />
        <Area type="monotone" dataKey="gray" stackId="1" stroke="black" fill="black" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
