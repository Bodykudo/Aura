import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { ClipboardX } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
import { Input } from '@renderer/components/ui/input';
import { Label } from '@renderer/components/ui/label';
import { Button } from '@renderer/components/ui/button';

import useGlobalState from '@renderer/hooks/useGlobalState';
import { useToast } from '@renderer/components/ui/use-toast';

import placeholder from '@renderer/assets/placeholder.png';

const siftSchema = z.object({
  sigma: z.number(),
  k: z.number(),
  contrastThreshold: z.number(),
  edgeThreshold: z.number(),
  magnitudeThreshold: z.number()
});

const inputs = [
  { label: 'Sigma', name: 'sigma', min: 0.1, max: 3, step: 0.1 },
  { label: 'K', name: 'k', min: 0.1, max: 5, step: 0.1 },
  { label: 'Contrast Threshold', name: 'contrastThreshold', min: 0.01, max: 1, step: 0.01 },
  { label: 'Edge Threshold', name: 'edgeThreshold', min: 1, max: 100, step: 1 },
  { label: 'Magnitude Threshold', name: 'magnitudeThreshold', min: 0.01, max: 1, step: 0.01 }
];

function SIFT() {
  const ipcRenderer = (window as any).ipcRenderer;

  const {
    filesIds,
    setFileId,
    setUploadedImageURL,
    setProcessedImageURL,
    isProcessing,
    setIsProcessing
  } = useGlobalState();

  const form = useForm<z.infer<typeof siftSchema>>({
    resolver: zodResolver(siftSchema),
    defaultValues: {
      sigma: 1,
      k: 2,
      contrastThreshold: 0.04,
      edgeThreshold: 10,
      magnitudeThreshold: 0.2
    }
  });

  const { toast } = useToast();

  useEffect(() => {
    setIsProcessing(false);
    setFileId(0, null);
    setFileId(1, null);
    setUploadedImageURL(0, null);
    setUploadedImageURL(1, null);
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
        description: "Your images couldn't be processed, please try again later.",
        variant: 'destructive'
      });
      setIsProcessing(false);
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, []);

  const onSubmit = (data: z.infer<typeof siftSchema>) => {
    const body = {
      originalImage: filesIds[0],
      templateImage: filesIds[1],
      sigma: data.sigma,
      k: data.k,
      contrastThreshold: data.contrastThreshold,
      edgeThreshold: data.edgeThreshold,
      magnitudeThreshold: data.magnitudeThreshold
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: '/api/sift'
    });
  };

  return (
    <div>
      <Heading
        title="SIFT"
        description="Apply SIFT to an image to detect keypoints and match features."
        icon={ClipboardX}
        iconColor="text-teal-600"
        bgColor="bg-teal-600/10"
      />
      <div className="px-4 lg:px-8">
        <div className="mb-4">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="flex flex-wrap gap-4 justify-between items-end"
            >
              <div className="flex flex-wrap gap-2">
                {inputs.map((input) => {
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
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Apply SIFT
              </Button>
            </form>
          </Form>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col md:flex-row gap-4 w-full">
            <div className="flex flex-col gap-1 w-full">
              <p className="font-medium text-xl">Image</p>
              <Dropzone index={0} />
            </div>
            <div className="flex flex-col gap-1 w-full">
              <p className="font-medium text-xl">Template</p>
              <Dropzone index={1} />
            </div>
          </div>
          <OutputImage index={0} placeholder={placeholder} />
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/sift')({
  component: SIFT
});
