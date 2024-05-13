import { useEffect } from 'react';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { ScanFace } from 'lucide-react';

import Heading from '@renderer/components/Heading';
import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
import { Form, FormControl, FormField, FormItem } from '@renderer/components/ui/form';
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

const faceAnalysisSchema = z.object({
  type: z.enum(['faceDetection', 'faceRecognition']).nullable()
});

const analysisOptions = [
  { label: 'Face Detection', value: 'faceDetection' },
  { label: 'Face Recognition', value: 'faceRecognition' }
];

function Face() {
  const ipcRenderer = window.ipcRenderer;

  const { filesIds, setProcessedImageURL, isProcessing, setIsProcessing, reset } = useGlobalState();
  const { data } = useHandleProcessing({
    fallbackFn: () => setIsProcessing(false),
    errorMessage: "Face analysis couldn't be applied to your image. Please try again."
  });

  const form = useForm<z.infer<typeof faceAnalysisSchema>>({
    resolver: zodResolver(faceAnalysisSchema)
  });

  useEffect(() => {
    reset();
  }, []);

  useEffect(() => {
    if (data && data.image) {
      setProcessedImageURL(0, data.image);
    }
  }, [data]);

  const onSubmit = (data: z.infer<typeof faceAnalysisSchema>) => {
    const body = {
      type: data.type
    };

    setIsProcessing(true);
    ipcRenderer.send('process:image', {
      body,
      url: `/api/face/${filesIds[0]}`
    });
  };

  return (
    <div>
      <Heading
        title="Face Detection & Recognition"
        description="Detect and recognize faces in an image."
        icon={ScanFace}
        iconColor="text-blue-600"
        bgColor="bg-blue-500/10"
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
                      <Label htmlFor="faceAnalysisType">Face Analysis Type</Label>
                      <Select disabled={isProcessing} onValueChange={field.onChange}>
                        <FormControl id="faceAnalysisType">
                          <SelectTrigger>
                            <SelectValue placeholder="Select analysis type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Face Analysis</SelectLabel>
                            {analysisOptions.map((option) => (
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
              </div>
              <Button disabled={!filesIds[0] || isProcessing} type="submit">
                Apply Face Analysis
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

export const Route = createLazyFileRoute('/face')({
  component: Face
});
