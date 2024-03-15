import useGlobalState from '@renderer/hooks/useGlobalState';
import { Cloud, File } from 'lucide-react';
import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { useToast } from './ui/use-toast';
import { cn } from '@renderer/lib/utils';

interface DropzoneProps {
  index: number;
}

export default function Dropzone({ index }: DropzoneProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [isHover, setIsHover] = useState(false);

  const [progressInterval, setProgressInterval] = useState<NodeJS.Timeout | null>(null);
  const [progressValue, setProgressValue] = useState(0);

  const { toast } = useToast();
  const dropArea = useRef<HTMLLabelElement>(null);

  const { uploadedImagesURLs, setUploadedImageURL, setFileId, setProcessedImageURL, isProcessing } =
    useGlobalState();

  const ipcRenderer = (window as any).ipcRenderer;

  const startSimulateProgress = () => {
    setProgressValue(0);
    const interval = setInterval(() => {
      setProgressValue((value) => {
        if (value >= 90) {
          clearInterval(interval);
          return value;
        }
        return value + 10;
      });
    }, 500);

    return interval;
  };

  useEffect(() => {
    ipcRenderer.on('upload:done', (event: any) => {
      if (event.index === index) {
        setIsUploading(false);
        if (event.data.fileId) {
          toast({
            title: 'Image uploaded',
            description: 'Your image has been uploaded successfully.'
          });
          setFileId(index, event.data.fileId);
        }
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        setIsUploading(false);
        setIsUploaded(true);
        setProgressInterval(null);
      }
    });

    ipcRenderer.on('upload:error', (event: any) => {
      if (event.index === index) {
        toast({
          title: 'Something went wrong',
          description: "Your image couldn't be uploaded, please try again later.",
          variant: 'destructive'
        });
        setIsUploading(false);
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        setProgressInterval(null);
      }
    });
  }, [progressInterval]);

  useEffect(() => {
    const dropAreaElement = dropArea.current;
    if (dropAreaElement) {
      dropAreaElement.addEventListener('dragover', handleDragOver);
      dropAreaElement.addEventListener('dragleave', handleDragLeave);
      dropAreaElement.addEventListener('drop', handleDrop);
    }

    return () => {
      if (dropAreaElement) {
        dropAreaElement.removeEventListener('dragover', handleDragOver);
        dropAreaElement.removeEventListener('dragleave', handleDragLeave);
        dropAreaElement.removeEventListener('drop', handleDrop);
      }
    };
  }, []);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsHover(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsHover(false);
  };

  const handleDrop = (e: DragEvent) => {
    if (isProcessing) return;

    e.preventDefault();
    e.stopPropagation();
    setIsHover(false);
    if (e.dataTransfer) {
      const { files } = e.dataTransfer;
      if (files.length > 1) {
        return toast({
          title: 'Too many files',
          description: 'You can only upload 1 file.',
          variant: 'destructive'
        });
      }

      if (
        files[0].type !== 'image/png' &&
        files[0].type !== 'image/jpg' &&
        files[0].type !== 'image/jpeg'
      ) {
        return toast({
          title: 'Unsupported file',
          description: 'Please upload only PNG, JPG, or JPEG files.',
          variant: 'destructive'
        });
      }

      const file = files[0];
      const url = URL.createObjectURL(file);
      setUploadedImageURL(index, url);
      setIsUploading(true);
      ipcRenderer.send('upload:data', { index, file: file.path });
      return setProgressInterval(startSimulateProgress());
    }
  };

  const handleChangeImage = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const file = files[0];
      const url = URL.createObjectURL(file);
      setUploadedImageURL(index, url);
      setIsUploading(true);
      setProgressInterval(startSimulateProgress());
      ipcRenderer.send('upload:data', { index, file: file.path });
    }
  };

  return (
    <div className="flex flex-col gap-2 flex-1">
      <label
        ref={dropArea}
        htmlFor="poster"
        className={cn(
          'relative transition-all duration-300 h-72 lg:h-80 xl:h-96 rounded-md shadow-sm border-2 border-dashed flex items-center justify-center  border-gray-400 dark:border-gray-600 outline-none',
          !isUploading &&
            !isProcessing &&
            'cursor-pointer hover:bg-gray-300 dark:hover:bg-gray-700',
          isHover && !isUploading && !isProcessing
            ? 'bg-gray-300 dark:bg-gray-700'
            : 'bg-gray-300/70 dark:bg-gray-700/70'
        )}
      >
        <input
          id="file"
          type="file"
          name="file"
          accept="image/*"
          onChange={handleChangeImage}
          disabled={isUploading || isProcessing}
          className={cn(
            'absolute z-30 opacity-0 w-full h-full',
            isUploading || isProcessing ? 'cursor-not-allowed' : 'cursor-pointer'
          )}
        />
        {uploadedImagesURLs[index] && (
          <img src={uploadedImagesURLs[index] ?? ''} className="h-full p-2" alt="uploaded image" />
        )}
        {!uploadedImagesURLs[index] && (
          <div className="space-y-4 text-gray-500 ">
            {isHover ? (
              <div className="flex flex-col gap-6">
                <div className="justify-center flex text-6xl">
                  <File />
                </div>
                <p className="text-center font-medium text-2xl">Yes, right there</p>
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                <div className="justify-center flex text-6xl">
                  <Cloud />
                </div>
                <div className="flex flex-col gap-2">
                  <p className="text-center font-medium text-2xl">
                    Click, or drop your images here
                  </p>
                  <p className="text-center font-medium text-sm">
                    Allowed Files: PNG, JPG, JPEG files
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </label>
      {isUploading && <Progress className="max-w-[250px] h-2" value={progressValue} />}
      {isUploaded && (
        <Button
          className="w-[200px]"
          disabled={isProcessing}
          onClick={() => {
            setIsUploaded(false);
            setUploadedImageURL(index, null);
            setFileId(index, null);
            setProcessedImageURL(index, null);
          }}
        >
          Reset
        </Button>
      )}
    </div>
  );
}
