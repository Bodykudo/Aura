import useGlobalState from '@renderer/hooks/useGlobalState';
import { Cloud, File, Loader2, Upload } from 'lucide-react';
import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';

interface DropzoneProps {
  index: number;
}

export default function Dropzone({ index }: DropzoneProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [isHover, setIsHover] = useState(false);
  const [progressValue, setProgressValue] = useState(0);
  const [progressInterval, setProgressInterval] = useState<NodeJS.Timeout | null>(null);
  const [currentFilePath, setCurrentFilePath] = useState<string | null>(null);

  const dropArea = useRef<HTMLLabelElement>(null);

  const { uploadedImagesURLs, setUploadedImageURL, setFileId, setProcessedImageURL } =
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
      setIsUploading(false);
      if (event.data.id) {
        setFileId(index, event.data.id);
      }
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      setIsUploading(false);
      setIsUploaded(true);
      setProgressInterval(null);
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
    e.preventDefault();
    e.stopPropagation();
    setIsHover(false);
    if (e.dataTransfer) {
      const { files } = e.dataTransfer;
      if (files.length > 1) {
        console.log('You can only upload 1 file.');
        return;
      }

      if (
        files[0].type !== 'image/png' &&
        files[0].type !== 'image/jpg' &&
        files[0].type !== 'image/jpeg'
      ) {
        console.log('Unsupported file, please upload only PNG, JPG, or JPEG files.');
        return;
      }
    }
  };

  const handleChangeImage = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const file = files[0];
      setCurrentFilePath(file.path);
      const url = URL.createObjectURL(file);
      setUploadedImageURL(index, url);
    }
  };

  const handleUpload = () => {
    setIsUploading(true);
    ipcRenderer.send('upload:data', { file: currentFilePath });
    setProgressInterval(startSimulateProgress());
  };

  return (
    <div className="flex flex-col gap-2 flex-1">
      <label
        ref={dropArea}
        htmlFor="poster"
        className={`relative hover:bg-gray-300 transition-all duration-300 h-72 lg:h-80 xl:h-96 rounded-md shadow-sm border-2 border-dashed cursor-pointer flex items-center justify-center  border-gray-400 outline-none ${
          isHover ? 'bg-gray-300' : 'bg-gray-300/70'
        }`}
      >
        <input
          id="file"
          type="file"
          name="file"
          accept="image/*"
          onChange={handleChangeImage}
          className="absolute z-30 cursor-pointer opacity-0 w-full h-full"
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
                <h3 className="text-center font-medium text-2xl">Yes, right there</h3>
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                <div className="justify-center flex text-6xl">
                  <Cloud />
                </div>
                <div className="flex flex-col gap-2">
                  <h3 className="text-center font-medium text-2xl">
                    Click, or drop your images here
                  </h3>
                  <p className="text-center font-medium text-sm">
                    Allowed Files: PNG, JPG, JPEG files
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </label>
      {!isUploaded && (
        <div className="flex flex-col justify-between gap-4 md:flex-row items-center">
          <Button
            disabled={!uploadedImagesURLs[index] || isUploading}
            onClick={handleUpload}
            className="w-[200px]"
          >
            {isUploading ? (
              <>
                <Loader2 className="h-4 w-4 mr-1.5 animate-spin" /> Uploading
              </>
            ) : (
              <>
                <Upload className="h-4 w-4 mr-1.5" /> Upload
              </>
            )}
          </Button>
          {isUploading && <Progress className="max-w-[250px] h-2" value={progressValue} />}
        </div>
      )}
      {isUploaded && (
        <Button
          className="w-[200px]"
          onClick={() => {
            setIsUploaded(false);
            setUploadedImageURL(index, null);
            setCurrentFilePath(null);
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
