import { createLazyFileRoute } from '@tanstack/react-router';
import { Cloud, File } from 'lucide-react';
import { ChangeEvent, useEffect, useRef, useState } from 'react';

function Filters() {
  const ipcRenderer = (window as any).ipcRenderer;

  useEffect(() => {
    ipcRenderer.on('upload:done', (event: any) => {
      console.log(event);
      if (event?.data) {
        console.log(event.data);
      }
    });
  }, []);

  const [isHover, setIsHover] = useState(false);
  const dropArea = useRef<HTMLLabelElement>(null);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const file = files[0];
      console.log(file);
      ipcRenderer.send('upload:data', { file: file.path });
    }
  };

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

      if (files[0].type !== 'text/csv') {
        console.log('Unsupported file, please upload only CSV files.');
        return;
      }

      // Change the current file state to the dropped file
    }
  };

  return (
    <div className="px-4 py-6">
      <div className="flex gap-4 w-full">
        <div className="flex-grow">
          <label
            ref={dropArea}
            htmlFor="poster"
            className={`relative hover:bg-gray-300 transition-all duration-300 h-72 lg:h-80 xl:h-96 rounded-3xl shadow-sm border-2 border-dashed cursor-pointer flex items-center justify-center  border-gray-400 outline-none ${
              isHover ? 'bg-gray-300' : 'bg-gray-300/70'
            }`}
          >
            <input
              id="file"
              type="file"
              name="file"
              accept=".csv"
              onChange={handleUpload}
              className="absolute z-30 cursor-pointer opacity-0 w-full h-full"
            />
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
          </label>
        </div>

        <div className="flex-grow">
          <label
            ref={dropArea}
            htmlFor="poster"
            className={`relative hover:bg-gray-300 transition-all duration-300 h-72 lg:h-80 xl:h-96 rounded-3xl shadow-sm border-2 border-dashed cursor-pointer flex items-center justify-center  border-gray-400 outline-none ${
              isHover ? 'bg-gray-300' : 'bg-gray-300/70'
            }`}
          >
            <input
              id="file"
              type="file"
              name="file"
              accept=".csv"
              onChange={handleUpload}
              className="absolute z-30 cursor-pointer opacity-0 w-full h-full"
            />
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
          </label>
        </div>
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/filters')({
  component: Filters
});
