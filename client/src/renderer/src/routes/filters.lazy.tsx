import Dropzone from '@renderer/components/Dropzone';
import OutputImage from '@renderer/components/OutputImage';
import { Button } from '@renderer/components/ui/button';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@renderer/components/ui/select';
import useGlobalState from '@renderer/hooks/useGlobalState';
import { createLazyFileRoute } from '@tanstack/react-router';
import { useEffect } from 'react';

function Filters() {
  const ipcRenderer = (window as any).ipcRenderer;
  const { setProcessedImageURL } = useGlobalState();

  useEffect(() => {
    // ipcRenderer.on('upload:done', (event: any) => {
    //   console.log(event);
    //   if (event?.data) {
    //     console.log(event.data);
    //   }
    // });
    ipcRenderer.on('image:received', (event: any) => {
      console.log(event);
      setProcessedImageURL(event);
      if (event?.data) {
        console.log(event.data);
      }
    });
  }, []);

  const handleClick = () => {
    console.log('clicked');
    ipcRenderer.send('get:image');
  };

  return (
    <div className="px-4 py-6">
      <div className="mb-4">
        <div className="flex justify-between">
          <Select>
            <SelectTrigger className="w-[250px]">
              <SelectValue placeholder="Select filter type" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Filters</SelectLabel>
                <SelectItem value="average">Average Filter</SelectItem>
                <SelectItem value="gaussian">Gaussian Filter</SelectItem>
                <SelectItem value="laplacian">Laplacian Filter</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>

          <Button onClick={handleClick}>Filter</Button>
        </div>
      </div>
      <div className="flex flex-col md:flex-row gap-4 w-full">
        <Dropzone />
        <OutputImage />
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/filters')({
  component: Filters
});
