import { useEffect, useState } from 'react';

import { useToast } from '@renderer/components/ui/use-toast';

export default function useHandleProcessing({
  fallbackFn,
  errorMessage
}: {
  fallbackFn: () => void;
  errorMessage: string;
}) {
  const ipcRenderer = window.ipcRenderer;
  const [data, setData] = useState<any>(null);

  const { toast } = useToast();

  useEffect(() => {
    const imageReceivedListener = (event: any) => {
      setData(event.data);
      fallbackFn();
    };
    ipcRenderer.on('image:received', imageReceivedListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, [fallbackFn]);

  useEffect(() => {
    const imageErrorListener = () => {
      toast({
        title: 'Something went wrong',
        description: errorMessage,
        variant: 'destructive'
      });
      fallbackFn();
    };
    ipcRenderer.on('image:error', imageErrorListener);

    return () => {
      ipcRenderer.removeAllListeners();
    };
  }, [fallbackFn]);

  return { data };
}
