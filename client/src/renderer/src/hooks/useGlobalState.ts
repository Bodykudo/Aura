import { create } from 'zustand';

interface IUseGlobalState {
  fileId: string | null;
  setFileId: (fileId: string | null) => void;
  uploadedImageURL: string | null;
  setUploadedImageURL: (uploadedImageURL: string | null) => void;
  processedImageURL: string | null;
  setProcessedImageURL: (processedImageURL: string | null) => void;
}

const useGlobalState = create<IUseGlobalState>((set) => ({
  fileId: null,
  setFileId: (fileId) => set({ fileId }),
  uploadedImageURL: null,
  setUploadedImageURL: (uploadedImageURL) => set({ uploadedImageURL }),
  processedImageURL: null,
  setProcessedImageURL: (processedImageURL) => set({ processedImageURL })
}));

export default useGlobalState;
