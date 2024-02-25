import { create } from 'zustand';

interface IUseGlobalState {
  filesIds: string[] | null[];
  setFileId: (index: number, fileId: string | null) => void;
  uploadedImagesURLs: string[] | null[];
  setUploadedImageURL: (index: number, imageURL: string | null) => void;
  processedImagesURLs: string[] | null[];
  setProcessedImageURL: (index: number, imageURL: string | null) => void;
}

const useGlobalState = create<IUseGlobalState>((set) => ({
  filesIds: [null, null],
  setFileId: (index, fileId) =>
    set((state) => ({ filesIds: state.filesIds.map((id, i) => (i === index ? fileId : id)) })),
  uploadedImagesURLs: [null, null, null],
  setUploadedImageURL: (index, imageURL) =>
    set((state) => ({
      uploadedImagesURLs: state.uploadedImagesURLs.map((url, i) => (i === index ? imageURL : url))
    })),
  processedImagesURLs: [null, null, null],
  setProcessedImageURL: (index, imageURL) =>
    set((state) => ({
      processedImagesURLs: state.processedImagesURLs.map((url, i) => (i === index ? imageURL : url))
    }))
}));

export default useGlobalState;
