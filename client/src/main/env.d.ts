/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly MAIN_VITE_PUBLIC_API_URL: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
