/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GH_OWNER?: string;
  readonly VITE_GH_REPO?: string;
  readonly VITE_GH_BRANCH?: string;
  readonly VITE_LIKES_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
