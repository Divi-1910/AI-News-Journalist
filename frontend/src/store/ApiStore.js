import { atom } from "jotai";

// API configuration
export const apiBaseUrlAtom = atom(
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
);

// API loading states
export const apiLoadingAtom = atom(false);
export const apiErrorAtom = atom(null);
