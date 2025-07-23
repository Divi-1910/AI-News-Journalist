import { atom } from "jotai";

// Auth atoms
export const userAtom = atom(null);
export const isLoadingAtom = atom(false);
export const isAuthenticatedAtom = atom(false);

// User preferences atom
export const userPreferencesAtom = atom({
  newsPersonality: null,
  favoriteTopics: [],
  contentLength: "medium",
  onboardingCompleted: false
});

// UI state atoms
export const currentPersonalityAtom = atom(0);
export const showLoginButtonAtom = atom(false);
