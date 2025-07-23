import { atom } from "jotai";

export const userAtom = atom(null);
export const isLoadingAtom = atom(false);
export const isAuthenticatedAtom = atom(false);

export const currentPersonalityAtom = atom(0);
export const showLoginButtonAtom = atom(false);

export const needsOnboardingAtom = atom((get) => {
  const user = get(userAtom);
  const isAuthenticated = get(isAuthenticatedAtom);

  return isAuthenticated && user && !user.onboarding_completed;
});
