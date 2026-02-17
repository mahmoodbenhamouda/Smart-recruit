const authStore = {
  token: null,
  user: null
};

export function setAuthStore ({ token, user }) {
  authStore.token = token ?? null;
  authStore.user = user ?? null;
}

export function clearAuthStore () {
  authStore.token = null;
  authStore.user = null;
}

export function getAuthStore () {
  return authStore;
}

