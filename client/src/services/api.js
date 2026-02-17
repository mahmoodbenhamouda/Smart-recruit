import axios from 'axios';
import { getAuthStore } from '../state/authStore.js';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  withCredentials: false,
  headers: {
    'Content-Type': 'application/json'
  }
});

api.interceptors.request.use((config) => {
  const store = getAuthStore();
  if (store?.token) {
    config.headers.Authorization = `Bearer ${store.token}`;
  }
  return config;
});

export default api;

