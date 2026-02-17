import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { clearAuthStore, setAuthStore } from './authStore.js';
import api from '../services/api.js';

export const AuthContext = createContext(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

const TOKEN_KEY = 'talent-bridge-token';
const USER_KEY = 'talent-bridge-user';

export function AuthProvider ({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem(USER_KEY);
    return stored ? JSON.parse(stored) : null;
  });

  useEffect(() => {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  }, [token]);

  useEffect(() => {
    if (user) {
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(USER_KEY);
    }
  }, [user]);

  const value = useMemo(() => ({
    token,
    user,
    login: async (email, password) => {
      const response = await api.post('/auth/login', { email, password });
      const { token: newToken, user: newUser } = response.data;
      setToken(newToken);
      setUser(newUser);
      setAuthStore({ token: newToken, user: newUser });
      return newUser;
    },
    register: async (formData) => {
      const response = await api.post('/auth/register', formData);
      const { token: newToken, user: newUser } = response.data;
      setToken(newToken);
      setUser(newUser);
      setAuthStore({ token: newToken, user: newUser });
      return newUser;
    },
    logout: () => {
      setToken(null);
      setUser(null);
      clearAuthStore();
    }
  }), [token, user]);

  useEffect(() => {
    setAuthStore({ token, user });
    return () => {
      clearAuthStore();
    };
  }, [token, user]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

