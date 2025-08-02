import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json"
  }
});

const getAuthToken = () => {
  return localStorage.getItem("Infiya_token");
};

const setAuthToken = (token) => {
  localStorage.setItem("Infiya_token", token);
};

const removeAuthToken = () => {
  localStorage.removeItem("Infiya_token");
};

apiClient.interceptors.request.use(
  (config) => {
    const token = getAuthToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    if (import.meta.env.DEV) {
      console.log(
        `API Request: ${config.method?.toUpperCase()} ${config.url}`,
        {
          data: config.data,
          params: config.params
        }
      );
    }
    return config;
  },
  (error) => {
    console.log(`Request Interceptor Error : ${error}`);
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    if (import.meta.env.DEV) {
      console.log(
        `API Response: ${response.config.method?.toUpperCase()} ${
          response.config.url
        }`,
        {
          status: response.status,
          data: response.data
        }
      );
    }
    return response;
  },
  (error) => {
    if (error.response) {
      const { status, data } = error.response;

      if (status === 401) {
        console.log("Authentication Failed - Removing stored credentials");
        removeAuthToken();

        if (window.location.pathname !== "/") {
          window.location.href = "/";
        }
      }

      if (status >= 500) {
        console.error(`Server Error : ${data}`);
      }
      if (import.meta.env.DEV) {
        console.error(
          `❌ API Error: ${error.config?.method?.toUpperCase()} ${
            error.config?.url
          }`,
          {
            status,
            data,
            message: error.message
          }
        );
      }
    } else if (error.request) {
      console.error(`Network Error : No Response Received : ${error.request}`);
    } else {
      console.error(`Request Setup Error : `, error.message);
    }

    return Promise.reject(error);
  }
);

export const handleApiError = (
  error,
  defaultMessage = "An unexpected error occurred"
) => {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error.message) {
    return error.message;
  }

  return defaultMessage;
};

export const AuthApi = {
  googleAuth: async (tokenData) => {
    try {
      const response = await apiClient.post("/api/auth/google", tokenData);

      if (response.data.success && response.data.token) {
        setAuthToken(response.data.token);
        localStorage.setItem(
          "Infiya_user_info",
          JSON.stringify(response.data.user)
        );
      }

      return response.data;
    } catch (error) {
      console.error(`Google Auth Error: ${error}`);
      throw new Error(error.response?.data?.detail || "Google Auth Failed");
    }
  },

  getCurrentUser: async () => {
    try {
      const response = await apiClient.get("/api/auth/me");
      return response.data;
    } catch (error) {
      console.error(`Failed to get Current User:`, error);
      throw new Error(
        error.response?.data?.detail || "Failed to fetch user Information"
      );
    }
  },

  logout: () => {
    removeAuthToken();
    window.location.href = "/";
  },

  refreshToken: async () => {
    try {
      const response = await apiClient.post("/api/auth/refresh");
      if (response.data.token) {
        setAuthToken(response.data.token);
      }
      return response.data;
    } catch (error) {
      console.error(`Failed to refresh Token:`, error);
      removeAuthToken();
      throw new Error(
        error.response?.data?.detail || "Failed to refresh Token"
      );
    }
  },

  updateUserPreferences: async (preferences) => {
    try {
      const response = await apiClient.put(
        "/api/users/preferences",
        preferences
      );
      return response.data;
    } catch (error) {
      console.error("Failed to update preferences:", error);
      throw new Error(
        error.response?.data?.detail ||
          "Failed to update preferences. Please try again."
      );
    }
  }
};

export const UserApi = {
  updateUserPreferences: async (preferences) => {
    try {
      const response = await apiClient.put(
        "/api/users/preferences",
        preferences
      );
      return response.data;
    } catch (error) {
      console.error("Failed to update preferences:", error);
      throw new Error(
        error.response?.data?.detail ||
          "Failed to update preferences. Please try again."
      );
    }
  }
};

export const ChatApi = {
  loadChatHistory: async (limit) => {
    try {
      const response = await apiClient.get(`/api/chats/history?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.log("Failed to get the chat history : ", error);
      throw new Error(
        error.response?.data?.detail || "Failed to get the chat history"
      );
    }
  },

  sendMessage: async (messageContent) => {
    try {
      const response = await apiClient.post(`/api/chats/send`, {
        message: messageContent
      });
      return response.data;
    } catch (error) {
      console.log("Failed to send message : ", error);
      throw new Error(error.response?.data?.detail || "Failed to send message");
    }
  },

  clearChat: async () => {
    try {
      const response = await apiClient.delete(`/api/chats/clear`);
      return response.data;
    } catch (error) {
      console.log("Failed to clear chat : ", error);
      throw new Error(error.response?.data?.detail || "Failed to clear chat");
    }
  }
};

export default apiClient;
export { getAuthToken, setAuthToken, removeAuthToken, API_BASE_URL };
