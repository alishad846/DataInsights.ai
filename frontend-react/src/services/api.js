import axios from 'axios';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

const api = axios.create({
    baseURL: API_URL,
});

// Attach auth token to every request
api.interceptors.request.use((config) => {
    const token = sessionStorage.getItem('token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});

// ---- Auth ----
export const loginUser = async (email, password, role) => {
    const response = await api.post('/auth/login', { email, password, role });
    return response.data;
};

export const signupUser = async ({ firstName, lastName, phone, email, password, role }) => {
    const response = await api.post('/auth/signup', { firstName, lastName, phone, email, password, role });
    return response.data;
};

// Sync role from backend (call on layout mount to pick up admin role changes)
export const getMe = async () => {
    try {
        const response = await api.get('/auth/me');
        if (response.data.success) {
            const user = response.data.user;
            sessionStorage.setItem('role', user.role);
            sessionStorage.setItem('userName', user.full_name || user.name || (user.email ? user.email.split('@')[0] : 'User'));
            return user;
        }
    } catch (err) {
        console.warn('Failed to sync user:', err.message);
    }
    return null;
};

// ---- User Management (Admin) ----
export const getUsers = async (roleFilter) => {
    const params = roleFilter && roleFilter !== 'all' ? { role: roleFilter } : {};
    const response = await api.get('/auth/users', { params });
    return response.data;
};

export const getUserStats = async () => {
    const response = await api.get('/auth/users/stats');
    return response.data;
};

export const createUser = async (payload) => {
    const response = await api.post('/auth/users', payload);
    return response.data;
};

export const updateUserRole = async (email, role) => {
    const response = await api.put(`/auth/users/${encodeURIComponent(email)}/role`, { role });
    return response.data;
};

export const updateUserStatus = async (email, status) => {
    const response = await api.put(`/auth/users/${encodeURIComponent(email)}/status`, { status });
    return response.data;
};

export const deleteUser = async (email) => {
    const response = await api.delete(`/auth/users/${encodeURIComponent(email)}`);
    return response.data;
};

export const getPendingUsers = async () => {
    const response = await api.get('/auth/users/pending');
    return response.data;
};

export const approveUser = async (email, approved) => {
    const response = await api.put(`/auth/users/${encodeURIComponent(email)}/approve`, { approved });
    return response.data;
};

export const uploadDataset = async (file) => {
    const formData = new FormData();
    formData.append('dataset', file);

    // Let axios set the proper boundary for multipart/form-data
    const response = await api.post('/upload', formData);
    return response.data;
};

export const getDatasetStatus = async (datasetId) => {
    const response = await api.get(`/dataset-status/${datasetId}`);
    return response.data;
};

export const getDashboardConfig = async (datasetId) => {
    try {
        const response = await api.get(`/dashboard/${datasetId}`);
        return response.data;
    } catch (err) {
        console.warn("Dashboard config not available:", err.message);
        return { success: false, charts: [], insights: [], executive_summary: "" };
    }
};

export const getAnalytics = async (datasetId) => {
    const response = await api.get('/analytics', { params: { datasetId } });
    return response.data;
};

export const askQuery = async (datasetId, question) => {
    const response = await api.post('/query', {
        datasetId: datasetId,
        question: question,
    });
    return response.data;
};

export const getDatasets = async () => {
    const response = await api.get('/datasets');
    return response.data;
};

export const getDatasetsAdmin = async () => {
    const response = await api.get('/datasets-admin');
    return response.data;
};

export const getAvailableDatasetsToRequest = async () => {
    const response = await api.get('/datasets/available-to-request');
    return response.data;
};


export const deleteDataset = async (datasetId) => {
    const response = await api.delete(`/datasets/${datasetId}`);
    return response.data;
};

// ---- Dataset Assignment ----
export const assignDataset = async (datasetId, userIds) => {
    const response = await api.post(`/datasets/${datasetId}/assign`, { userIds });
    return response.data;
};

export const unassignDataset = async (datasetId, userId) => {
    const response = await api.delete(`/datasets/${datasetId}/unassign`, { data: { userId } });
    return response.data;
};

export const getDatasetAssignments = async (datasetId) => {
    const response = await api.get(`/datasets/${datasetId}/assignments`);
    return response.data;
};

export const getDatasetPreview = async (id, page = 1) => {
    const response = await api.get(`/datasets/${id}/preview`, { params: { page } });
    return response.data;
};

/**
 * Downloads a dataset file using a temporary link
 */
export const downloadDataset = async (id, fileName) => {
    const response = await api.get(`/datasets/${id}/download`, {
        responseType: 'blob',
    });
    
    // Create a link to download the blob
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', fileName || 'dataset.csv');
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return true;
};

// ---- Password Reset ----
export const forgotPassword = async (email) => {
    const response = await api.post('/auth/forgot-password', { email });
    return response.data;
};

export const resetPassword = async (token, newPassword) => {
    const response = await api.post('/auth/reset-password', { token, newPassword });
    return response.data;
};

export const getCleanedData = async (datasetId, params = {}) => {
    const response = await api.get(`/cleaned-data/${datasetId}`, { params });
    return response.data;
};

export const getAnalysis = async (datasetId) => {
    const response = await api.get(`/datasets/${datasetId}/analysis`);
    return response.data;
};

export const transformDataset = async (datasetId, type, params) => {
    const response = await api.post(`/datasets/${datasetId}/transform`, { type, params });
    return response.data;
};

export const finalizeDataset = async (datasetId) => {
    const response = await api.post(`/datasets/${datasetId}/finalize`);
    return response.data;
};

export const getActivityLogs = async (filters = {}) => {
    const response = await api.get('/activity-logs', { params: filters });
    return response.data;
};

export const getActivityStats = async () => {
    const response = await api.get('/activity-stats');
    return response.data;
};

/** Admin dashboard: query_counts per day from PostgreSQL query_logs */
export const getQueryVolume = async (days = 7) => {
    const response = await api.get('/query-logs/volume', { params: { days } });
    return response.data;
};

// ---- Permissions & Requests ----
export const requestPermission = async (datasetId, permissionType) => {
    const response = await api.post('/permissions/request', { datasetId, permissionType });
    return response.data;
};

export const getPendingPermissions = async () => {
    const response = await api.get('/permissions/pending');
    return response.data;
};

export const resolvePermission = async (requestId, status) => {
    const response = await api.post('/permissions/resolve', { requestId, status });
    return response.data;
};

export const getBadgeCounts = async () => {
    const response = await api.get('/permissions/badge-counts');
    return response.data;
};

// ---- Chat Sessions & Lifecycle ----
export const startChatSession = async (datasetId) => {
    const response = await api.post('/chat/start', { datasetId });
    return response.data;
};

export const endChatSession = async (datasetId, reason) => {
    const response = await api.post('/chat/end', { datasetId, reason });
    return response.data;
};

export const pauseCleaning = async (datasetId) => {
    const response = await api.post(`/datasets/${datasetId}/pause`);
    return response.data;
};

export const logout = async () => {
    const response = await api.post('/auth/logout');
    return response.data;
};

export default api;
