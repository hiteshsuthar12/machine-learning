import axios from 'axios';

// API base URL — uses proxy in dev, direct URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

/**
 * Health check endpoint
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Get model information and metrics
 */
export const getModelInfo = async () => {
  const response = await api.get('/model-info');
  return response.data;
};

/**
 * Submit prediction request
 * @param {Object} formData - Patient data matching the prediction schema
 */
export const submitPrediction = async (formData) => {
  const payload = {
    age: parseInt(formData.age),
    gender: formData.gender,
    polyuria: formData.polyuria,
    polydipsia: formData.polydipsia,
    sudden_weight_loss: formData.sudden_weight_loss,
    weakness: formData.weakness,
    polyphagia: formData.polyphagia,
    genital_thrush: formData.genital_thrush,
    visual_blurring: formData.visual_blurring,
    delayed_healing: formData.delayed_healing,
    muscle_stiffness: formData.muscle_stiffness,
  };

  const response = await api.post('/predict', payload);
  return response.data;
};

export default api;
