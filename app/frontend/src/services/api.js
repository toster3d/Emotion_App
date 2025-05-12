import axios from 'axios';

const API_BASE_URL = '/api/v1';

// Create an axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API health check
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('API Health check error:', error);
    throw error;
  }
};

// Predict emotion from uploaded audio file
export const predictEmotionFromFile = async (file, sampleRate = null) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    if (sampleRate) {
      formData.append('sample_rate', sampleRate);
    }
    
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Predict emotion from file error:', error);
    throw error;
  }
};

// Predict emotion from recorded audio
export const predictEmotionFromRecording = async (audioBlob, sampleRate = 24000) => {
  try {
    const formData = new FormData();
    formData.append('audio_data', audioBlob);
    formData.append('sample_rate', sampleRate);
    
    const response = await api.post('/record', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Predict emotion from recording error:', error);
    throw error;
  }
};

export default api; 