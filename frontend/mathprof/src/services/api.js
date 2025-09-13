import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const mathAPI = {
  solveProblem: async (question, difficultyLevel = 'intermediate') => {
    const response = await api.post('/api/v1/solve', {
      question,
      difficulty_level: difficultyLevel,
    });
    return response.data;
  },

  submitFeedback: async (feedbackId, rating, comments = null) => {
    const response = await api.post('/api/v1/feedback', {
      feedback_id: feedbackId,
      rating,
      comments,
    });
    return response.data;
  },

  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;