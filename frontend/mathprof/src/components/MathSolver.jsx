import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  Alert,
  CircularProgress
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function MathSolver() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setResponse(null);

    try {
      const result = await axios.post(`${API_URL}/api/v1/solve`, {
        question: question.trim(),
        difficulty_level: 'intermediate'
      });

      setResponse(result.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while solving the problem');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        🧮 Math Routing Agent
      </Typography>
      
      <Typography variant="h6" color="text.secondary" align="center" sx={{ mb: 4 }}>
        Your AI-powered mathematics tutor with step-by-step solutions
      </Typography>

      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              variant="outlined"
              label="Enter your math question"
              placeholder="e.g., Solve the equation 2x + 3 = 7"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
            />
            <Button
              type="submit"
              variant="contained"
              size="large"
              endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
              disabled={loading || !question.trim()}
              sx={{ minWidth: 120, height: 'fit-content' }}
            >
              {loading ? 'Solving...' : 'Solve'}
            </Button>
          </Box>
        </form>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {response && (
        <Card elevation={2}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Chip 
                label={response.source === 'knowledge_base' ? 'Knowledge Base' : 'Web Search'}
                color={response.source === 'knowledge_base' ? 'primary' : 'secondary'}
                size="small"
              />
              <Chip 
                label={`Confidence: ${Math.round(response.confidence * 100)}%`}
                variant="outlined"
                size="small"
              />
            </Box>

            <Typography variant="h6" gutterBottom>
              Question:
            </Typography>
            <Typography variant="body1" paragraph>
              {response.question}
            </Typography>

            <Typography variant="h6" gutterBottom>
              Solution Steps:
            </Typography>
            <Box component="ol" sx={{ pl: 2 }}>
              {response.steps.map((step, index) => (
                <Typography component="li" key={index} sx={{ mb: 1 }}>
                  {step}
                </Typography>
              ))}
            </Box>

            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Final Answer:
            </Typography>
            <Paper elevation={1} sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
              <Typography variant="body1">
                {response.answer}
              </Typography>
            </Paper>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default MathSolver;