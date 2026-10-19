import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import PredictionForm from '../components/PredictionForm';
import PredictionResult from '../components/PredictionResult';
import LoadingState from '../components/LoadingState';
import { submitPrediction } from '../services/api';

const Prediction = () => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await submitPrediction(formData);
      setResult(response);
    } catch (err) {
      if (err.response) {
        const detail = err.response.data?.detail;
        if (typeof detail === 'string') {
          setError(detail);
        } else if (Array.isArray(detail)) {
          setError(detail.map(d => d.msg).join('. '));
        } else {
          setError('Invalid input. Please check your entries and try again.');
        }
      } else if (err.request) {
        setError(
          'Unable to connect to the prediction server. Please ensure the backend is running on http://localhost:8000'
        );
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewAssessment = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-dark">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Risk Assessment</h1>
          <p className="text-slate-400 mt-2">
            Enter patient information and symptoms to generate an ML-based diabetes risk prediction.
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3 animate-fade-in">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm text-red-300 font-medium">Prediction Error</p>
              <p className="text-sm text-red-200/70 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && <LoadingState />}

        {/* Result */}
        {!isLoading && result && (
          <PredictionResult result={result} onNewAssessment={handleNewAssessment} />
        )}

        {/* Form (hidden when loading or showing result) */}
        {!isLoading && !result && (
          <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} />
        )}
      </div>
    </div>
  );
};

export default Prediction;
