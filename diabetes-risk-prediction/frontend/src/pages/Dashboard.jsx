import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Server, Database, Cpu, TrendingUp } from 'lucide-react';
import ModelComparison from '../components/ModelComparison';
import FeatureImportance from '../components/FeatureImportance';
import { getModelInfo, checkHealth } from '../services/api';

const MetricCard = ({ label, value, icon: Icon, color = 'primary' }) => {
  const colors = {
    primary: 'text-primary-400 bg-primary-600/15',
    accent: 'text-accent-400 bg-accent-600/15',
    amber: 'text-amber-400 bg-amber-600/15',
    blue: 'text-blue-400 bg-blue-600/15',
    purple: 'text-purple-400 bg-purple-600/15',
  };

  return (
    <div className="card-hover">
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colors[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <p className="metric-value text-2xl">{value}</p>
          <p className="metric-label">{label}</p>
        </div>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [infoData, healthData] = await Promise.all([
          getModelInfo(),
          checkHealth(),
        ]);
        setModelInfo(infoData);
        setHealth(healthData);
      } catch (err) {
        setError(
          err.request
            ? 'Unable to connect to the backend. Ensure the FastAPI server is running.'
            : 'Failed to load model information.'
        );
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-dark flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-slate-700 border-t-primary-500 rounded-full animate-spin mx-auto" />
          <p className="mt-4 text-slate-400">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-dark flex items-center justify-center p-4">
        <div className="card max-w-md text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Connection Error</h2>
          <p className="text-slate-400 text-sm">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="btn-primary mt-4"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const rfMetrics = modelInfo?.models?.random_forest?.metrics;
  const selectedModel = modelInfo?.selected_model || 'Random Forest';

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Model Dashboard</h1>
          <p className="text-slate-400 mt-2">
            Performance metrics and model comparison from 5-Fold Stratified Cross-Validation
          </p>
        </div>

        {/* Status Bar */}
        <div className="flex flex-wrap gap-3 mb-8">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-full">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-xs text-emerald-300 font-medium">API {health?.status}</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-primary-500/10 border border-primary-500/30 rounded-full">
            <Cpu className="w-3.5 h-3.5 text-primary-400" />
            <span className="text-xs text-primary-300 font-medium">Model: {selectedModel}</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-700/50 border border-slate-600/50 rounded-full">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs text-slate-300 font-medium">{modelInfo?.dataset?.n_samples} samples</span>
          </div>
        </div>

        {/* Top Metrics */}
        {rfMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
            <MetricCard
              label="Accuracy"
              value={`${(rfMetrics.accuracy.mean * 100).toFixed(1)}%`}
              icon={CheckCircle}
              color="accent"
            />
            <MetricCard
              label="Precision"
              value={`${(rfMetrics.precision.mean * 100).toFixed(1)}%`}
              icon={TrendingUp}
              color="blue"
            />
            <MetricCard
              label="Recall"
              value={`${(rfMetrics.recall.mean * 100).toFixed(1)}%`}
              icon={Server}
              color="amber"
            />
            <MetricCard
              label="F1 Score"
              value={`${(rfMetrics.f1_score.mean * 100).toFixed(1)}%`}
              icon={Cpu}
              color="purple"
            />
            <MetricCard
              label="ROC-AUC"
              value={`${(rfMetrics.roc_auc.mean * 100).toFixed(1)}%`}
              icon={TrendingUp}
              color="primary"
            />
          </div>
        )}

        {/* Model Comparison */}
        <div className="mb-8">
          <ModelComparison modelsData={modelInfo?.models} />
        </div>

        {/* Feature Importance */}
        {modelInfo?.models?.random_forest?.feature_importance && (
          <div className="mb-8">
            <FeatureImportance
              featureImportance={modelInfo.models.random_forest.feature_importance}
            />
          </div>
        )}

        {/* Confusion Matrices */}
        {modelInfo?.models && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {['logistic_regression', 'random_forest'].map((modelKey) => {
              const model = modelInfo.models[modelKey];
              if (!model?.confusion_matrix) return null;
              const cm = model.confusion_matrix;
              const name = model.name;

              return (
                <div key={modelKey} className="card">
                  <h3 className="text-lg font-semibold text-white mb-4">Confusion Matrix: {name}</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr>
                          <th className="py-2 px-3 text-slate-500"></th>
                          <th className="py-2 px-3 text-center text-slate-400">Pred. Negative</th>
                          <th className="py-2 px-3 text-center text-slate-400">Pred. Positive</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="py-2 px-3 text-slate-400 font-medium">Actual Negative</td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center justify-center w-14 h-10 bg-emerald-500/15 text-emerald-400 font-bold rounded-lg">
                              {cm[0][0]}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center justify-center w-14 h-10 bg-red-500/15 text-red-400 font-bold rounded-lg">
                              {cm[0][1]}
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td className="py-2 px-3 text-slate-400 font-medium">Actual Positive</td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center justify-center w-14 h-10 bg-red-500/15 text-red-400 font-bold rounded-lg">
                              {cm[1][0]}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center justify-center w-14 h-10 bg-emerald-500/15 text-emerald-400 font-bold rounded-lg">
                              {cm[1][1]}
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Model Details */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Training Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Training Date</span>
                <span className="text-slate-200">{modelInfo?.training_date?.split('T')[0] || 'N/A'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Dataset Samples</span>
                <span className="text-slate-200">{modelInfo?.dataset?.n_samples}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Features</span>
                <span className="text-slate-200">{modelInfo?.dataset?.n_features}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Target</span>
                <span className="text-slate-200">{modelInfo?.dataset?.target}</span>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">CV Method</span>
                <span className="text-slate-200">{modelInfo?.cross_validation?.method}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">CV Folds</span>
                <span className="text-slate-200">{modelInfo?.cross_validation?.n_splits}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Selected Model</span>
                <span className="text-slate-200">{modelInfo?.selected_model}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-700/50">
                <span className="text-slate-400">Selection Criterion</span>
                <span className="text-slate-200 text-right">{modelInfo?.selection_criterion}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
