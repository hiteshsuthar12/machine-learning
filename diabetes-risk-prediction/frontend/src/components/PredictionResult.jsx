import React from 'react';
import { CheckCircle, XCircle, AlertCircle, RotateCcw, FileText } from 'lucide-react';
import RiskGauge from './RiskGauge';
import FeatureImportance from './FeatureImportance';

const PredictionResult = ({ result, onNewAssessment }) => {
  if (!result) return null;

  const isPositive = result.prediction === 'Positive';

  const predictionStyles = isPositive
    ? { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', label: 'Positive' }
    : { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', label: 'Negative' };

  const PredIcon = predictionStyles.icon;

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Main Result Card */}
      <div className="card">
        <div className="text-center mb-2">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Diabetes Risk Assessment</p>
        </div>

        <div className="flex flex-col lg:flex-row items-center gap-8 py-4">
          {/* Gauge */}
          <div className="flex-shrink-0">
            <RiskGauge probability={result.probability} riskLevel={result.risk_level} />
          </div>

          {/* Details */}
          <div className="flex-1 space-y-4 text-center lg:text-left">
            {/* Prediction */}
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg ${predictionStyles.bg} border ${predictionStyles.border}`}>
              <PredIcon className={`w-5 h-5 ${predictionStyles.color}`} />
              <span className={`font-semibold ${predictionStyles.color}`}>
                Prediction: {predictionStyles.label}
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <p className="text-xs text-slate-500">Probability</p>
                <p className="text-xl font-bold text-white">{(result.probability * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <p className="text-xs text-slate-500">Risk Category</p>
                <p className="text-xl font-bold text-white">{result.risk_level}</p>
              </div>
              <div className="p-3 bg-slate-900/50 rounded-lg col-span-2">
                <p className="text-xs text-slate-500">Model Used</p>
                <p className="text-lg font-semibold text-white">{result.model}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-4 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-amber-200/70">
              {result.disclaimer}
            </p>
          </div>
        </div>
      </div>

      {/* Input Summary */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-4 h-4 text-slate-400" />
          <h3 className="text-lg font-semibold text-white">Input Summary</h3>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {Object.entries(result.input_summary).map(([key, value]) => (
            <div key={key} className="p-3 bg-slate-900/40 rounded-lg">
              <p className="text-xs text-slate-500 truncate">{key}</p>
              <p className={`text-sm font-medium ${
                value === 'Yes' ? 'text-primary-400' : value === 'No' ? 'text-slate-400' : 'text-white'
              }`}>
                {value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Importance */}
      {result.feature_importance && (
        <FeatureImportance featureImportance={result.feature_importance} />
      )}

      {/* New Assessment Button */}
      <div className="flex justify-center">
        <button
          onClick={onNewAssessment}
          className="btn-primary flex items-center gap-2"
          id="new-assessment-btn"
        >
          <RotateCcw className="w-4 h-4" />
          New Assessment
        </button>
      </div>
    </div>
  );
};

export default PredictionResult;
