import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

const ModelComparison = ({ modelsData }) => {
  if (!modelsData) return null;

  const lr = modelsData.logistic_regression;
  const rf = modelsData.random_forest;

  if (!lr || !rf) return null;

  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
  const metricLabels = {
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1_score: 'F1 Score',
    roc_auc: 'ROC-AUC',
  };

  const barData = metrics.map((m) => ({
    name: metricLabels[m],
    'Logistic Regression': parseFloat((lr.metrics[m].mean * 100).toFixed(2)),
    'Random Forest': parseFloat((rf.metrics[m].mean * 100).toFixed(2)),
  }));

  const radarData = metrics.map((m) => ({
    metric: metricLabels[m],
    lr: parseFloat((lr.metrics[m].mean * 100).toFixed(1)),
    rf: parseFloat((rf.metrics[m].mean * 100).toFixed(1)),
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 shadow-lg">
          <p className="text-sm text-white font-medium mb-1">{label}</p>
          {payload.map((p, i) => (
            <p key={i} className="text-sm" style={{ color: p.color }}>
              {p.name}: {p.value.toFixed(2)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Comparison Table */}
      <div className="card overflow-x-auto">
        <h3 className="text-lg font-semibold text-white mb-4">Cross-Validation Results</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Metric</th>
              <th className="text-center py-3 px-4 text-blue-400 font-medium">Logistic Regression</th>
              <th className="text-center py-3 px-4 text-orange-400 font-medium">Random Forest</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m} className="border-b border-slate-700/50 hover:bg-slate-700/20 transition-colors">
                <td className="py-3 px-4 text-slate-300">{metricLabels[m]}</td>
                <td className="py-3 px-4 text-center text-slate-200">
                  {(lr.metrics[m].mean * 100).toFixed(2)}%
                  <span className="text-slate-500 text-xs ml-1">(±{(lr.metrics[m].std * 100).toFixed(2)})</span>
                </td>
                <td className="py-3 px-4 text-center text-slate-200">
                  {(rf.metrics[m].mean * 100).toFixed(2)}%
                  <span className="text-slate-500 text-xs ml-1">(±{(rf.metrics[m].std * 100).toFixed(2)})</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Metrics Comparison</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={{ stroke: '#475569' }} />
                <YAxis domain={[50, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickFormatter={(v) => `${v}%`} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                <Bar dataKey="Logistic Regression" fill="#3b82f6" radius={[4, 4, 0, 0]} maxBarSize={40} />
                <Bar dataKey="Random Forest" fill="#f97316" radius={[4, 4, 0, 0]} maxBarSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Performance Radar</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <PolarRadiusAxis domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 9 }} />
                <Radar name="Logistic Regression" dataKey="lr" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Random Forest" dataKey="rf" stroke="#f97316" fill="#f97316" fillOpacity={0.15} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
