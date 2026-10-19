import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const FeatureImportance = ({ featureImportance }) => {
  if (!featureImportance || Object.keys(featureImportance).length === 0) {
    return null;
  }

  const data = Object.entries(featureImportance)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name: name.length > 18 ? name.substring(0, 16) + '...' : name,
      fullName: name,
      importance: parseFloat((value * 100).toFixed(2)),
    }));

  const maxImportance = Math.max(...data.map(d => d.importance));

  const getBarColor = (value) => {
    const ratio = value / maxImportance;
    if (ratio > 0.7) return '#6366f1';
    if (ratio > 0.4) return '#818cf8';
    return '#a5b4fc';
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0];
      return (
        <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 shadow-lg">
          <p className="text-sm text-white font-medium">{item.payload.fullName}</p>
          <p className="text-sm text-primary-400">{item.value.toFixed(2)}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-white mb-1">Model Feature Importance</h3>
      <p className="text-xs text-slate-500 mb-4">
        Indicates how much each feature contributed to the Random Forest model's decisions.
        This is not a measure of medical causation.
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis 
              type="number" 
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={{ stroke: '#475569' }}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis 
              type="category" 
              dataKey="name" 
              tick={{ fill: '#cbd5e1', fontSize: 11 }}
              axisLine={{ stroke: '#475569' }}
              width={120}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]} maxBarSize={28}>
              {data.map((entry, index) => (
                <Cell key={index} fill={getBarColor(entry.importance)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default FeatureImportance;
