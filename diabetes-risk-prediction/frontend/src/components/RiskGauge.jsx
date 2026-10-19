import React from 'react';

const RiskGauge = ({ probability, riskLevel }) => {
  const percentage = Math.round(probability * 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const riskColors = {
    Low: { stroke: '#10b981', bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
    Moderate: { stroke: '#f59e0b', bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
    High: { stroke: '#ef4444', bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30' },
  };

  const colors = riskColors[riskLevel] || riskColors.Moderate;

  return (
    <div className="flex flex-col items-center">
      {/* SVG Gauge */}
      <div className="relative w-40 h-40">
        <svg className="w-40 h-40 transform -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50" cy="50" r="45"
            stroke="#334155"
            strokeWidth="8"
            fill="none"
          />
          {/* Progress circle */}
          <circle
            cx="50" cy="50" r="45"
            stroke={colors.stroke}
            strokeWidth="8"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{
              transition: 'stroke-dashoffset 1.5s ease-out',
            }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-white">{percentage}%</span>
          <span className="text-xs text-slate-400 mt-0.5">probability</span>
        </div>
      </div>

      {/* Risk Level Badge */}
      <div className={`mt-4 px-4 py-2 rounded-lg ${colors.bg} border ${colors.border}`}>
        <span className={`text-sm font-semibold ${colors.text}`}>
          {riskLevel} Risk
        </span>
      </div>
    </div>
  );
};

export default RiskGauge;
