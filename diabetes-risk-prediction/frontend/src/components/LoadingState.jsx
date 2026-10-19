import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingState = ({ message = 'Processing your assessment...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-slate-700 border-t-primary-500 rounded-full animate-spin" />
        <Loader2 className="w-6 h-6 text-primary-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
      </div>
      <p className="mt-6 text-slate-400 text-sm font-medium">{message}</p>
      <p className="mt-2 text-slate-500 text-xs">
        Running ML prediction pipeline...
      </p>
    </div>
  );
};

export default LoadingState;
