import React from 'react';
import { TrendingUp, Clock, Zap } from 'lucide-react';

const WorkflowStats = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-white/10">
      <div className="flex items-center gap-2 mb-2">
        <TrendingUp className="w-4 h-4 text-cyan-400" />
        <span className="text-sm font-medium text-cyan-400">Workflow Stats</span>
      </div>
      <div className="grid grid-cols-3 gap-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="text-gray-400">Intent:</span>
          <span className="text-white font-medium">{stats.intent || 'N/A'}</span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3 text-gray-400" />
          <span className="text-gray-400">Duration:</span>
          <span className="text-white font-medium">{stats.total_duration_ms || 0}ms</span>
        </div>
        <div className="flex items-center gap-1">
          <Zap className="w-3 h-3 text-gray-400" />
          <span className="text-gray-400">API Calls:</span>
          <span className="text-white font-medium">{stats.api_calls_count || 0}</span>
        </div>
      </div>
    </div>
  );
};

export default WorkflowStats;