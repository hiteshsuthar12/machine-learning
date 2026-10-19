import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Home, ClipboardList, BarChart3 } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/predict', label: 'Assessment', icon: ClipboardList },
    { path: '/dashboard', label: 'Dashboard', icon: BarChart3 },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 bg-primary-600 rounded-lg flex items-center justify-center group-hover:bg-primary-500 transition-colors">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div className="hidden sm:block">
              <span className="text-white font-semibold text-lg">DiabetesPredict</span>
              <span className="text-primary-400 text-xs block -mt-1">ML Risk Assessment</span>
            </div>
          </Link>

          {/* Nav Items */}
          <div className="flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-600/20 text-primary-400 border border-primary-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
