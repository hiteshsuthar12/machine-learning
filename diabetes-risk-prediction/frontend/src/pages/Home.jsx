import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, Brain, BarChart3, Shield, ArrowRight, GitBranch, Layers } from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: Brain,
      title: 'Machine Learning Models',
      desc: 'Logistic Regression & Random Forest classifiers trained on clinical symptom data',
    },
    {
      icon: GitBranch,
      title: '5-Fold Cross-Validation',
      desc: 'Stratified K-Fold CV ensures reliable performance estimates across balanced folds',
    },
    {
      icon: BarChart3,
      title: 'Model Explainability',
      desc: 'Feature importance analysis shows which symptoms drive predictions',
    },
    {
      icon: Shield,
      title: 'Risk Assessment',
      desc: 'Probability-based risk categorization with clear confidence indicators',
    },
  ];

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-dark">
      {/* Hero Section */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-12">
        <div className="text-center animate-fade-in">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-primary-600/10 border border-primary-500/20 rounded-full mb-8">
            <Activity className="w-4 h-4 text-primary-400" />
            <span className="text-sm text-primary-300 font-medium">ML-Based Risk Prediction</span>
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
            Diabetes Risk
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-accent-400">
              Prediction System
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-4">
            Machine learning-based diabetes risk assessment using patient demographics
            and symptom data, powered by Logistic Regression and Random Forest models
            with Stratified 5-Fold Cross-Validation.
          </p>

          <p className="text-xs text-slate-500 mb-10 max-w-lg mx-auto">
            This is an ML application for educational and research purposes. It is NOT a medical diagnostic tool.
            Always consult a qualified healthcare professional.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/predict"
              className="btn-primary text-lg px-8 py-4 flex items-center gap-2 group"
              id="start-assessment-btn"
            >
              Start Assessment
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/dashboard"
              className="btn-secondary text-lg px-8 py-4 flex items-center gap-2"
              id="view-dashboard-btn"
            >
              <BarChart3 className="w-5 h-5" />
              View Dashboard
            </Link>
          </div>
        </div>
      </div>

      {/* Tech Stack Pills */}
      <div className="max-w-3xl mx-auto px-4 mb-16">
        <div className="flex flex-wrap items-center justify-center gap-2">
          {['Logistic Regression', 'Random Forest', '5-Fold CV', 'FastAPI', 'React', 'scikit-learn'].map((tech) => (
            <span
              key={tech}
              className="px-3 py-1 text-xs font-medium text-slate-400 bg-slate-800/60 border border-slate-700/50 rounded-full"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {features.map(({ icon: Icon, title, desc }, index) => (
            <div
              key={title}
              className="card-hover"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-primary-600/15 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Icon className="w-5 h-5 text-primary-400" />
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">{title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">System Architecture</h3>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
            {[
              'Patient Input',
              'React Frontend',
              'FastAPI Backend',
              'sklearn Pipeline',
              'Random Forest Model',
              'Risk Prediction',
            ].map((step, i, arr) => (
              <React.Fragment key={step}>
                <span className="px-3 py-2 bg-slate-900/60 border border-slate-700/50 rounded-lg text-slate-300 font-medium">
                  {step}
                </span>
                {i < arr.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-slate-600 hidden sm:block" />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
