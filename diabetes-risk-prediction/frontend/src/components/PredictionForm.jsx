import React from 'react';
import { User, Stethoscope } from 'lucide-react';

const INITIAL_FORM_DATA = {
  age: '',
  gender: 'Male',
  polyuria: 'No',
  polydipsia: 'No',
  sudden_weight_loss: 'No',
  weakness: 'No',
  polyphagia: 'No',
  genital_thrush: 'No',
  visual_blurring: 'No',
  delayed_healing: 'No',
  muscle_stiffness: 'No',
};

const SYMPTOM_DESCRIPTIONS = {
  polyuria: { label: 'Polyuria', desc: 'Excessive urination' },
  polydipsia: { label: 'Polydipsia', desc: 'Excessive thirst' },
  sudden_weight_loss: { label: 'Sudden Weight Loss', desc: 'Unexplained weight loss' },
  weakness: { label: 'Weakness', desc: 'General body weakness' },
  polyphagia: { label: 'Polyphagia', desc: 'Excessive hunger' },
  genital_thrush: { label: 'Genital Thrush', desc: 'Fungal infection' },
  visual_blurring: { label: 'Visual Blurring', desc: 'Blurred vision' },
  delayed_healing: { label: 'Delayed Healing', desc: 'Slow wound healing' },
  muscle_stiffness: { label: 'Muscle Stiffness', desc: 'Stiff muscles' },
};

const ToggleSwitch = ({ value, onChange, id }) => (
  <div className="toggle-group">
    <button
      type="button"
      id={`${id}-yes`}
      className={`toggle-option ${value === 'Yes' ? 'active' : ''}`}
      onClick={() => onChange('Yes')}
    >
      Yes
    </button>
    <button
      type="button"
      id={`${id}-no`}
      className={`toggle-option ${value === 'No' ? 'active' : ''}`}
      onClick={() => onChange('No')}
    >
      No
    </button>
  </div>
);

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = React.useState(INITIAL_FORM_DATA);
  const [errors, setErrors] = React.useState({});

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  const validate = () => {
    const newErrors = {};
    const age = parseInt(formData.age);
    if (!formData.age || isNaN(age)) {
      newErrors.age = 'Please enter a valid age';
    } else if (age < 1 || age > 120) {
      newErrors.age = 'Age must be between 1 and 120';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      onSubmit(formData);
    }
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8 animate-fade-in">
      {/* Patient Information Section */}
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-primary-600/20 rounded-lg flex items-center justify-center">
            <User className="w-5 h-5 text-primary-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Patient Information</h3>
            <p className="text-sm text-slate-400">Basic demographic details</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Age */}
          <div>
            <label htmlFor="age-input" className="label">Age <span className="text-red-400">*</span></label>
            <input
              id="age-input"
              type="number"
              min="1"
              max="120"
              placeholder="Enter age (1-120)"
              value={formData.age}
              onChange={(e) => handleChange('age', e.target.value)}
              className={`input-field ${errors.age ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
            />
            {errors.age && (
              <p className="text-red-400 text-xs mt-1">{errors.age}</p>
            )}
          </div>

          {/* Gender */}
          <div>
            <label className="label">Gender <span className="text-red-400">*</span></label>
            <div className="toggle-group">
              <button
                type="button"
                id="gender-male"
                className={`toggle-option ${formData.gender === 'Male' ? 'active' : ''}`}
                onClick={() => handleChange('gender', 'Male')}
              >
                Male
              </button>
              <button
                type="button"
                id="gender-female"
                className={`toggle-option ${formData.gender === 'Female' ? 'active' : ''}`}
                onClick={() => handleChange('gender', 'Female')}
              >
                Female
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Symptoms Section */}
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-accent-600/20 rounded-lg flex items-center justify-center">
            <Stethoscope className="w-5 h-5 text-accent-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Symptoms Assessment</h3>
            <p className="text-sm text-slate-400">Select symptoms currently present</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {Object.entries(SYMPTOM_DESCRIPTIONS).map(([key, { label, desc }]) => (
            <div
              key={key}
              className={`flex items-center justify-between p-4 rounded-lg border transition-all duration-200 ${
                formData[key] === 'Yes'
                  ? 'bg-primary-600/10 border-primary-500/30'
                  : 'bg-slate-900/30 border-slate-700/50 hover:border-slate-600'
              }`}
            >
              <div className="flex-1 min-w-0 mr-3">
                <p className="text-sm font-medium text-white">{label}</p>
                <p className="text-xs text-slate-400 truncate">{desc}</p>
              </div>
              <ToggleSwitch
                id={key}
                value={formData[key]}
                onChange={(v) => handleChange(key, v)}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 justify-end">
        <button type="button" onClick={handleReset} className="btn-secondary" id="reset-btn">
          Reset Form
        </button>
        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary flex items-center justify-center gap-2"
          id="submit-btn"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Processing...
            </>
          ) : (
            'Run Assessment'
          )}
        </button>
      </div>
    </form>
  );
};

export default PredictionForm;
