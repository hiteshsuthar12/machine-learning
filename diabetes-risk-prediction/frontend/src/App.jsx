import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Prediction from './pages/Prediction';
import Dashboard from './pages/Dashboard';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-slate-900">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
