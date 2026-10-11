import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ReportsPage from './pages/ReportsPage';
import NewReportPage from './pages/NewReportPage';
import MatchingPage from './pages/MatchingPage';
import NLPExperimentPage from './pages/NLPExperimentPage';
import AboutPage from './pages/AboutPage';
import { api } from './services/api';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [selectedMatchItem, setSelectedMatchItem] = useState(null);

  const refreshStats = async () => {
    const data = await api.getStats();
    setStats(data);
  };

  useEffect(() => {
    refreshStats();
  }, [currentTab]);

  const handleMatchItem = (item) => {
    setSelectedMatchItem(item);
    setCurrentTab('matching');
  };

  return (
    <div className="app-container">
      <Sidebar 
        currentTab={currentTab} 
        setCurrentTab={setCurrentTab} 
        stats={stats} 
      />

      <main className="main-wrapper">
        {currentTab === 'dashboard' && (
          <Dashboard stats={stats} onNavigate={setCurrentTab} />
        )}

        {currentTab === 'reports' && (
          <ReportsPage onMatchItem={handleMatchItem} />
        )}

        {currentTab === 'new-report' && (
          <NewReportPage onSuccess={refreshStats} onNavigate={setCurrentTab} />
        )}

        {currentTab === 'matching' && (
          <MatchingPage initialItem={selectedMatchItem} />
        )}

        {currentTab === 'nlp-lab' && (
          <NLPExperimentPage />
        )}

        {currentTab === 'about' && (
          <AboutPage />
        )}
      </main>
    </div>
  );
}
