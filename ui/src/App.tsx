import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Overview } from './components/Overview';
import { AnalysisWorkspace } from './components/AnalysisWorkspace';
import { ResultsDashboard } from './components/ResultsDashboard';
import { ExecutionTraceView } from './components/ExecutionTraceView';
import { AgentActivityView } from './components/AgentActivityView';
import { FeedbackView } from './components/FeedbackView';
import { FeedbackModal } from './components/FeedbackModal';
import { SpecialistCardsView } from './components/SpecialistCardsView';
import {
  NavigationTab,
  AnalysisResponse,
  AgentStatusResponse,
  DashboardStats,
} from './types';
import { ApiClient } from './api/client';
import { AlertCircle, X } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavigationTab>('overview');
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [agentStatus, setAgentStatus] = useState<AgentStatusResponse | null>(null);
  const [isAgentLoading, setIsAgentLoading] = useState<boolean>(false);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);

  const [activeResult, setActiveResult] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isFeedbackModalOpen, setIsFeedbackModalOpen] = useState<boolean>(false);

  // Initial health and status poll
  const checkHealthAndAgents = () => {
    ApiClient.getHealth()
      .then((h) => setBackendHealthy(h.status === 'healthy'))
      .catch(() => setBackendHealthy(false));

    setIsAgentLoading(true);
    ApiClient.getAgentStatuses()
      .then((data) => setAgentStatus(data))
      .catch((err) => console.warn('Could not fetch agent statuses:', err))
      .finally(() => setIsAgentLoading(false));
    ApiClient.getDashboardStats()
      .then((data) => setDashboardStats(data))
      .catch((err) => console.warn('Could not fetch dashboard statistics:', err));
  };

  useEffect(() => {
    checkHealthAndAgents();
    const interval = setInterval(checkHealthAndAgents, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleRunAnalysis = async (params: {
    query: string;
    mode: 'auto' | 'agent';
    agent?: string;
    business_id?: string;
    start_date?: string;
    end_date?: string;
    city?: string;
  }) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    try {
      const res = await ApiClient.analyze(params);
      setActiveResult(res);
      setCurrentTab('analyze');
    } catch (err: any) {
      setErrorMsg(err.message || 'Analysis failed. Check backend connection.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        backendHealthy={backendHealthy}
        agentStatus={agentStatus}
        dashboardStats={dashboardStats}
        hasActiveResult={activeResult !== null}
      />

      {/* Main Workspace Area */}
      <main className="main-content">
        {/* Error Alert Banner */}
        {errorMsg && (
          <div
            className="card"
            style={{
              background: 'var(--status-danger-bg)',
              borderColor: 'var(--status-danger)',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px 18px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertCircle size={18} color="var(--status-danger)" />
              <span style={{ fontSize: '13.5px', color: 'var(--status-danger)', fontWeight: 500 }}>
                {errorMsg}
              </span>
            </div>
            <button onClick={() => setErrorMsg(null)} style={{ color: 'var(--status-danger)' }}>
              <X size={16} />
            </button>
          </div>
        )}

        {/* Dynamic Tab Views */}
        {currentTab === 'overview' && (
          <Overview
            onNavigate={setCurrentTab}
            onSelectScenario={() => setCurrentTab('analyze')}
            stats={dashboardStats}
          />
        )}

        {currentTab === 'analyze' && (
          <AnalysisWorkspace
            onRunAnalysis={handleRunAnalysis}
            isLoading={isAnalyzing}
            resultsSlot={
              activeResult ? (
                <ResultsDashboard
                  result={activeResult}
                  onOpenFeedback={() => setIsFeedbackModalOpen(true)}
                />
              ) : null
            }
          />
        )}

        {currentTab === 'categories' && (activeResult ? <SpecialistCardsView type="categories" result={activeResult} /> : <EmptyAnalysisState />)}

        {currentTab === 'trends' && (activeResult ? <SpecialistCardsView type="trends" result={activeResult} /> : <EmptyAnalysisState />)}

        {currentTab === 'root_cause' && (activeResult ? <SpecialistCardsView type="root_cause" result={activeResult} /> : <EmptyAnalysisState />)}

        {currentTab === 'actions' && (activeResult ? <SpecialistCardsView type="actions" result={activeResult} /> : <EmptyAnalysisState />)}

        {currentTab === 'trace' && (activeResult ? <ExecutionTraceView
            events={activeResult.execution_trace || []}
            requestId={activeResult.request_id}
            totalDurationMs={activeResult.total_duration_ms}
          /> : <EmptyAnalysisState />)}

        {currentTab === 'agents' && (
          <AgentActivityView
            statusData={agentStatus}
            onRefresh={checkHealthAndAgents}
            isLoading={isAgentLoading}
          />
        )}

        {currentTab === 'feedback' && (
          <FeedbackView onOpenModal={() => setIsFeedbackModalOpen(true)} />
        )}

        {/* Global Feedback Modal */}
        <FeedbackModal
          isOpen={isFeedbackModalOpen}
          onClose={() => setIsFeedbackModalOpen(false)}
          requestId={activeResult ? activeResult.request_id : 'req-general'}
        />
      </main>
    </div>
  );
};

export default App;

const EmptyAnalysisState: React.FC = () => (
  <div className="card" style={{ padding: '28px' }}>
    <h2 style={{ fontSize: '20px', color: 'var(--text-primary)', marginBottom: '8px' }}>No active analysis</h2>
    <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
      Run a question in Analyze Operations to populate this view with grounded findings.
    </p>
  </div>
);
