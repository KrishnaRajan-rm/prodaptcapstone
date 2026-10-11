import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Bot,
  Filter,
  Play,
  RotateCcw,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Building2,
  Calendar,
} from 'lucide-react';
import { ScenarioItem } from '../types';
import { ApiClient } from '../api/client';

interface AnalysisWorkspaceProps {
  onRunAnalysis: (params: {
    query: string;
    mode: 'auto' | 'agent';
    agent?: string;
    business_id?: string;
    start_date?: string;
    end_date?: string;
    city?: string;
  }) => void;
  isLoading: boolean;
  activeScenarioId?: string;
  resultsSlot?: React.ReactNode;
}

export const AnalysisWorkspace: React.FC<AnalysisWorkspaceProps> = ({
  onRunAnalysis,
  isLoading,
  resultsSlot,
}) => {
  const [query, setQuery] = useState(
    'Customers are repeatedly complaining about slow service and long wait times. What operational issues are causing this, and what should we fix first?'
  );
  const [mode, setMode] = useState<'auto' | 'agent'>('auto');
  const [agent, setAgent] = useState<string>('classification');
  const [businessId, setBusinessId] = useState<string>('_ab50qdWOk0DdB6XOrBitw');
  const [startDate, setStartDate] = useState<string>('2020-01');
  const [endDate, setEndDate] = useState<string>('2022-01');
  const [city, setCity] = useState<string>('New Orleans');
  const [showFilters, setShowFilters] = useState<boolean>(true);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [activeStepIndex, setActiveStepIndex] = useState<number>(0);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  const pipelineStages = [
    { title: '1. Input Guardrails', desc: 'Validating safety, constraints & format' },
    { title: '2. Hybrid RAG Retrieval', desc: 'Searching dense vectors + lexical BM25 fusion' },
    { title: '3. PII Interceptor Gate 2', desc: 'Presidio masking & token vaulting' },
    { title: '4. Parallel Specialist Agents', desc: 'Classification & Trend Analysis execution' },
    { title: '5. Root-Cause Analysis', desc: 'Synthesizing evidence-backed hypotheses' },
    { title: '6. Action Planner Agent', desc: 'Formulating ranked actions & KPIs' },
    { title: '7. CrewAI Consensus Drafter', desc: 'Synthesizing unified manager report' },
    { title: '8. CrewAI Report Validator', desc: 'Grounding verification & privacy audit' },
    { title: '9. Master Quality Gate', desc: 'Grounding, staleness & confidence scoring' },
    { title: '10. Reversible PII Decoder', desc: 'Decoding context for authorized manager' },
  ];

  useEffect(() => {
    let timer: any;
    let stepTimer: any;
    if (isLoading) {
      setElapsedSeconds(0);
      setActiveStepIndex(0);
      timer = setInterval(() => {
        setElapsedSeconds((s) => s + 1);
      }, 1000);
      stepTimer = setInterval(() => {
        setActiveStepIndex((prev) => (prev < pipelineStages.length - 1 ? prev + 1 : prev));
      }, 1000);
    } else {
      setActiveStepIndex(0);
      setElapsedSeconds(0);
    }
    return () => {
      clearInterval(timer);
      clearInterval(stepTimer);
    };
  }, [isLoading]);

  useEffect(() => {
    ApiClient.getScenarios()
      .then((data) => setScenarios(data))
      .catch((err) => console.warn('Could not fetch scenarios:', err));
  }, []);

  const handleApplyScenario = (sc: ScenarioItem) => {
    setQuery(sc.sample_query);
    setMode(sc.mode);
    if (sc.agent) setAgent(sc.agent);
    if (sc.business_id) setBusinessId(sc.business_id);
    if (sc.suggested_start_date) setStartDate(sc.suggested_start_date);
    if (sc.suggested_end_date) setEndDate(sc.suggested_end_date);
    if (sc.city) setCity(sc.city);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    onRunAnalysis({
      query: query.trim(),
      mode,
      agent: mode === 'agent' ? agent : undefined,
      business_id: businessId.trim() || undefined,
      start_date: startDate.trim() || undefined,
      end_date: endDate.trim() || undefined,
      city: city.trim() || undefined,
    });
  };

  const handleReset = () => {
    setQuery('');
    setBusinessId('');
    setStartDate('2020-01');
    setEndDate('2022-01');
    setCity('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Header */}
      <div>
        <h2 style={{ fontSize: '24px', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>
          Operational Analysis Workspace
        </h2>
        <p style={{ fontSize: '14.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Query customer reviews and restaurant operations using automatic multi-agent routing or direct specialist diagnosis.
        </p>
      </div>

      {/* Main Query Card */}
      <form onSubmit={handleSubmit} className="card" style={{ padding: '24px' }}>
        {/* Mode Toggle Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'inline-flex', background: 'var(--bg-card-subtle)', padding: '3px', borderRadius: 'var(--radius-md)' }}>
            <button
              type="button"
              onClick={() => setMode('auto')}
              style={{
                padding: '7px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: mode === 'auto' ? 600 : 500,
                color: mode === 'auto' ? 'var(--text-primary)' : 'var(--text-secondary)',
                backgroundColor: mode === 'auto' ? '#FFFFFF' : 'transparent',
                boxShadow: mode === 'auto' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.15s ease',
              }}
            >
              <Sparkles size={15} color={mode === 'auto' ? 'var(--accent)' : 'currentColor'} />
              <span>Auto Workflow (LangGraph + CrewAI)</span>
            </button>
            <button
              type="button"
              onClick={() => setMode('agent')}
              style={{
                padding: '7px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: mode === 'agent' ? 600 : 500,
                color: mode === 'agent' ? 'var(--text-primary)' : 'var(--text-secondary)',
                backgroundColor: mode === 'agent' ? '#FFFFFF' : 'transparent',
                boxShadow: mode === 'agent' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.15s ease',
              }}
            >
              <Bot size={15} color={mode === 'agent' ? 'var(--accent)' : 'currentColor'} />
              <span>Individual Specialist Agent</span>
            </button>
          </div>

          {/* If Single Agent selected, show dropdown */}
          {mode === 'agent' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Select Agent:</span>
              <select
                value={agent}
                onChange={(e) => setAgent(e.target.value)}
                style={{ padding: '6px 12px', fontSize: '13px', fontWeight: 500 }}
              >
                <option value="classification">Classification Agent (Aspects & Severity)</option>
                <option value="trends">Trend Analysis Agent (Historical Slopes)</option>
                <option value="root_cause">Root-Cause Agent (Diagnostic Hypotheses)</option>
                <option value="action_planner">Action Planner Agent (Ranked Actions & KPIs)</option>
              </select>
            </div>
          )}
        </div>

        {/* Textarea */}
        <div style={{ position: 'relative', marginBottom: '16px' }}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter operational question or customer experience issue..."
            rows={4}
            style={{
              width: '100%',
              resize: 'vertical',
              fontSize: '14.5px',
              lineHeight: '1.6',
              padding: '14px 16px',
            }}
          />
        </div>

        {/* Filter Toggle & Filter Inputs */}
        <div style={{ marginBottom: '20px' }}>
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              color: 'var(--text-secondary)',
              marginBottom: showFilters ? '12px' : '0',
            }}
          >
            <Filter size={14} />
            <span>{showFilters ? 'Hide Operational Metadata Filters' : 'Show Operational Metadata Filters'}</span>
          </button>

          {showFilters && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '12px',
                padding: '14px',
                background: 'var(--bg-card-subtle)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Business ID (Optional)
                </label>
                <input
                  type="text"
                  value={businessId}
                  onChange={(e) => setBusinessId(e.target.value)}
                  placeholder="e.g. _ab50qdWOk0DdB6XOrBitw"
                  style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  City / Metro (Optional)
                </label>
                <input
                  type="text"
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  placeholder="e.g. New Orleans, Philadelphia"
                  style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Start Date (YYYY-MM)
                </label>
                <input
                  type="text"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  placeholder="2020-01"
                  style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  End Date (YYYY-MM)
                </label>
                <input
                  type="text"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  placeholder="2022-01"
                  style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Live Execution State Stepper */}
        {isLoading && (
          <div
            style={{
              marginTop: '16px',
              marginBottom: '16px',
              padding: '18px 20px',
              borderRadius: '8px',
              background: 'var(--bg-card-subtle)',
              border: '1px solid var(--accent)',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span
                  style={{
                    width: '16px',
                    height: '16px',
                    border: '2.5px solid var(--accent)',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    display: 'inline-block',
                    animation: 'spin 0.8s linear infinite',
                  }}
                />
                <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Active Execution State: {pipelineStages[activeStepIndex].title}
                </span>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '12px' }}>
                Elapsed: {elapsedSeconds}s • Stage {activeStepIndex + 1} of {pipelineStages.length}
              </span>
            </div>

            {/* Progress Bar */}
            <div style={{ width: '100%', height: '6px', background: 'var(--border-subtle)', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${((activeStepIndex + 1) / pipelineStages.length) * 100}%`,
                  height: '100%',
                  background: 'var(--accent)',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>

            {/* Stepper items */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
                gap: '8px',
                marginTop: '4px',
              }}
            >
              {pipelineStages.map((stg, i) => {
                const isDone = i < activeStepIndex;
                const isCurrent = i === activeStepIndex;
                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      fontSize: '12px',
                      color: isCurrent ? 'var(--accent)' : isDone ? 'var(--text-primary)' : 'var(--text-tertiary)',
                      fontWeight: isCurrent ? 700 : isDone ? 500 : 400,
                    }}
                  >
                    {isDone ? (
                      <CheckCircle2 size={14} color="var(--status-success)" />
                    ) : isCurrent ? (
                      <span
                        style={{
                          width: '10px',
                          height: '10px',
                          borderRadius: '50%',
                          background: 'var(--accent)',
                          display: 'inline-block',
                          boxShadow: '0 0 6px var(--accent)',
                        }}
                      />
                    ) : (
                      <span
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          background: 'var(--border-strong)',
                          display: 'inline-block',
                        }}
                      />
                    )}
                    <span>{stg.title}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={handleReset}
            disabled={isLoading}
          >
            <RotateCcw size={15} />
            <span>Clear</span>
          </button>

          <button
            type="submit"
            className="btn-primary"
            disabled={isLoading || !query.trim()}
            style={{ padding: '10px 24px' }}
          >
            {isLoading ? (
              <>
                <span
                  style={{
                    width: '14px',
                    height: '14px',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderTopColor: '#FFFFFF',
                    borderRadius: '50%',
                    display: 'inline-block',
                    animation: 'spin 0.8s linear infinite',
                  }}
                />
                <span>Synthesizing Multi-Agent Analysis...</span>
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                <span>Run Analysis</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Generated Operational Answer & Results Dashboard (Directly Below Question) */}
      {resultsSlot && (
        <div style={{ marginTop: '8px', marginBottom: '8px' }}>
          {resultsSlot}
        </div>
      )}

      {/* Quick Practice Scenarios Grid */}
      <div style={{ marginTop: resultsSlot ? '16px' : '0' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Zap size={18} color="var(--accent)" />
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Practice Demonstration Scenarios
            </h3>
            <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
              (Grounded in Yelp Open Dataset)
            </span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          {scenarios.map((sc) => (
            <div
              key={sc.id}
              className="card card-clickable"
              onClick={() => handleApplyScenario(sc)}
              style={{
                cursor: 'pointer',
                padding: '16px 18px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {sc.title}
                  </span>
                  <span className={sc.mode === 'auto' ? 'badge badge-neutral' : 'badge badge-warning'}>
                    {sc.mode === 'auto' ? 'Auto Flow' : 'Single Agent'}
                  </span>
                </div>
                <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '10px' }}>
                  {sc.description}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-tertiary)', borderTop: '1px solid var(--border-subtle)', paddingTop: '8px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Building2 size={12} />
                  <span>{sc.business_name || 'General Scope'}</span>
                </span>
                <span style={{ color: 'var(--accent)', fontWeight: 500 }}>Load Scenario →</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
