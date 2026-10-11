import React from 'react';
import {
  FileText,
  MapPin,
  Calendar,
  Cpu,
  ShieldCheck,
  Award,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import { DashboardStats, NavigationTab } from '../types';

interface OverviewProps {
  onNavigate: (tab: NavigationTab) => void;
  onSelectScenario: (id: string) => void;
  stats: DashboardStats | null;
}

export const Overview: React.FC<OverviewProps> = ({ onNavigate, stats }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Hero Welcome */}
      <div>
        <h2 style={{ fontSize: '26px', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>
          Operations & Review Assistant
        </h2>
        <p style={{ fontSize: '15px', color: 'var(--text-secondary)', marginTop: '6px', maxWidth: '750px' }}>
          Autonomous multi-agent intelligence for restaurant operational diagnosis, customer complaint root-cause analysis,
          evidence grounding, and prioritized corrective action planning.
        </p>
      </div>

      {/* KPI Stats Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '18px' }}>
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Active Review Corpus</span>
            <div style={{ padding: '6px', borderRadius: '8px', background: 'var(--accent-light)', color: 'var(--accent)' }}>
              <FileText size={18} />
            </div>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>{stats ? stats.total_reviews.toLocaleString() : 'Unavailable'}</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {stats ? `${stats.total_complaints.toLocaleString()} complaint signals in the indexed aggregates` : 'Waiting for aggregate data'}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Geographic Coverage</span>
            <div style={{ padding: '6px', borderRadius: '8px', background: '#EAF8EE', color: 'var(--status-success)' }}>
              <MapPin size={18} />
            </div>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>{stats ? `${stats.cities_covered} Metros` : 'Unavailable'}</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {stats?.top_complaint_category ? `Top complaint signal: ${stats.top_complaint_category}` : 'Coverage data unavailable'}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Historical Dataset Cutoff</span>
            <div style={{ padding: '6px', borderRadius: '8px', background: '#FFF6E8', color: 'var(--status-warning)' }}>
              <Calendar size={18} />
            </div>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>{stats?.last_review_month || 'Unavailable'}</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {stats ? `${stats.first_review_month || 'Unknown'} to ${stats.last_review_month || 'Unknown'}` : 'Dataset window unavailable'}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Agent Architecture</span>
            <div style={{ padding: '6px', borderRadius: '8px', background: '#F0E6FF', color: '#8A2BE2' }}>
              <Cpu size={18} />
            </div>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>4 Specialists</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            LangGraph flow + CrewAI synthesis + Google A2A
          </div>
        </div>
      </div>

      {/* System Architecture Overview */}
      <div className="card">
        <h3 style={{ fontSize: '17px', fontWeight: 600, marginBottom: '16px' }}>System Architecture & Workflow</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div style={{ display: 'flex', gap: '14px' }}>
            <div style={{ padding: '10px', borderRadius: '10px', background: 'var(--bg-card-subtle)', height: 'fit-content' }}>
              <ShieldCheck size={20} color="var(--accent)" />
            </div>
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>PII Gate 2 & Reversible Vault</h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Masks customer names, emails, and phone numbers with request-scoped tokens prior to any external LLM prompt, actively preventing sensitive data leakage.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px' }}>
            <div style={{ padding: '10px', borderRadius: '10px', background: 'var(--bg-card-subtle)', height: 'fit-content' }}>
              <TrendingUp size={20} color="var(--status-success)" />
            </div>
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Hybrid RAG Semantic Retrieval</h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Combines BM25 keyword matching with dense MiniLM vector similarity and Reciprocal Rank Fusion (RRF) to retrieve verified review excerpts.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px' }}>
            <div style={{ padding: '10px', borderRadius: '10px', background: 'var(--bg-card-subtle)', height: 'fit-content' }}>
              <Cpu size={20} color="#8A2BE2" />
            </div>
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>CrewAI Report Consolidation</h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Synthesizes completed specialist findings into a cohesive operations report (Drafter Agent) and verifies claim groundedness against evidence (Validator Agent).
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px' }}>
            <div style={{ padding: '10px', borderRadius: '10px', background: 'var(--bg-card-subtle)', height: 'fit-content' }}>
              <Award size={20} color="var(--status-warning)" />
            </div>
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Quality Gate & Confidence Scoring</h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Calculates transparent 0-100 confidence scores based on evidence relevance, source traceability, agent consensus, and data staleness penalties.
              </p>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div style={{ marginTop: '24px', paddingTop: '18px', borderTop: '1px solid var(--border-subtle)', display: 'flex', gap: '12px' }}>
          <button className="btn-primary" onClick={() => onNavigate('analyze')}>
            <span>Launch Analysis Workspace</span>
            <ArrowRight size={16} />
          </button>
          <button className="btn-secondary" onClick={() => onNavigate('agents')}>
            <span>View Agent Heartbeats</span>
          </button>
        </div>
      </div>
    </div>
  );
};
