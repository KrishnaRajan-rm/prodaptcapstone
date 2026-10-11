import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  Layers,
  TrendingUp,
  AlertCircle,
  CheckSquare,
  GitCommit,
  Activity,
  Award,
  UtensilsCrossed,
} from 'lucide-react';
import { NavigationTab, AgentStatusResponse, DashboardStats } from '../types';

interface SidebarProps {
  currentTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  backendHealthy: boolean;
  agentStatus: AgentStatusResponse | null;
  hasActiveResult: boolean;
  dashboardStats: DashboardStats | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  backendHealthy,
  agentStatus,
  hasActiveResult,
  dashboardStats,
}) => {
  const navItems = [
    { id: 'overview' as NavigationTab, label: 'Overview', icon: LayoutDashboard },
    { id: 'analyze' as NavigationTab, label: 'Analyze Operations', icon: Sparkles },
    { id: 'categories' as NavigationTab, label: 'Categories & Aspects', icon: Layers },
    { id: 'trends' as NavigationTab, label: 'Trends & Forecasts', icon: TrendingUp },
    { id: 'root_cause' as NavigationTab, label: 'Root Causes', icon: AlertCircle },
    { id: 'actions' as NavigationTab, label: 'Action Planner', icon: CheckSquare },
    { id: 'trace' as NavigationTab, label: 'Execution Trace', icon: GitCommit, disabled: !hasActiveResult },
    { id: 'agents' as NavigationTab, label: 'Agent Activity', icon: Activity },
    { id: 'feedback' as NavigationTab, label: 'Feedback & Eval', icon: Award },
  ];

  const availableCount = agentStatus ? agentStatus.total_available : 0;
  const configuredCount = agentStatus ? agentStatus.total_configured : 4;

  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '28px', padding: '0 8px' }}>
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            backgroundColor: '#0071E3',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#FFFFFF',
          }}
        >
          <UtensilsCrossed size={20} />
        </div>
        <div>
          <h1 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>
            Restaurant Ops
          </h1>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>AI Review Assistant</p>
        </div>
      </div>

      {/* Navigation List */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => !item.disabled && onSelectTab(item.id)}
              disabled={item.disabled}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                fontSize: '13.5px',
                fontWeight: isActive ? 600 : 500,
                color: isActive ? 'var(--accent)' : item.disabled ? 'var(--text-tertiary)' : 'var(--text-primary)',
                backgroundColor: isActive ? 'var(--accent-light)' : 'transparent',
                textAlign: 'left',
                transition: 'all 0.15s ease',
                opacity: item.disabled ? 0.45 : 1,
              }}
            >
              <Icon size={18} color={isActive ? 'var(--accent)' : 'currentColor'} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Health & Status Indicators */}
      <div
        style={{
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
          {dashboardStats ? `${dashboardStats.total_reviews.toLocaleString()} reviews indexed` : 'Dataset metrics unavailable'}
        </div>
        {dashboardStats && (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Complaint signals</span>
              <span style={{ fontWeight: 600 }}>{dashboardStats.total_complaints.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Average rating</span>
              <span style={{ fontWeight: 600 }}>{dashboardStats.average_rating?.toFixed(1) || 'N/A'} / 5</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Top signal</span>
              <span style={{ fontWeight: 600, textAlign: 'right', maxWidth: '120px' }}>{dashboardStats.top_complaint_category || 'N/A'}</span>
            </div>
          </>
        )}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Gateway API:</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: backendHealthy ? 'var(--status-success)' : 'var(--status-danger)',
              }}
            />
            <span style={{ fontWeight: 500, color: backendHealthy ? 'var(--status-success)' : 'var(--status-danger)' }}>
              {backendHealthy ? 'Online (8000)' : 'Offline'}
            </span>
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>A2A Agents:</span>
          <span style={{ fontWeight: 500, color: availableCount > 0 ? 'var(--status-success)' : 'var(--status-warning)' }}>
            {availableCount}/{configuredCount} Active
          </span>
        </div>
      </div>
    </aside>
  );
};
