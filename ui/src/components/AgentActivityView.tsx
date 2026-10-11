import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Server,
  Zap,
  Globe,
  Tag,
  ShieldCheck,
} from 'lucide-react';
import { AgentStatusResponse } from '../types';
import { ApiClient } from '../api/client';

interface AgentActivityViewProps {
  statusData: AgentStatusResponse | null;
  onRefresh: () => void;
  isLoading: boolean;
}

export const AgentActivityView: React.FC<AgentActivityViewProps> = ({
  statusData,
  onRefresh,
  isLoading,
}) => {
  const agents = statusData ? statusData.agents : {};
  const totalAvailable = statusData ? statusData.total_available : 0;
  const totalConfigured = statusData ? statusData.total_configured : 4;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header & Refresh */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
              A2A Specialist Agent Mesh & Microservices
            </h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '3px' }}>
            Live status probes, port mappings, and Google ADK Agent Cards across dedicated endpoints.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
            Last checked: {statusData?.last_checked || 'Just now'}
          </span>
          <button
            className="btn-secondary"
            onClick={onRefresh}
            disabled={isLoading}
            style={{ padding: '7px 14px', fontSize: '13px' }}
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
            <span>Refresh Probes</span>
          </button>
        </div>
      </div>

      {/* Summary KPI Banner */}
      <div className="card" style={{ padding: '16px 20px', background: 'var(--bg-card-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Server size={22} color="var(--accent)" />
            <div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Mesh Status: {totalAvailable === totalConfigured ? 'All Services Operational' : 'Partial / Standby Availability'}
              </div>
              <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                {totalAvailable} of {totalConfigured} specialist microservices responding to A2A connectivity probes.
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <span className="badge badge-success" style={{ fontSize: '12px' }}>
              LangGraph Gateway: Port 8000
            </span>
            <span className="badge badge-neutral" style={{ fontSize: '12px' }}>
              Protocol: Google A2A (JSON-RPC 2.0)
            </span>
          </div>
        </div>
      </div>

      {/* Specialist Agent Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '18px' }}>
        {Object.entries(agents).map(([key, info]) => {
          const isOnline = info.status === 'available';
          return (
            <div
              key={key}
              className="card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                padding: '20px 22px',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span className={`badge ${isOnline ? 'badge-success' : 'badge-danger'}`}>
                    {isOnline ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                    <span>{isOnline ? 'AVAILABLE' : 'STANDBY'}</span>
                  </span>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Port {info.port}
                  </span>
                </div>

                <h4 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
                  {info.name}
                </h4>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '16px' }}>
                  {info.description}
                </p>
              </div>

              <div
                style={{
                  borderTop: '1px solid var(--border-subtle)',
                  paddingTop: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Service Endpoint:</span>
                  <code>{info.url}</code>
                </div>
                {info.latency_ms !== undefined && info.latency_ms !== null && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Probe Latency:</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{info.latency_ms} ms</strong>
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Agent Card:</span>
                  <span style={{ color: 'var(--accent)', fontWeight: 500 }}>
                    /.well-known/agent-card.json
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
