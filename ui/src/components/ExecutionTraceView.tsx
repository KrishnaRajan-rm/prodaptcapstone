import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  GitCommit,
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { ExecutionTraceEvent } from '../types';

interface ExecutionTraceViewProps {
  events: ExecutionTraceEvent[];
  requestId?: string;
  totalDurationMs?: number;
}

export const ExecutionTraceView: React.FC<ExecutionTraceViewProps> = ({
  events,
  requestId,
  totalDurationMs,
}) => {
  const [expandedEvents, setExpandedEvents] = useState<Record<string, boolean>>({});

  const toggleExpand = (id: string) => {
    setExpandedEvents((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const expandAll = () => {
    const all: Record<string, boolean> = {};
    events.forEach((e) => {
      all[e.event_id] = true;
    });
    setExpandedEvents(all);
  };

  const collapseAll = () => {
    setExpandedEvents({});
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <GitCommit size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Execution Trace & Telemetry Timeline
            </h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '3px' }}>
            Request ID: <code>{requestId || 'N/A'}</code> • Total Pipeline Latency:{' '}
            <strong>{totalDurationMs ? `${(totalDurationMs / 1000).toFixed(2)}s` : '0.5s'}</strong>
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn-secondary" onClick={expandAll} style={{ padding: '6px 12px', fontSize: '12.5px' }}>
            Expand All
          </button>
          <button className="btn-secondary" onClick={collapseAll} style={{ padding: '6px 12px', fontSize: '12.5px' }}>
            Collapse All
          </button>
        </div>
      </div>

      {/* Timeline Stream */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', position: 'relative' }}>
        {events.map((event, idx) => {
          const isExpanded = expandedEvents[event.event_id] ?? false;
          return (
            <div
              key={event.event_id || idx}
              className="card card-clickable"
              style={{
                padding: '16px 20px',
                borderLeft: `4px solid ${
                  event.status === 'completed'
                    ? 'var(--status-success)'
                    : event.status === 'blocked'
                    ? 'var(--status-danger)'
                    : event.status === 'skipped'
                    ? 'var(--border-strong)'
                    : 'var(--status-warning)'
                }`,
              }}
              onClick={() => toggleExpand(event.event_id)}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {event.status === 'completed' && <CheckCircle2 size={18} color="var(--status-success)" />}
                  {event.status === 'blocked' && <XCircle size={18} color="var(--status-danger)" />}
                  {event.status === 'flagged' && <AlertTriangle size={18} color="var(--status-warning)" />}
                  {event.status === 'skipped' && <Clock size={18} color="var(--text-tertiary)" />}

                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '14.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {event.component}
                      </span>
                      <span className="badge badge-neutral" style={{ fontSize: '11px', textTransform: 'capitalize' }}>
                        {event.stage.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                      {event.output_summary}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                  {event.duration_ms !== undefined && (
                    <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)', fontWeight: 500 }}>
                      {event.duration_ms.toFixed(1)} ms
                    </span>
                  )}
                  {isExpanded ? <ChevronDown size={18} color="var(--text-secondary)" /> : <ChevronRight size={18} color="var(--text-secondary)" />}
                </div>
              </div>

              {/* Expandable Details Panel */}
              {isExpanded && (
                <div
                  style={{
                    marginTop: '14px',
                    paddingTop: '14px',
                    borderTop: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                    fontSize: '13px',
                  }}
                  onClick={(e) => e.stopPropagation()}
                >
                  {event.input_summary && (
                    <div>
                      <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Input Context: </span>
                      <span style={{ color: 'var(--text-primary)' }}>{event.input_summary}</span>
                    </div>
                  )}

                  {/* LLM Input Payload */}
                  {event.input_payload && (
                    <div style={{ marginTop: '6px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--accent)', fontSize: '12px' }}>
                        Input Payload to LLM / Node:
                      </span>
                      <pre
                        style={{
                          background: 'var(--bg-card-subtle)',
                          padding: '10px 12px',
                          borderRadius: '6px',
                          fontSize: '11.5px',
                          maxHeight: '180px',
                          overflowY: 'auto',
                          marginTop: '4px',
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        {JSON.stringify(event.input_payload, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Stage Output Payload */}
                  {event.output_payload && (
                    <div style={{ marginTop: '6px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--status-success)', fontSize: '12px' }}>
                        Output Payload Produced by Stage:
                      </span>
                      <pre
                        style={{
                          background: 'var(--bg-card-subtle)',
                          padding: '10px 12px',
                          borderRadius: '6px',
                          fontSize: '11.5px',
                          maxHeight: '180px',
                          overflowY: 'auto',
                          marginTop: '4px',
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        {JSON.stringify(event.output_payload, null, 2)}
                      </pre>
                    </div>
                  )}

                  {event.source_ids && event.source_ids.length > 0 && (
                    <div>
                      <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Referenced Sources: </span>
                      <span style={{ color: 'var(--accent)' }}>{event.source_ids.join(', ')}</span>
                    </div>
                  )}

                  {event.warnings && event.warnings.length > 0 && (
                    <div style={{ background: 'var(--status-warning-bg)', padding: '8px 12px', borderRadius: '6px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--status-warning)' }}>Notice: </span>
                      <span style={{ color: 'var(--text-primary)' }}>{event.warnings.join(' | ')}</span>
                    </div>
                  )}

                  {event.error_summary && (
                    <div style={{ background: 'var(--status-danger-bg)', padding: '8px 12px', borderRadius: '6px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--status-danger)' }}>Error: </span>
                      <span style={{ color: 'var(--text-primary)' }}>{event.error_summary}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
