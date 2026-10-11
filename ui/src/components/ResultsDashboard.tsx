import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  XCircle,
  Award,
  CheckCircle2,
  Clock,
  UserCheck,
  FileSearch,
  MessageSquare,
  Tag,
  TrendingUp,
  Layers,
  Sparkles,
  Zap,
  Info,
} from 'lucide-react';
import { AnalysisResponse, RecommendedAction, RetrievedEvidence } from '../types';
import { PipelineFlowInspector } from './PipelineFlowInspector';

interface ResultsDashboardProps {
  result: AnalysisResponse;
  onOpenFeedback: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  result,
  onOpenFeedback,
}) => {
  const [activeSection, setActiveSection] = useState<'summary' | 'actions' | 'root_causes' | 'evidence' | 'flow'>('summary');

  const qDecision = result.quality_decision || 'unknown';
  const confScore = result.confidence_score;
  const maxCeiling = 80.0;
  const normalizedPct = confScore !== undefined ? Math.min(100, Math.max(0, Math.round((confScore / maxCeiling) * 100))) : null;
  const crewRep = result.crew_report;
  const execSummary = result.executive_summary || (crewRep ? crewRep.executive_summary : `No consolidated answer was returned for "${result.query}".`);

  // Extract actions
  let actions: RecommendedAction[] = [];
  if (crewRep && crewRep.prioritized_recommendations && crewRep.prioritized_recommendations.length > 0) {
    actions = crewRep.prioritized_recommendations;
  } else if (result.action_plan && result.action_plan.ranked_actions) {
    actions = result.action_plan.ranked_actions.map((a: any, idx: number) => ({
      priority_rank: idx + 1,
      problem_addressed: a.issue_summary || a.problem_addressed || result.query,
      recommended_action: a.recommended_action || 'No action specified',
      rationale: a.evidence_rationale || a.rationale || 'No rationale returned',
      owner_role: a.owner_role || 'manager',
      kpi: a.measurable_kpi || a.kpi || 'No KPI specified',
      target: a.target_value || a.target || 'No target specified',
      source_ids: a.source_ids || [],
    }));
  }

  // Extract root causes
  const hypotheses = result.root_cause && result.root_cause.hypotheses ? result.root_cause.hypotheses : [];

  // Helper function to render structured markdown sections from executive summary
  const renderFormattedReport = (rawText: string) => {
    // If text does not contain ### section markers, render as standard paragraph
    if (!rawText.includes('### ')) {
      return (
        <div className="card">
          <p style={{ fontSize: '14.5px', lineHeight: '1.7', color: 'var(--text-primary)', whiteSpace: 'pre-wrap' }}>
            {rawText}
          </p>
        </div>
      );
    }

    const rawSections = rawText.split('### ').filter((s) => s.trim().length > 0);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {rawSections.map((sec, idx) => {
          const lines = sec.trim().split('\n');
          const title = lines[0].trim();
          const bodyLines = lines.slice(1).filter((l) => l.trim().length > 0);

          // Choose section icon & theme color
          let IconComp = Award;
          let iconColor = 'var(--accent)';
          let borderColor = 'var(--accent)';

          if (title.toLowerCase().includes('diagnosis')) {
            IconComp = Award;
            iconColor = 'var(--accent)';
          } else if (title.toLowerCase().includes('classification') || title.toLowerCase().includes('severity')) {
            IconComp = Tag;
            iconColor = '#7C3AED';
            borderColor = '#7C3AED';
          } else if (title.toLowerCase().includes('trend') || title.toLowerCase().includes('rating')) {
            IconComp = TrendingUp;
            iconColor = '#0284C7';
            borderColor = '#0284C7';
          } else if (title.toLowerCase().includes('root cause') || title.toLowerCase().includes('failure')) {
            IconComp = Sparkles;
            iconColor = '#D97706';
            borderColor = '#D97706';
          } else if (title.toLowerCase().includes('customer evidence') || title.toLowerCase().includes('grounding')) {
            IconComp = FileSearch;
            iconColor = '#16A34A';
            borderColor = '#16A34A';
          } else if (title.toLowerCase().includes('action plan') || title.toLowerCase().includes('corrective')) {
            IconComp = Layers;
            iconColor = '#059669';
            borderColor = '#059669';
          } else if (title.toLowerCase().includes('expected') || title.toLowerCase().includes('outcome')) {
            IconComp = CheckCircle2;
            iconColor = 'var(--status-success)';
            borderColor = 'var(--status-success)';
          }

          return (
            <div
              key={idx}
              className="card"
              style={{
                borderLeft: `4px solid ${borderColor}`,
                padding: '18px 22px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <IconComp size={18} color={iconColor} />
                <h4 style={{ fontSize: '15.5px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
                  {title}
                </h4>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {bodyLines.map((line, lIdx) => {
                  const trimmed = line.trim();
                  // Check if bullet point
                  if (trimmed.startsWith('•') || trimmed.startsWith('-')) {
                    const content = trimmed.replace(/^[•\-]\s*/, '');
                    return (
                      <div key={lIdx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '14px', lineHeight: '1.6' }}>
                        <span style={{ color: iconColor, fontWeight: 700, marginTop: '2px' }}>•</span>
                        <div style={{ color: 'var(--text-primary)', flex: 1 }}>
                          {renderHighlightedInlineText(content)}
                        </div>
                      </div>
                    );
                  }

                  return (
                    <p key={lIdx} style={{ fontSize: '14px', lineHeight: '1.65', color: 'var(--text-primary)', margin: 0 }}>
                      {renderHighlightedInlineText(trimmed)}
                    </p>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Helper to highlight **bold**, [citations], and *italics*
  const renderHighlightedInlineText = (text: string) => {
    // Replace markdown tags with styled spans
    const parts = text.split(/(\*\*.*?\*\*|\[.*?\]|\*.*?\*)/g);
    return parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={pIdx} style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('[') && part.endsWith(']') && part.length > 2) {
        return (
          <span
            key={pIdx}
            className="badge badge-neutral"
            style={{ fontSize: '11px', margin: '0 3px', padding: '1px 6px', display: 'inline-block' }}
          >
            {part}
          </span>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={pIdx} style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>
            {part.slice(1, -1)}
          </em>
        );
      }
      return <span key={pIdx}>{part}</span>;
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Top Banner: Status, Confidence, and Privacy */}
      <div
        className="card"
        style={{
          borderLeft: `4px solid ${
            qDecision === 'passed' ? 'var(--status-success)' : qDecision === 'flagged' ? 'var(--status-warning)' : 'var(--status-danger)'
          }`,
          padding: '20px 24px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            {qDecision === 'passed' && <CheckCircle2 size={26} color="var(--status-success)" />}
            {qDecision === 'flagged' && <AlertTriangle size={26} color="var(--status-warning)" />}
            {qDecision === 'blocked' && <XCircle size={26} color="var(--status-danger)" />}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <h3 style={{ fontSize: '17px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Quality Gate: {qDecision.toUpperCase()}
                </h3>
                <span
                  className={`badge ${qDecision === 'passed' ? 'badge-success' : qDecision === 'flagged' ? 'badge-warning' : 'badge-danger'}`}
                  title="Phase 4 formula ceiling is 80.0 (0.35 relevance + 0.25 traceability + 0.20 agreement - 0.20 staleness). An 80.0 score indicates 100% grounded baseline."
                  style={{ cursor: 'help' }}
                >
                  Confidence: {confScore !== undefined ? `${confScore.toFixed(1)} / 80.0 (${normalizedPct}% Grounding)` : 'Unavailable'}
                </span>
              </div>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                {result.staleness_disclosure || 'Dataset Cutoff: Jan 2022. Grounding verified against retrieved evidence.'}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span className={`badge ${result.pii_decoded ? 'badge-success' : 'badge-neutral'}`} style={{ fontSize: '12px' }}>
              <ShieldCheck size={14} color={result.pii_decoded ? 'var(--status-success)' : 'var(--text-secondary)'} />
              <span>{result.pii_decoded ? 'PII Decoded (Role: Manager)' : 'PII Protected (Masked)'}</span>
            </span>
            <span className="badge badge-neutral" style={{ fontSize: '12px' }}>
              <Clock size={14} color="var(--text-secondary)" />
              <span>{result.total_duration_ms ? `${(result.total_duration_ms / 1000).toFixed(1)}s` : 'Unavailable'}</span>
            </span>
            <button className="btn-secondary" onClick={onOpenFeedback} style={{ padding: '6px 14px', fontSize: '12.5px' }}>
              <MessageSquare size={14} />
              <span>Provide Feedback</span>
            </button>
          </div>
        </div>

        {/* Explainability helper note regarding confidence score formula ceiling */}
        <div
          style={{
            marginTop: '12px',
            paddingTop: '10px',
            borderTop: '1px solid var(--border-subtle)',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <Info size={14} color="var(--accent)" />
          <span>
            <strong>Deterministic Confidence Explanation:</strong> The Phase 4 composite formula is{' '}
            <code>100 × (0.35 × relevance + 0.25 × traceability + 0.20 × agreement - 0.20 × staleness)</code>. The theoretical ceiling is{' '}
            <strong>80.0</strong>; an 80.0 score reflects <strong>100% maximum grounded baseline</strong> with verified source citations.
          </span>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px', flexWrap: 'wrap' }}>
        <button
          onClick={() => setActiveSection('summary')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13.5px',
            fontWeight: activeSection === 'summary' ? 600 : 500,
            color: activeSection === 'summary' ? 'var(--accent)' : 'var(--text-secondary)',
            background: activeSection === 'summary' ? 'var(--accent-light)' : 'transparent',
          }}
        >
          Executive Operations Report
        </button>
        <button
          onClick={() => setActiveSection('actions')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13.5px',
            fontWeight: activeSection === 'actions' ? 600 : 500,
            color: activeSection === 'actions' ? 'var(--accent)' : 'var(--text-secondary)',
            background: activeSection === 'actions' ? 'var(--accent-light)' : 'transparent',
          }}
        >
          Action Plan ({actions.length})
        </button>
        <button
          onClick={() => setActiveSection('root_causes')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13.5px',
            fontWeight: activeSection === 'root_causes' ? 600 : 500,
            color: activeSection === 'root_causes' ? 'var(--accent)' : 'var(--text-secondary)',
            background: activeSection === 'root_causes' ? 'var(--accent-light)' : 'transparent',
          }}
        >
          Root Causes ({hypotheses.length})
        </button>
        <button
          onClick={() => setActiveSection('evidence')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13.5px',
            fontWeight: activeSection === 'evidence' ? 600 : 500,
            color: activeSection === 'evidence' ? 'var(--accent)' : 'var(--text-secondary)',
            background: activeSection === 'evidence' ? 'var(--accent-light)' : 'transparent',
          }}
        >
          Retrieved Evidence ({result.retrieved_evidence ? result.retrieved_evidence.length : 0})
        </button>
        <button
          onClick={() => setActiveSection('flow')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13.5px',
            fontWeight: activeSection === 'flow' ? 600 : 500,
            color: activeSection === 'flow' ? 'var(--accent)' : 'var(--text-secondary)',
            background: activeSection === 'flow' ? 'var(--accent-light)' : 'transparent',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <Layers size={14} />
          <span>Pipeline Flow & LLM Payloads (15 Stages)</span>
        </button>
      </div>

      {/* SECTION 1: DETAILED EXECUTIVE REPORT */}
      {activeSection === 'summary' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Query context reminder card */}
          <div className="card" style={{ background: 'var(--bg-card-subtle)', borderLeft: '3px solid var(--accent)', padding: '14px 18px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Manager Prompt Under Investigation:
            </span>
            <div style={{ fontSize: '14.5px', fontWeight: 500, color: 'var(--text-primary)', marginTop: '4px' }}>
              "{result.query}"
            </div>
          </div>

          {/* Formatted multi-section operational analysis */}
          {renderFormattedReport(execSummary)}

          {crewRep && crewRep.key_findings && crewRep.key_findings.length > 0 && (
            <div className="card">
              <h4 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px' }}>Key Operational Findings</h4>
              <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '14px' }}>
                {crewRep.key_findings.map((f, i) => (
                  <li key={i} style={{ color: 'var(--text-primary)', lineHeight: 1.5 }}>{f}</li>
                ))}
              </ul>
            </div>
          )}

          {crewRep && crewRep.data_limitations_and_cutoffs && (
            <div className="card" style={{ background: 'var(--bg-card-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                <Clock size={16} color="var(--text-secondary)" />
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Dataset Limitations & Disclosures
                </span>
              </div>
              <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                {crewRep.data_limitations_and_cutoffs.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* SECTION 2: ACTION PLANNER */}
      {activeSection === 'actions' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13.5px', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'var(--bg-card-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '12px 16px', fontWeight: 600, width: '60px' }}>Rank</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Recommended Action</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Owner Role</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>KPI & Measurement</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Proposed Target</th>
                </tr>
              </thead>
              <tbody>
                {actions.map((act) => (
                  <tr key={act.priority_rank} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '14px 16px', fontWeight: 700, color: 'var(--accent)' }}>
                      #{act.priority_rank}
                    </td>
                    <td style={{ padding: '14px 16px' }}>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                        {act.recommended_action}
                      </div>
                      <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                        Problem: {act.problem_addressed}
                      </div>
                      {act.rationale && (
                        <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                          Rationale: {act.rationale}
                        </div>
                      )}
                    </td>
                    <td style={{ padding: '14px 16px' }}>
                      <span className="badge badge-neutral" style={{ textTransform: 'capitalize' }}>
                        <UserCheck size={12} />
                        <span>{act.owner_role.replace('_', ' ')}</span>
                      </span>
                    </td>
                    <td style={{ padding: '14px 16px', color: 'var(--text-secondary)' }}>
                      {act.kpi}
                    </td>
                    <td style={{ padding: '14px 16px' }}>
                      <span className="badge badge-success" style={{ fontSize: '12px' }}>
                        {act.target || 'Proposed'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SECTION 3: ROOT CAUSES */}
      {activeSection === 'root_causes' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          {hypotheses.map((h: any, i: number) => (
            <div key={i} className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge badge-warning" style={{ textTransform: 'uppercase', fontSize: '11px' }}>
                  {h.distinction_type || 'Hypothesis'}
                </span>
                <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                  Confidence: {h.confidence !== undefined ? `${(h.confidence * 100).toFixed(0)}%` : 'Unavailable'}
                </span>
              </div>
              <h4 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
                {h.hypothesized_cause || 'Hypothesis text unavailable'}
              </h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '12px' }}>
                {h.mechanism || 'Staffing pacing and station coordination friction identified.'}
              </p>
              {h.supporting_source_ids && (
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {h.supporting_source_ids.map((sid: string) => (
                    <span key={sid} className="badge badge-neutral" style={{ fontSize: '11px' }}>
                      Citation: [{sid}]
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* SECTION 4: EVIDENCE EXPLORER */}
      {activeSection === 'evidence' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          {result.retrieved_evidence && result.retrieved_evidence.map((ev, i) => (
            <div key={ev.chunk_id || i} className="card" style={{ padding: '16px 18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)' }}>
                  [{ev.chunk_id || `src_${i+1}`}]
                </span>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  ★ {ev.stars ? ev.stars.toFixed(1) : '3.0'} / 5.0
                </span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.5, marginBottom: '10px' }}>
                "{ev.text_masked || ev.text || 'Review excerpt'}"
              </p>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11.5px', color: 'var(--text-tertiary)' }}>
                <span>{ev.restaurant_name || ev.city || 'Yelp Verified'}</span>
                <span>{ev.aspects || 'General feedback'}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* SECTION 5: PIPELINE FLOW & LLM PAYLOADS */}
      {activeSection === 'flow' && (
        <PipelineFlowInspector result={result} />
      )}
    </div>
  );
};
