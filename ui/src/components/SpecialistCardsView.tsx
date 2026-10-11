import React from 'react';
import {
  Tag,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  ListOrdered,
  UserCheck,
  Target,
  Clock,
  Sparkles,
  BarChart3,
  Flame,
} from 'lucide-react';
import { AnalysisResponse } from '../types';

interface SpecialistCardsViewProps {
  type: 'categories' | 'trends' | 'root_cause' | 'actions';
  result: AnalysisResponse;
}

export const SpecialistCardsView: React.FC<SpecialistCardsViewProps> = ({ type, result }) => {
  // 1. Complaint Categories & Aspects View
  if (type === 'categories') {
    const cls = result.classification || {};
    const categories: string[] = cls.categories || cls.complaint_categories || [];
    const aspects: string[] = cls.aspects || cls.aspect_tags || [];
    const severity: string | undefined = cls.severity || cls.severity_level;
    const sentiment: string | undefined = cls.sentiment;
    const snippets: string[] = cls.evidence_snippets || [];

    const getSeverityBadgeClass = (sev: string) => {
      switch (sev.toLowerCase()) {
        case 'critical':
        case 'high':
          return 'badge-danger';
        case 'medium':
          return 'badge-warning';
        default:
          return 'badge-success';
      }
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Classification Specialist Findings (Port 8001)
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Aspect-based sentiment, severity assessment, and grounded evidence extractions.
          </p>
        </div>

        {/* Overview Metric Badges */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Assessed Severity</span>
            <div style={{ marginTop: '8px' }}>
              <span className={`badge ${getSeverityBadgeClass(severity)}`} style={{ fontSize: '13px', textTransform: 'uppercase' }}>
                <Flame size={14} />
                <span>{severity}</span>
              </span>
            </div>
          </div>

          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Customer Sentiment</span>
            <div style={{ marginTop: '8px', fontSize: '16px', fontWeight: 600, textTransform: 'capitalize', color: sentiment === 'negative' ? 'var(--status-danger)' : 'var(--text-primary)' }}>
              {sentiment || 'Unavailable'}
            </div>
          </div>

          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Primary Aspect</span>
            <div style={{ marginTop: '8px', fontSize: '15px', fontWeight: 600, color: 'var(--accent)' }}>
              {aspects[0] || 'Unavailable'}
            </div>
          </div>
        </div>

        {/* Categories & Aspects Detail Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          <div className="card">
            <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Tag size={16} color="var(--accent)" />
              <span>Complaint Categories ({categories.length})</span>
            </h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {categories.length > 0 ? categories.map((c, i) => (
                <span key={i} className="badge badge-neutral" style={{ fontSize: '13px', padding: '6px 12px' }}>
                  {c}
                </span>
              )) : <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>No classification returned.</span>}
            </div>
          </div>

          <div className="card">
            <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={16} color="var(--accent)" />
              <span>Extracted Aspects ({aspects.length})</span>
            </h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {aspects.length > 0 ? aspects.map((a, i) => (
                <span key={i} className="badge badge-neutral" style={{ fontSize: '13px', padding: '6px 12px' }}>
                  {a}
                </span>
              )) : <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>No aspect tags returned.</span>}
            </div>
          </div>
        </div>

        {/* Evidence Snippets */}
        {snippets.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px' }}>Grounded Review Snippets</h3>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13.5px' }}>
              {snippets.map((snip, i) => (
                <li key={i} style={{ color: 'var(--text-primary)', fontStyle: 'italic' }}>
                  "{snip}"
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  // 2. Historical Trends & Forecasts View
  if (type === 'trends') {
    const tr = result.trends || {};
    const totalReviews = tr.total_reviews_analyzed;
    const avgRating = tr.average_rating;
    const period = tr.analysis_period;
    const improving: string[] = tr.improving_aspects || [];
    const worsening: string[] = tr.worsening_aspects || [];
    const forecast = tr.forecast || {};
    const forecastVals = forecast.forecast_values || {};

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Trend Analysis Specialist (Port 8002)
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Historical aggregations computed from DuckDB reviews with statistical forecast horizon.
          </p>
        </div>

        {/* Metric Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Corpus Reviews Analyzed</span>
            <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--accent)', marginTop: '4px' }}>
              {totalReviews !== undefined ? totalReviews.toLocaleString() : 'Unavailable'}
            </div>
          </div>

          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Historical Mean Rating</span>
            <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>
              {avgRating !== undefined ? `★ ${avgRating.toFixed(1)} / 5.0` : 'Unavailable'}
            </div>
          </div>

          <div className="card" style={{ padding: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>Dataset Timeline Window</span>
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: '8px' }}>
              {period || 'Unavailable'}
            </div>
          </div>
        </div>

        {/* Worsening vs Improving */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          <div className="card" style={{ borderLeft: '4px solid var(--status-danger)' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--status-danger)', marginBottom: '10px' }}>
              Worsening Operational Aspects
            </h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {worsening.length > 0 ? worsening.map((w, i) => (
                <span key={i} className="badge badge-danger" style={{ fontSize: '12.5px' }}>
                  {w}
                </span>
              )) : <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>No worsening aspects returned.</span>}
            </div>
          </div>

          <div className="card" style={{ borderLeft: '4px solid var(--status-success)' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--status-success)', marginBottom: '10px' }}>
              Stable / Improving Aspects
            </h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {improving.length > 0 ? (
                improving.map((imp, i) => (
                  <span key={i} className="badge badge-success" style={{ fontSize: '12.5px' }}>
                    {imp}
                  </span>
                ))
              ) : (
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>No improving trends observed in this window</span>
              )}
            </div>
          </div>
        </div>

        {/* Statistical Projections */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart3 size={16} color="var(--accent)" />
              <span>Statistical Rating Projections (Historical Forecast Horizon)</span>
            </h3>
            <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
              Labeled: Projection (Not Current Fact)
            </span>
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'var(--bg-card-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '8px 12px' }}>Projected Horizon</th>
                <th style={{ padding: '8px 12px' }}>Forecasted Average Rating</th>
                <th style={{ padding: '8px 12px' }}>Trajectory Confidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(forecastVals).length > 0 ? Object.entries(forecastVals).map(([m, val]: [string, any]) => (
                <tr key={m} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 600 }}>{m}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--status-warning)' }}>
                    ★ {typeof val === 'number' ? val.toFixed(2) : val}
                  </td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{forecast.model_type || 'Deterministic forecast'}</td>
                </tr>
              )) : <tr><td colSpan={3} style={{ padding: '12px', color: 'var(--text-secondary)' }}>No forecast was returned.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // 3. Root Cause Hypotheses View
  if (type === 'root_cause') {
    const rc = result.root_cause || {};
    const hypotheses = rc.hypotheses || [];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Root-Cause Specialist Hypotheses (Port 8003)
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Causal mechanisms synthesized from review evidence chunks with source provenance.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {hypotheses.length > 0 ? hypotheses.map((h: any, idx: number) => (
            <div key={idx} className="card" style={{ padding: '18px 20px', borderLeft: '4px solid var(--accent)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent)' }}>
                  Hypothesis #{idx + 1}: {h.hypothesis_id || `H${idx + 1}`}
                </span>
                <span className="badge badge-neutral" style={{ fontSize: '11.5px' }}>
                  Confidence: {h.confidence !== undefined ? `${(h.confidence * 100).toFixed(0)}%` : 'Unavailable'}
                </span>
              </div>

              <h4 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
                {h.hypothesized_cause || 'Unnamed hypothesis'}
              </h4>

              <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '14px' }}>
                {h.mechanism || 'Mechanism not provided.'}
              </p>

              {h.supporting_evidence && h.supporting_evidence.length > 0 && (
                <div style={{ background: 'var(--bg-card-subtle)', padding: '12px', borderRadius: '6px', marginBottom: '10px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Supporting Customer Excerpts:
                  </span>
                  <ul style={{ paddingLeft: '18px', marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '13px' }}>
                    {h.supporting_evidence.map((ev: string, i: number) => (
                      <li key={i} style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>
                        "{ev}"
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {h.supporting_source_ids && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Grounding Citations:</span>
                  {h.supporting_source_ids.map((sid: string) => (
                    <span key={sid} className="badge badge-neutral" style={{ fontSize: '11px' }}>
                      [{sid}]
                    </span>
                  ))}
                </div>
              )}
            </div>
          )) : <div className="card"><span style={{ color: 'var(--text-secondary)' }}>No root-cause hypotheses were returned for this analysis.</span></div>}
        </div>
      </div>
    );
  }

  // 4. Action Planner Recommendations View
  if (type === 'actions') {
    const ap = result.action_plan || {};
    const rankedActions = ap.ranked_actions || [];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Action Planner Specialist Recommendations (Port 8004)
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Prioritized corrective operational plans scored via documented formula (Frequency, Severity, Trend, Rating Impact).
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {rankedActions.length > 0 ? rankedActions.map((act: any, idx: number) => {
            const scoreDetails = act.scoring_details || {};
            return (
              <div key={idx} className="card" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent)' }}>
                      #{act.rank || idx + 1}
                    </span>
                    <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {act.recommended_action || 'Unnamed recommendation'}
                    </h3>
                  </div>

                  <span className="badge badge-success" style={{ fontSize: '13px' }}>
                    Priority Score: {act.priority_score !== undefined ? act.priority_score.toFixed(1) : 'Unavailable'} / 100
                  </span>
                </div>

                <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '14px' }}>
                  <strong>Problem Solved:</strong> {act.issue_summary || act.problem_addressed || 'Service throughput and guest experience issue'}
                </p>

                {act.evidence_rationale && (
                  <p style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginBottom: '14px' }}>
                    <strong>Evidence Rationale:</strong> {act.evidence_rationale}
                  </p>
                )}

                {/* KPI and Target Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px', background: 'var(--bg-card-subtle)', padding: '14px', borderRadius: '8px' }}>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Suggested Owner Role</span>
                    <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '3px', textTransform: 'capitalize' }}>
                      <UserCheck size={14} style={{ display: 'inline', marginRight: '4px' }} />
                      {act.owner_role || 'General Manager'}
                    </div>
                  </div>

                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Measurable KPI</span>
                    <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '3px' }}>
                      <Target size={14} style={{ display: 'inline', marginRight: '4px' }} />
                      {act.measurable_kpi || 'Unavailable'}
                    </div>
                  </div>

                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Proposed Target</span>
                    <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--status-success)', marginTop: '3px' }}>
                      {act.target_value || 'Proposed 20% Reduction'}
                    </div>
                  </div>

                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Measurement Horizon</span>
                    <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '3px' }}>
                      <Clock size={14} style={{ display: 'inline', marginRight: '4px' }} />
                      {act.measurement_period || '30 days'}
                    </div>
                  </div>
                </div>
              </div>
            );
          }) : <div className="card"><span style={{ color: 'var(--text-secondary)' }}>No recommendations were returned for this analysis.</span></div>}
        </div>
      </div>
    );
  }

  return null;
};
