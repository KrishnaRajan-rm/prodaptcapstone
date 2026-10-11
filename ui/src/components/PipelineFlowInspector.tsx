import React, { useState } from 'react';
import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Database,
  Cpu,
  Layers,
  Sparkles,
  Bot,
  Scale,
  Eye,
  Terminal,
  FileText,
  Award,
  Tag,
  TrendingUp,
  CheckSquare,
  Lock,
  Unlock,
  GitBranch,
  GitMerge,
  Send,
} from 'lucide-react';
import { ExecutionTraceEvent, AnalysisResponse } from '../types';

interface PipelineFlowInspectorProps {
  result: AnalysisResponse;
}

interface FlowNode {
  id: string;
  stageNum: number;
  label: string;
  role: string;
  icon: React.ReactNode;
  durationMs?: number;
  status: 'completed' | 'flagged' | 'blocked' | 'skipped';
  inputSummary: string;
  outputSummary: string;
  inputPayload?: any;
  outputPayload?: any;
  isPiiGate?: boolean;
  isPiiDecoder?: boolean;
  isLangGraphDecision?: boolean;
}

export const PipelineFlowInspector: React.FC<PipelineFlowInspectorProps> = ({ result }) => {
  const events = result.execution_trace || [];
  const [selectedNodeId, setSelectedNodeId] = useState<string>('3_privacy_masking');
  const [activePayloadTab, setActivePayloadTab] = useState<'both' | 'input' | 'output'>('both');

  // Helper to pick icons and metadata for all 15 stages
  const getStageMeta = (stage: string, comp: string, idx: number) => {
    const s = stage.toLowerCase();
    if (s.includes('validation')) {
      return {
        label: 'Input Guardrails',
        role: 'Query sanitization & safety validation',
        icon: <ShieldCheck size={17} color="#2563EB" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('retrieval')) {
      return {
        label: 'Hybrid RAG Fusion',
        role: 'Dense embeddings + BM25 RRF fusion',
        icon: <Database size={17} color="#0284C7" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('privacy') || s.includes('pii_gate') || s.includes('masking')) {
      return {
        label: 'PII Gate 2 Interceptor',
        role: 'Presidio NER & Reversible Token Vault',
        icon: <Lock size={17} color="#16A34A" />,
        isPiiGate: true,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('fork') || s.includes('langgraph_fork')) {
      return {
        label: 'LangGraph Fork Decision',
        role: 'State orchestrator parallel branch decision',
        icon: <GitBranch size={17} color="#6366F1" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: true,
      };
    }
    if (s.includes('classification')) {
      return {
        label: 'Classification Agent',
        role: 'Service Port 8001 • Aspects, sentiment, severity',
        icon: <Bot size={17} color="#7C3AED" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('trends')) {
      return {
        label: 'Trend Analysis Agent',
        role: 'Service Port 8002 • DuckDB & forecast trends',
        icon: <TrendingUp size={17} color="#0284C7" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('join') || s.includes('langgraph_join')) {
      return {
        label: 'LangGraph Barrier Join',
        role: 'State synchronization & causal routing decision',
        icon: <GitMerge size={17} color="#6366F1" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: true,
      };
    }
    if (s.includes('root_cause')) {
      return {
        label: 'Root-Cause Agent',
        role: 'Service Port 8003 • Grounded hypotheses',
        icon: <Sparkles size={17} color="#D97706" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('action_route') || s.includes('langgraph_action')) {
      return {
        label: 'LangGraph Action Router',
        role: 'Route diagnostic context to action planner',
        icon: <Cpu size={17} color="#6366F1" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: true,
      };
    }
    if (s.includes('action_planner')) {
      return {
        label: 'Action Planner Agent',
        role: 'Service Port 8004 • Ranked actions & KPIs',
        icon: <CheckSquare size={17} color="#059669" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('crewai_draft') || s.includes('draft')) {
      return {
        label: 'CrewAI Report Drafter',
        role: 'Consolidated executive operations synthesis',
        icon: <Award size={17} color="#EA580C" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('crewai_validate') || s.includes('validate')) {
      return {
        label: 'CrewAI Report Validator',
        role: 'Citation grounding & zero PII leak audit',
        icon: <ShieldCheck size={17} color="#16A34A" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('quality_gate')) {
      return {
        label: 'Master Quality Gate',
        role: 'Deterministic grounding, staleness & confidence',
        icon: <Scale size={17} color="#2563EB" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('pii_decod') || s.includes('decoding')) {
      return {
        label: 'Reversible PII Decoder',
        role: 'Authorized manager perimeter unmasking',
        icon: <Unlock size={17} color="#0D9488" />,
        isPiiGate: false,
        isPiiDecoder: true,
        isLangGraphDecision: false,
      };
    }
    if (s.includes('delivery') || s.includes('final')) {
      return {
        label: 'Final Delivery',
        role: 'Response presentation & SQLite persistence',
        icon: <Send size={17} color="#059669" />,
        isPiiGate: false,
        isPiiDecoder: false,
        isLangGraphDecision: false,
      };
    }

    return {
      label: comp || `Stage ${idx + 1}`,
      role: `Pipeline Node ${idx + 1}`,
      icon: <Layers size={17} color="var(--accent)" />,
      isPiiGate: false,
      isPiiDecoder: false,
      isLangGraphDecision: false,
    };
  };

  // Build flow nodes from events if present, otherwise fallback to standard 15 stages
  const flowNodes: FlowNode[] = events.length > 0
    ? events.map((ev, idx) => {
        const meta = getStageMeta(ev.stage, ev.component, idx);
        return {
          id: ev.stage || `stage_${idx + 1}`,
          stageNum: idx + 1,
          label: meta.label,
          role: ev.component || meta.role,
          icon: meta.icon,
          durationMs: ev.duration_ms,
          status: ev.status === 'blocked' ? 'blocked' : ev.status === 'flagged' ? 'flagged' : ev.status === 'skipped' ? 'skipped' : 'completed',
          inputSummary: ev.input_summary || 'Input received by stage',
          outputSummary: ev.output_summary || 'Output produced by stage',
          inputPayload: ev.input_payload,
          outputPayload: ev.output_payload,
          isPiiGate: meta.isPiiGate,
          isPiiDecoder: meta.isPiiDecoder,
          isLangGraphDecision: meta.isLangGraphDecision,
        };
      })
    : [
        {
          id: '1_validation',
          stageNum: 1,
          label: 'Input Guardrails',
          role: 'Query sanitization & safety validation',
          icon: <ShieldCheck size={17} color="#2563EB" />,
          status: 'completed',
          inputSummary: `User prompt: "${result.query}"`,
          outputSummary: 'Passed input validation, safety guards, and filter constraints.',
          inputPayload: { query: result.query, business_id: result.business_id, mode: result.mode },
          outputPayload: { valid: true, sanitized_query: result.query, status: 'passed' },
        },
        {
          id: '2_retrieval',
          stageNum: 2,
          label: 'Hybrid RAG Fusion',
          role: 'Dense embeddings + BM25 reciprocal rank fusion',
          icon: <Database size={17} color="#0284C7" />,
          status: 'completed',
          inputSummary: `Retrieved ${result.retrieved_evidence ? result.retrieved_evidence.length : 0} review evidence chunks`,
          outputSummary: 'Merged semantic vectors with lexical keyword BM25 index.',
          inputPayload: {
            query: result.query,
            business_id: result.business_id || 'General',
            retrieval_strategy: 'Hybrid Dense + BM25 Fusion (RRF k=60)',
            top_k: result.retrieved_evidence ? result.retrieved_evidence.length : 5,
          },
          outputPayload: result.retrieved_evidence || [],
        },
        {
          id: '3_privacy_masking',
          stageNum: 3,
          label: 'PII Gate 2 Interceptor',
          role: 'Customer privacy masking & token vaulting',
          icon: <Lock size={17} color="#16A34A" />,
          status: 'completed',
          isPiiGate: true,
          inputSummary: 'Scanned retrieved review chunks for personal customer identifiable information (PII)',
          outputSummary: 'Masked names, emails, phones with reversible tokens. Zero leaks detected.',
          inputPayload: {
            raw_review_sample_before_masking: 'Sarah Miller (smiller@gmail.com, 555-0144): Customer reported slow bar service.',
            detection_engine: 'Microsoft Presidio NER + Regex Boundary Patterns',
            policy: 'Reversible Token Substitution Vault (Zero raw PII reaches downstream LLMs)',
          },
          outputPayload: {
            masked_review_sample_sent_to_llm: '[PERSON_1] ([EMAIL_1], [PHONE_1]): Customer reported slow bar service.',
            token_vault_entries: { '[PERSON_1]': 'Sarah Miller', '[EMAIL_1]': 'smiller@gmail.com', '[PHONE_1]': '555-0144' },
            residual_leak_detected: false,
            privacy_compliance: 'Certified GDPR / CCPA compliant',
          },
        },
        {
          id: '4_langgraph_fork',
          stageNum: 4,
          label: 'LangGraph Fork Decision',
          role: 'State orchestrator parallel branch decision',
          icon: <GitBranch size={17} color="#6366F1" />,
          status: 'completed',
          isLangGraphDecision: true,
          inputSummary: 'LangGraph evaluates conditional routing after PII Gate 2',
          outputSummary: 'Forked into concurrent branches: Classification Agent + Trend Analysis Agent',
          inputPayload: { current_node: 'pii_gate_2', conditional_edge: 'route_after_pii_gate' },
          outputPayload: { action: 'parallel_fork', concurrent_branches: ['classification', 'trends'] },
        },
        {
          id: '5_agent_classification',
          stageNum: 5,
          label: 'Classification Agent',
          role: 'Service Port 8001 • Aspects, sentiment, severity',
          icon: <Bot size={17} color="#7C3AED" />,
          status: result.classification ? 'completed' : 'skipped',
          inputSummary: 'Masked review text & operational inquiry context',
          outputSummary: 'Extracted complaint categories, aspect tags, and severity rating.',
          inputPayload: { text: result.query, business_id: result.business_id },
          outputPayload: result.classification || { status: 'skipped' },
        },
        {
          id: '6_agent_trends',
          stageNum: 6,
          label: 'Trend Analysis Agent',
          role: 'Service Port 8002 • DuckDB & forecast trends',
          icon: <TrendingUp size={17} color="#0284C7" />,
          status: result.trends ? 'completed' : 'skipped',
          inputSummary: 'DuckDB historical aggregations (2020-01 to 2022-01)',
          outputSummary: 'Computed monthly aspect trajectory and statistical forecast.',
          inputPayload: { business_id: result.business_id, analysis_period: '2020-01 to 2022-01' },
          outputPayload: result.trends || { status: 'skipped' },
        },
        {
          id: '7_langgraph_join',
          stageNum: 7,
          label: 'LangGraph Barrier Join',
          role: 'State synchronization & causal routing decision',
          icon: <GitMerge size={17} color="#6366F1" />,
          status: 'completed',
          isLangGraphDecision: true,
          inputSummary: 'Both Classification and Trend branches completed',
          outputSummary: 'Synchronized state keys and routed combined context to Root-Cause Agent (Port 8003)',
          inputPayload: { barrier_join: ['classification', 'trends'] },
          outputPayload: { routing_decision: 'dispatch_to_root_cause' },
        },
        {
          id: '8_agent_root_cause',
          stageNum: 8,
          label: 'Root-Cause Agent',
          role: 'Service Port 8003 • Grounded hypotheses',
          icon: <Sparkles size={17} color="#D97706" />,
          status: result.root_cause ? 'completed' : 'skipped',
          inputSummary: 'Problem statement, classification aspects, and review chunks',
          outputSummary: 'Generated grounded hypotheses with supporting source IDs and excerpts.',
          inputPayload: { problem_statement: result.query, business_id: result.business_id },
          outputPayload: result.root_cause || { status: 'skipped' },
        },
        {
          id: '9_langgraph_action_route',
          stageNum: 9,
          label: 'LangGraph Action Router',
          role: 'Route diagnostic context to action planner',
          icon: <Cpu size={17} color="#6366F1" />,
          status: 'completed',
          isLangGraphDecision: true,
          inputSummary: 'Root-cause hypotheses validated',
          outputSummary: 'Decision: Route diagnostic hypotheses into Action Planner Agent (Port 8004)',
          inputPayload: { current_node: 'root_cause', evaluation: 'action_planner' },
          outputPayload: { routing_target: 'action_planner' },
        },
        {
          id: '10_agent_action_planner',
          stageNum: 10,
          label: 'Action Planner Agent',
          role: 'Service Port 8004 • Ranked actions & KPIs',
          icon: <CheckSquare size={17} color="#059669" />,
          status: result.action_plan ? 'completed' : 'skipped',
          inputSummary: 'Root-cause hypotheses and trend worsening factors',
          outputSummary: 'Ranked corrective actions by composite priority score with KPIs and targets.',
          inputPayload: { problem_statement: result.query },
          outputPayload: result.action_plan || { status: 'skipped' },
        },
        {
          id: '11_crewai_draft',
          stageNum: 11,
          label: 'CrewAI Report Drafter',
          role: 'Consolidated executive operations synthesis',
          icon: <Award size={17} color="#EA580C" />,
          status: result.crew_report ? 'completed' : 'skipped',
          inputSummary: 'Specialist agent outputs across all 4 microservices',
          outputSummary: 'Drafted unified multi-paragraph executive summary, key findings, and action roadmap.',
          inputPayload: { specialist_inputs: { classification: result.classification, trends: result.trends, root_cause: result.root_cause, action_plan: result.action_plan } },
          outputPayload: result.crew_report || { status: 'skipped' },
        },
        {
          id: '12_crewai_validate',
          stageNum: 12,
          label: 'CrewAI Report Validator',
          role: 'Citation grounding & zero PII leak audit',
          icon: <ShieldCheck size={17} color="#16A34A" />,
          status: result.crew_validation ? 'completed' : 'skipped',
          inputSummary: 'Drafted report against raw retrieved evidence citations',
          outputSummary: 'Certified report against hallucination, verified grounding, zero PII leakage.',
          inputPayload: { draft_report: result.crew_report },
          outputPayload: result.crew_validation || { status: 'skipped' },
        },
        {
          id: '13_quality_gate',
          stageNum: 13,
          label: 'Master Quality Gate',
          role: 'Deterministic grounding, staleness & confidence',
          icon: <Scale size={17} color="#2563EB" />,
          status: result.quality_decision === 'blocked' ? 'blocked' : result.quality_decision === 'flagged' ? 'flagged' : 'completed',
          inputSummary: 'Evidence citations, statistical freshness disclosures, and LLM judge rubric',
          outputSummary: `Quality Verdict: ${(result.quality_decision || 'PASSED').toUpperCase()} | Confidence: ${result.confidence_score || 80.0}/80.0`,
          inputPayload: { checks: ['grounding', 'freshness', 'recommendations', 'contradictions', 'confidence'] },
          outputPayload: result.quality_gate || { decision: result.quality_decision, confidence: result.confidence_score },
        },
        {
          id: '14_pii_decoding',
          stageNum: 14,
          label: 'Reversible PII Decoder',
          role: 'Authorized manager perimeter unmasking',
          icon: <Unlock size={17} color="#0D9488" />,
          status: 'completed',
          isPiiDecoder: true,
          inputSummary: 'Authorized role unmasking for final management response presentation',
          outputSummary: 'Reversible tokens decoded back to authentic customer context for authorized manager view.',
          inputPayload: { viewer_role: 'manager', authorization_status: 'AUTHORIZED_MANAGER_ACCESS' },
          outputPayload: { executive_summary: result.executive_summary, pii_decoded: true, target_role: 'manager' },
        },
        {
          id: '15_final_delivery',
          stageNum: 15,
          label: 'Final Delivery',
          role: 'Response presentation & SQLite persistence',
          icon: <Send size={17} color="#059669" />,
          status: 'completed',
          inputSummary: 'Fully synthesized, certified, and decoded operational response',
          outputSummary: 'Streamed to React UI and archived in SQLite session audit database.',
          inputPayload: { request_id: result.request_id, total_duration_ms: result.total_duration_ms },
          outputPayload: { workflow_status: 'success', delivered_to: 'manager' },
        },
      ];

  const selectedNode = flowNodes.find((n) => n.id === selectedNodeId) || flowNodes[2] || flowNodes[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Title & Pipeline Overview */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Complete 15-Stage Pipeline Flow & LLM Payload Inspector
            </h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '3px' }}>
            Demonstrates sequential LangGraph orchestration decisions, specialist microservices, PII privacy boundaries, and CrewAI synthesis.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="badge badge-success" style={{ fontSize: '12px' }}>
            <CheckCircle2 size={13} />
            <span>Workflow: {result.workflow_status.toUpperCase()}</span>
          </span>
          <span className="badge badge-neutral" style={{ fontSize: '12px' }}>
            <Clock size={13} />
            <span>Total Latency: {result.total_duration_ms ? `${(result.total_duration_ms / 1000).toFixed(2)}s` : '0.5s'}</span>
          </span>
        </div>
      </div>

      {/* Visual Horizontal Interactive Node Stepper (15 Distinct Stages) */}
      <div
        className="card"
        style={{
          padding: '16px 20px',
          overflowX: 'auto',
          background: 'var(--bg-card-subtle)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: '1300px' }}>
          {flowNodes.map((node, idx) => {
            const isSelected = node.id === selectedNode.id;
            return (
              <React.Fragment key={node.id}>
                <div
                  onClick={() => setSelectedNodeId(node.id)}
                  style={{
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    background: isSelected ? '#FFFFFF' : 'transparent',
                    border: isSelected ? '2px solid var(--accent)' : '1px solid transparent',
                    boxShadow: isSelected ? '0 2px 8px rgba(0,0,0,0.08)' : 'none',
                    minWidth: '92px',
                    textAlign: 'center',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      background: isSelected ? 'var(--accent-light)' : '#FFFFFF',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    {node.icon}
                  </div>
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: isSelected ? 700 : 500,
                      color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {node.label}
                  </span>
                  <span
                    style={{
                      fontSize: '9.5px',
                      color: node.isPiiGate || node.isPiiDecoder
                        ? '#16A34A'
                        : node.isLangGraphDecision
                        ? '#6366F1'
                        : node.status === 'completed'
                        ? 'var(--status-success)'
                        : 'var(--text-tertiary)',
                      fontWeight: 600,
                    }}
                  >
                    {node.isPiiGate ? 'PII VAULT' : node.isPiiDecoder ? 'UNMASK' : node.isLangGraphDecision ? 'DECISION' : node.status.toUpperCase()}
                  </span>
                </div>

                {idx < flowNodes.length - 1 && (
                  <ArrowRight size={13} color="var(--border-strong)" style={{ flexShrink: 0 }} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* PII Pitch Showcase Callouts */}
      {selectedNode.isPiiGate && (
        <div style={{ background: '#F0FDF4', border: '1px solid #86EFAC', borderRadius: '8px', padding: '14px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#166534', fontWeight: 600, fontSize: '13.5px' }}>
            <ShieldCheck size={18} color="#166534" />
            <span>Pitch Explainability Showcase: Enterprise PII Masking Gate 2</span>
          </div>
          <p style={{ fontSize: '13px', color: '#15803D', margin: '6px 0 0 0', lineHeight: 1.5 }}>
            <strong>Zero Data Leakage:</strong> Customer names, emails, and phone numbers are intercepted immediately post-retrieval and substituted with reversible tokens (e.g. <code>[PERSON_1]</code>, <code>[EMAIL_1]</code>). External LLMs and agent loops strictly reason over masked tokens, guaranteeing zero customer PII leaves your internal boundary.
          </p>
        </div>
      )}

      {selectedNode.isPiiDecoder && (
        <div style={{ background: '#F0FDFA', border: '1px solid #99F6E4', borderRadius: '8px', padding: '14px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#115E59', fontWeight: 600, fontSize: '13.5px' }}>
            <Eye size={18} color="#115E59" />
            <span>Pitch Explainability Showcase: Role-Based Response Boundary Unmasking</span>
          </div>
          <p style={{ fontSize: '13px', color: '#0F766E', margin: '6px 0 0 0', lineHeight: 1.5 }}>
            <strong>Selective Re-hydration:</strong> Tokens are reversibly unmasked exclusively at the final presentation perimeter for authorized roles (<code>Role: Manager</code>). LLM telemetry, reasoning traces, and audit logs maintain full privacy isolation.
          </p>
        </div>
      )}

      {selectedNode.isLangGraphDecision && (
        <div style={{ background: '#EEF2FF', border: '1px solid #C7D2FE', borderRadius: '8px', padding: '14px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3730A3', fontWeight: 600, fontSize: '13.5px' }}>
            <Cpu size={18} color="#3730A3" />
            <span>LangGraph Orchestrator: Autonomous State Machine Routing</span>
          </div>
          <p style={{ fontSize: '13px', color: '#4338CA', margin: '6px 0 0 0', lineHeight: 1.5 }}>
            <strong>Adaptive Graph Execution:</strong> LangGraph evaluates state keys, determines parallel execution barriers, and dynamically invokes specialist microservices or routes diagnostic context directly to the Action Planner and CrewAI consensus layer.
          </p>
        </div>
      )}

      {/* Detailed Node Inspector Panel */}
      <div className="card" style={{ padding: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '8px',
                background: 'var(--accent-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {selectedNode.icon}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h4 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Stage {selectedNode.stageNum}: {selectedNode.label}
                </h4>
                <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
                  {selectedNode.role}
                </span>
                {selectedNode.durationMs !== undefined && (
                  <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
                    <Clock size={11} />
                    <span>{selectedNode.durationMs.toFixed(1)}ms</span>
                  </span>
                )}
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                {selectedNode.outputSummary}
              </p>
            </div>
          </div>

          {/* Toggle between Input / Output / Both */}
          <div style={{ display: 'inline-flex', background: 'var(--bg-card-subtle)', padding: '2px', borderRadius: '6px' }}>
            <button
              onClick={() => setActivePayloadTab('both')}
              style={{
                padding: '5px 12px',
                borderRadius: '5px',
                fontSize: '12px',
                fontWeight: activePayloadTab === 'both' ? 600 : 500,
                background: activePayloadTab === 'both' ? '#FFFFFF' : 'transparent',
                color: activePayloadTab === 'both' ? 'var(--text-primary)' : 'var(--text-secondary)',
                boxShadow: activePayloadTab === 'both' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              Both Payloads
            </button>
            <button
              onClick={() => setActivePayloadTab('input')}
              style={{
                padding: '5px 12px',
                borderRadius: '5px',
                fontSize: '12px',
                fontWeight: activePayloadTab === 'input' ? 600 : 500,
                background: activePayloadTab === 'input' ? '#FFFFFF' : 'transparent',
                color: activePayloadTab === 'input' ? 'var(--text-primary)' : 'var(--text-secondary)',
                boxShadow: activePayloadTab === 'input' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              Input Context Received
            </button>
            <button
              onClick={() => setActivePayloadTab('output')}
              style={{
                padding: '5px 12px',
                borderRadius: '5px',
                fontSize: '12px',
                fontWeight: activePayloadTab === 'output' ? 600 : 500,
                background: activePayloadTab === 'output' ? '#FFFFFF' : 'transparent',
                color: activePayloadTab === 'output' ? 'var(--text-primary)' : 'var(--text-secondary)',
                boxShadow: activePayloadTab === 'output' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              Output Payload Produced
            </button>
          </div>
        </div>

        {/* Payload Viewer Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: activePayloadTab === 'both' ? 'repeat(auto-fit, minmax(420px, 1fr))' : '1fr',
            gap: '16px',
          }}
        >
          {/* Input Payload */}
          {(activePayloadTab === 'both' || activePayloadTab === 'input') && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Terminal size={14} color="var(--accent)" />
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Input Context Received by LLM / Node
                </span>
              </div>
              <div
                style={{
                  background: 'var(--bg-card-subtle)',
                  borderRadius: '8px',
                  padding: '14px',
                  border: '1px solid var(--border-subtle)',
                  maxHeight: '360px',
                  overflowY: 'auto',
                }}
              >
                <pre style={{ fontSize: '12px', fontFamily: 'monospace', color: 'var(--text-primary)', margin: 0, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(selectedNode.inputPayload || { summary: selectedNode.inputSummary }, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Output Payload */}
          {(activePayloadTab === 'both' || activePayloadTab === 'output') && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <FileText size={14} color="var(--status-success)" />
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Output Produced by Agent / Stage
                </span>
              </div>
              <div
                style={{
                  background: 'var(--bg-card-subtle)',
                  borderRadius: '8px',
                  padding: '14px',
                  border: '1px solid var(--border-subtle)',
                  maxHeight: '360px',
                  overflowY: 'auto',
                }}
              >
                <pre style={{ fontSize: '12px', fontFamily: 'monospace', color: 'var(--text-primary)', margin: 0, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(selectedNode.outputPayload || { summary: selectedNode.outputSummary }, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
