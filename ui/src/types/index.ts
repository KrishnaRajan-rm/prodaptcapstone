export type NavigationTab =
  | 'overview'
  | 'analyze'
  | 'categories'
  | 'trends'
  | 'root_cause'
  | 'actions'
  | 'trace'
  | 'agents'
  | 'feedback';

export interface ExecutionTraceEvent {
  event_id: string;
  request_id: string;
  stage: string;
  component: string;
  status: 'completed' | 'failed' | 'skipped' | 'in_progress' | 'flagged' | 'blocked';
  started_at?: number;
  finished_at?: number;
  duration_ms?: number;
  input_summary: string;
  output_summary: string;
  input_payload?: any;
  output_payload?: any;
  source_ids: string[];
  warnings: string[];
  error_summary?: string;
}

export interface RecommendedAction {
  priority_rank: number;
  problem_addressed: string;
  recommended_action: string;
  rationale: string;
  owner_role: string;
  kpi: string;
  target?: string;
  source_ids: string[];
}

export interface CrewReport {
  executive_summary: string;
  key_findings: string[];
  operational_problems: string[];
  evidence_backed_causes: string[];
  prioritized_recommendations: RecommendedAction[];
  data_limitations_and_cutoffs: string[];
  evidence_source_ids: string[];
}

export interface CrewValidation {
  is_valid: boolean;
  severity: string;
  supported_claims_count: number;
  flagged_claims_count: number;
  issues: string[];
  remediation_notes: string[];
  pii_leakage_detected: boolean;
  staleness_disclosed: boolean;
}

export interface RetrievedEvidence {
  chunk_id: string;
  review_id?: string;
  business_id?: string;
  restaurant_name?: string;
  city?: string;
  state?: string;
  stars?: number;
  text_masked?: string;
  text?: string;
  aspects?: string;
  score?: number;
}

export interface AnalysisResponse {
  request_id: string;
  query: string;
  mode: 'auto' | 'agent';
  selected_agent?: string;
  business_id?: string;
  workflow_status: 'success' | 'partial_success' | 'failed';
  total_duration_ms: number;

  executive_summary?: string;
  crew_report?: CrewReport;
  crew_validation?: CrewValidation;
  crew_status?: string;

  classification?: any;
  trends?: any;
  root_cause?: any;
  action_plan?: any;

  retrieved_evidence: RetrievedEvidence[];
  quality_gate?: any;
  quality_decision?: 'passed' | 'flagged' | 'blocked';
  confidence_score?: number;
  staleness_disclosure?: string;

  agent_timings: Record<string, any>;
  stage_timings: Record<string, any>;
  execution_trace: ExecutionTraceEvent[];
  warnings: string[];
  errors: any[];
  provenance: Record<string, any>;
  pii_masked?: boolean;
  pii_decoded?: boolean;
}

export interface AgentStatusInfo {
  agent_id: string;
  name: string;
  description: string;
  port: number;
  url: string;
  status: 'available' | 'unavailable' | 'unknown';
  latency_ms?: number;
  card?: any;
  error?: string;
}

export interface AgentStatusResponse {
  agents: Record<string, AgentStatusInfo>;
  total_available: number;
  total_configured: number;
  last_checked: string;
}

export interface DashboardStats {
  total_reviews: number;
  total_complaints: number;
  average_rating?: number;
  cities_covered: number;
  first_review_month?: string;
  last_review_month?: string;
  top_complaint_category?: string;
  source: string;
}

export interface ScenarioItem {
  id: string;
  title: string;
  description: string;
  sample_query: string;
  mode: 'auto' | 'agent';
  agent?: string;
  business_id?: string;
  business_name?: string;
  city?: string;
  suggested_start_date?: string;
  suggested_end_date?: string;
}

export interface FeedbackSummary {
  total_feedback: number;
  average_rating: number;
  useful_count: number;
  not_useful_count: number;
  common_flags: Record<string, number>;
}

export interface EvaluationSummary {
  benchmark_name: string;
  status: 'operational' | 'pending' | 'unavailable';
  faithfulness_score?: number;
  relevancy_score?: number;
  passed: boolean;
  test_cases_evaluated: number;
  evaluated_at: string;
  details: Record<string, any>;
}
