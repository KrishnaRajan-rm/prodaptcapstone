import {
  AnalysisResponse,
  AgentStatusResponse,
  ScenarioItem,
  FeedbackSummary,
  EvaluationSummary,
  DashboardStats,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export class ApiClient {
  private static async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };

    try {
      const response = await fetch(url, { ...options, headers });
      if (!response.ok) {
        let errDetail = `HTTP ${response.status} ${response.statusText}`.trim();
        try {
          const errJson = await response.json();
          const detail = errJson?.detail;
          if (typeof detail === 'string') {
            errDetail = `HTTP ${response.status}: ${detail}`;
          } else if (detail && typeof detail === 'object') {
            const description = detail.message || 'Request failed';
            const causes = [detail.error, detail.cause, detail.action]
              .filter((value) => typeof value === 'string' && value.length > 0);
            if (detail.request_id) causes.unshift(`Request ID: ${detail.request_id}`);
            const issueDetails = Array.isArray(detail) ? detail : detail.errors;
            const issues = Array.isArray(issueDetails)
              ? issueDetails.map((issue: any) => {
                  const node = issue?.node ? `${issue.node}: ` : '';
                  const location = Array.isArray(issue?.loc) ? `${issue.loc.join('.')}: ` : '';
                  return `${node}${location}${issue?.error || issue?.msg || issue?.message || JSON.stringify(issue)}`;
                })
              : [];
            const warnings = Array.isArray(detail.warnings)
              ? detail.warnings.map((warning: any) => `Warning: ${typeof warning === 'string' ? warning : JSON.stringify(warning)}`)
              : [];
            errDetail = [`HTTP ${response.status}: ${description}`, ...causes, ...issues, ...warnings].join('\n');
          }
        } catch (_) {}
        throw new Error(errDetail);
      }
      return (await response.json()) as T;
    } catch (err: any) {
      console.error(`API Error on ${url}:`, err);
      throw err;
    }
  }

  static async analyze(params: {
    query: string;
    mode?: 'auto' | 'agent';
    agent?: string;
    business_id?: string;
    start_date?: string;
    end_date?: string;
    city?: string;
    viewer_role?: string;
    skip_llm_judge?: boolean;
  }): Promise<AnalysisResponse> {
    return this.request<AnalysisResponse>('/analysis', {
      method: 'POST',
      body: JSON.stringify({
        query: params.query,
        mode: params.mode || 'auto',
        agent: params.agent,
        business_id: params.business_id || undefined,
        start_date: params.start_date || undefined,
        end_date: params.end_date || undefined,
        city: params.city || undefined,
        viewer_role: params.viewer_role || 'manager',
        skip_llm_judge: params.skip_llm_judge ?? false,
      }),
    });
  }

  static async getAgentStatuses(): Promise<AgentStatusResponse> {
    return this.request<AgentStatusResponse>('/agents/status');
  }

  static async getScenarios(): Promise<ScenarioItem[]> {
    const res = await this.request<{ scenarios: ScenarioItem[] }>('/scenarios');
    return res.scenarios || [];
  }

  static async submitFeedback(payload: {
    request_id: string;
    rating: number;
    useful?: boolean;
    flags?: string;
    comment?: string;
  }): Promise<{ feedback_id: string; status: string; message: string }> {
    return this.request('/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  static async getFeedbackSummary(): Promise<FeedbackSummary> {
    return this.request<FeedbackSummary>('/feedback/summary');
  }

  static async getEvaluationSummary(): Promise<EvaluationSummary> {
    return this.request<EvaluationSummary>('/evaluation/summary');
  }

  static async getHealth(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/health');
  }

  static async getDashboardStats(): Promise<DashboardStats> {
    return this.request<DashboardStats>('/dashboard/stats');
  }
}
